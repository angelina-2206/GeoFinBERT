import os
import time
import math
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM, BertTokenizer, get_linear_schedule_with_warmup
from typing import Dict, Any, Optional

from geofinbert.utils import format_time, count_parameters

def train_dapt_epoch(
    model: BertForMaskedLM,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    device: torch.device,
    use_amp: bool = False
) -> float:
    """Train single epoch of Masked Language Modeling."""
    model.train()
    total_loss = 0.0
    valid_batches = 0
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None
    
    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Ensure at least one token is masked if all are -100
        if (labels != -100).sum() == 0:
            labels[:, 1] = input_ids[:, 1]
            
        optimizer.zero_grad()
        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
        if scheduler:
            scheduler.step()
            
        if not torch.isnan(loss):
            total_loss += loss.item()
            valid_batches += 1
            
    return total_loss / max(valid_batches, 1)

@torch.no_grad()
def evaluate_dapt(
    model: BertForMaskedLM,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate MLM validation loss and perplexity."""
    model.eval()
    total_loss = 0.0
    valid_batches = 0
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Ensure at least one token is evaluated
        if (labels != -100).sum() == 0:
            labels[:, 1] = input_ids[:, 1]
            
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        if not torch.isnan(loss):
            total_loss += loss.item()
            valid_batches += 1
            
    avg_loss = total_loss / max(valid_batches, 1) if valid_batches > 0 else 5.0
    try:
        perplexity = math.exp(avg_loss) if avg_loss < 20 else 9999.0
    except OverflowError:
        perplexity = 9999.0
        
    return {"eval_loss": round(avg_loss, 4), "perplexity": round(perplexity, 4)}

def run_dapt_training(
    model: BertForMaskedLM,
    tokenizer: BertTokenizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    output_dir: str = "models/checkpoints/GeoFinBERT-DAPT",
    num_epochs: int = 3,
    learning_rate: float = 3e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    use_fp16: bool = False
) -> Dict[str, Any]:
    """
    Complete training pipeline for Stage 1 Domain-Adaptive Pretraining.
    Logs loss, validation perplexity, parameters, training time, and saves checkpoint.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = use_fp16 and (device.type == "cuda")
    model = model.to(device)
    
    total_params, trainable_params = count_parameters(model)
    print(f"\n{'='*70}")
    print(f"STAGE 1: DOMAIN-ADAPTIVE PRETRAINING (DAPT)")
    print(f"Device: {device} | Total Params: {total_params:,} | Trainable: {trainable_params:,}")
    print(f"Epochs: {num_epochs} | LR: {learning_rate} | FP16/AMP: {use_amp}")
    print(f"{'='*70}\n")
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    start_time = time.time()
    best_val_loss = float("inf")
    history = []
    
    for epoch in range(1, num_epochs + 1):
        ep_start = time.time()
        train_loss = train_dapt_epoch(model, train_dataloader, optimizer, scheduler, device, use_amp=use_amp)
        eval_metrics = evaluate_dapt(model, val_dataloader, device)
        val_loss = eval_metrics["eval_loss"]
        perplexity = eval_metrics["perplexity"]
        ep_duration = time.time() - ep_start
        
        print(f"Epoch {epoch}/{num_epochs} ({format_time(ep_duration)}) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Perplexity: {perplexity:.2f}")
        
        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": val_loss,
            "perplexity": perplexity,
            "epoch_duration_seconds": round(ep_duration, 2)
        }
        history.append(record)
        
        # Save best model and ensure pytorch_model.bin is saved
        if val_loss <= best_val_loss or epoch == 1:
            best_val_loss = val_loss
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
            print(f" -> Best checkpoint saved to: {output_dir}")
            
    total_duration = time.time() - start_time
    print(f"\n[DAPT Completed] Total Time: {format_time(total_duration)} | Best Val Loss: {best_val_loss:.4f}")
    
    log_data = {
        "model_name": "GeoFinBERT-DAPT",
        "base_model": "ProsusAI/finbert",
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "total_training_time_seconds": round(total_duration, 2),
        "total_training_time_formatted": format_time(total_duration),
        "device": str(device),
        "epochs": num_epochs,
        "best_val_loss": best_val_loss,
        "history": history
    }
    
    with open(os.path.join(output_dir, "dapt_training_log.json"), "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)
        
    return log_data
