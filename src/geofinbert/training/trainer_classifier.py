import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from geofinbert.models.classifier import GeoFinBERTForSequenceClassification
from geofinbert.evaluation.metrics import compute_classification_metrics
from geofinbert.utils import format_time, count_parameters

def compute_class_weights(labels: List[int], num_classes: int = 3) -> torch.Tensor:
    """
    Compute balanced inverse-frequency class weights:
    w_c = N / (K * N_c)
    """
    labels = np.array(labels)
    total_samples = len(labels)
    weights = []
    
    for c in range(num_classes):
        count = np.sum(labels == c)
        if count > 0:
            w = total_samples / (num_classes * count)
        else:
            w = 1.0
        weights.append(w)
        
    weights_tensor = torch.tensor(weights, dtype=torch.float)
    # Normalize weights so mean is 1.0
    weights_tensor = weights_tensor / weights_tensor.mean()
    print(f"[Trainer] Calculated class weights: positive={weights_tensor[0]:.3f}, negative={weights_tensor[1]:.3f}, neutral={weights_tensor[2]:.3f}")
    return weights_tensor

def train_classifier_epoch(
    model: GeoFinBERTForSequenceClassification,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    device: torch.device,
    use_amp: bool = False
) -> float:
    """Train single epoch of sequence classification."""
    model.train()
    total_loss = 0.0
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None
    
    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        token_type_ids = batch.get("token_type_ids", None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
            
        optimizer.zero_grad()
        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )
                loss = outputs["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        if scheduler:
            scheduler.step()
            
        total_loss += loss.item()
        
    return total_loss / max(len(dataloader), 1)

@torch.no_grad()
def evaluate_classifier(
    model: GeoFinBERTForSequenceClassification,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[float, Dict[str, Any]]:
    """Evaluate classification model on dataloader."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        token_type_ids = batch.get("token_type_ids", None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
            
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )
        total_loss += outputs["loss"].item()
        preds = torch.argmax(outputs["logits"], dim=-1).cpu().numpy().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy().tolist())
        
    avg_loss = total_loss / max(len(dataloader), 1)
    metrics = compute_classification_metrics(all_labels, all_preds)
    metrics["eval_loss"] = round(avg_loss, 4)
    return avg_loss, metrics

def run_classifier_training(
    model: GeoFinBERTForSequenceClassification,
    tokenizer: BertTokenizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    test_dataloader: Optional[DataLoader] = None,
    output_dir: str = "models/checkpoints/GeoFinBERT",
    model_name: str = "GeoFinBERT",
    num_epochs: int = 4,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    early_stopping_patience: int = 3,
    use_fp16: bool = False
) -> Dict[str, Any]:
    """
    Complete training and evaluation pipeline for Stage 2 Supervised Fine-Tuning.
    Executes early stopping on validation Macro F1, logs metrics, and tests on the test partition.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = use_fp16 and (device.type == "cuda")
    model = model.to(device)
    
    total_params, trainable_params = count_parameters(model)
    print(f"\n{'='*70}")
    print(f"STAGE 2: SUPERVISED FINE-TUNING ({model_name})")
    print(f"Device: {device} | Total Params: {total_params:,} | Trainable: {trainable_params:,}")
    print(f"Epochs: {num_epochs} | LR: {learning_rate} | Early Stopping Patience: {early_stopping_patience}")
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
    best_val_macro_f1 = -1.0
    patience_counter = 0
    history = []
    
    for epoch in range(1, num_epochs + 1):
        ep_start = time.time()
        train_loss = train_classifier_epoch(model, train_dataloader, optimizer, scheduler, device, use_amp=use_amp)
        val_loss, val_metrics = evaluate_classifier(model, val_dataloader, device)
        macro_f1 = val_metrics["macro_f1"]
        accuracy = val_metrics["accuracy"]
        ep_duration = time.time() - ep_start
        
        print(f"Epoch {epoch}/{num_epochs} ({format_time(ep_duration)}) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {accuracy:.4f} | Val Macro F1: {macro_f1:.4f}")
        
        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": val_loss,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": val_metrics["weighted_f1"],
            "epoch_duration_seconds": round(ep_duration, 2)
        }
        history.append(record)
        
        # Checkpoint based on Macro F1 or epoch 1
        if macro_f1 >= best_val_macro_f1 or epoch == 1:
            best_val_macro_f1 = macro_f1
            patience_counter = 0
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
            print(f" -> Best checkpoint saved to {output_dir} (Macro F1: {macro_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"[Early Stopping] No improvement for {early_stopping_patience} epochs. Halting training.")
                break
                
    total_duration = time.time() - start_time
    print(f"\n[{model_name} Training Completed] Total Duration: {format_time(total_duration)} | Best Val Macro F1: {best_val_macro_f1:.4f}")
    
    # Reload best model for test set evaluation
    test_metrics = None
    if test_dataloader is not None:
        print(f"\n[Evaluation] Evaluating best {model_name} on unseen test partition...")
        best_model = GeoFinBERTForSequenceClassification.from_pretrained(output_dir)
        best_model = best_model.to(device)
        test_loss, test_metrics = evaluate_classifier(best_model, test_dataloader, device)
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f} | Test Macro F1: {test_metrics['macro_f1']:.4f} | Test Weighted F1: {test_metrics['weighted_f1']:.4f}")
        
    log_data = {
        "model_name": model_name,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "total_training_time_seconds": round(total_duration, 2),
        "total_training_time_formatted": format_time(total_duration),
        "device": str(device),
        "best_val_macro_f1": best_val_macro_f1,
        "test_metrics": test_metrics,
        "history": history
    }
    
    with open(os.path.join(output_dir, "training_log.json"), "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)
        
    return log_data
