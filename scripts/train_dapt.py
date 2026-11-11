import os
import argparse
import pandas as pd
from torch.utils.data import DataLoader

from geofinbert.config import load_config
from geofinbert.utils import set_seed, get_device_info
from geofinbert.models.dapt import build_dapt_model
from geofinbert.data.dataset import MLMTextDataset, get_mlm_collator
from geofinbert.training.trainer_dapt import run_dapt_training

def main():
    parser = argparse.ArgumentParser(description="GeoFinBERT Stage 1: Domain-Adaptive Pretraining")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Max training samples (for hardware calibration)")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Max validation samples")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("project", {}).get("seed", 42))
    
    device_info = get_device_info()
    print(f"[Hardware Check] {device_info}")
    
    # Load chronological splits
    splits_dir = config["data"]["splits_dir"]
    train_path = os.path.join(splits_dir, "dapt_train.parquet")
    val_path = os.path.join(splits_dir, "dapt_val.parquet")
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(f"DAPT split files not found in {splits_dir}. Run scripts/prepare_data.py first.")
        
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    
    if args.max_train_samples and len(train_df) > args.max_train_samples:
        train_df = train_df.iloc[:args.max_train_samples]
    if args.max_val_samples and len(val_df) > args.max_val_samples:
        val_df = val_df.iloc[:args.max_val_samples]
        
    print(f"[Data] Loaded DAPT training: {len(train_df):,} articles | Validation: {len(val_df):,} articles")
    
    # Build architecture transferring FinBERT encoder weights
    base_model_name = config["model"]["base_model_name"]
    model, tokenizer = build_dapt_model(
        base_model_name_or_path=base_model_name,
        tokenizer_name=config["model"]["tokenizer_name"],
        local_files_only=True
    )
    
    max_seq_len = config["model"].get("max_seq_length", 128)
    train_dataset = MLMTextDataset(train_df["text"].tolist(), tokenizer, max_length=max_seq_len)
    val_dataset = MLMTextDataset(val_df["text"].tolist(), tokenizer, max_length=max_seq_len)
    
    collator = get_mlm_collator(tokenizer, mlm_probability=config["stage1_dapt"].get("mlm_probability", 0.15))
    
    dapt_cfg = config["stage1_dapt"]
    batch_size = args.batch_size or dapt_cfg.get("per_device_train_batch_size", 16)
    eval_batch_size = dapt_cfg.get("per_device_eval_batch_size", 32)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0
    )
    
    output_dir = args.output_dir or dapt_cfg.get("output_dir", "models/checkpoints/GeoFinBERT-DAPT")
    epochs = args.epochs or dapt_cfg.get("num_train_epochs", 3)
    
    log = run_dapt_training(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        output_dir=output_dir,
        num_epochs=epochs,
        learning_rate=float(dapt_cfg.get("learning_rate", 3e-5)),
        weight_decay=float(dapt_cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(dapt_cfg.get("warmup_ratio", 0.1)),
        use_fp16=dapt_cfg.get("fp16", True) and device_info["cuda_available"]
    )
    print("\nStage 1 DAPT Training finished successfully.")

if __name__ == "__main__":
    main()
