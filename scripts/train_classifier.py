import os
import argparse
import pandas as pd
from torch.utils.data import DataLoader

from geofinbert.config import load_config
from geofinbert.utils import set_seed, get_device_info
from geofinbert.models.classifier import build_classifier_model
from geofinbert.data.dataset import SentimentClassificationDataset
from geofinbert.training.trainer_classifier import compute_class_weights, run_classifier_training

def train_single_classifier(
    model_type: str,
    config: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    class_weights,
    epochs: int,
    batch_size: int,
    device_info: dict
):
    """Train either baseline FinBERT or GeoFinBERT-DAPT adapted classifier."""
    ft_cfg = config["stage2_finetuning"]
    max_seq_len = config["model"].get("max_seq_length", 128)
    
    if model_type == "baseline":
        model_name = "FinBERT-Baseline"
        backbone = config["model"]["base_model_name"]
        is_dapt = False
        output_dir = ft_cfg.get("baseline_output_dir", "models/checkpoints/FinBERT-Baseline")
    else:
        model_name = "GeoFinBERT"
        backbone = config["stage1_dapt"].get("output_dir", "models/checkpoints/GeoFinBERT-DAPT")
        is_dapt = True
        output_dir = ft_cfg.get("adapted_output_dir", "models/checkpoints/GeoFinBERT")
        
    model, tokenizer = build_classifier_model(
        backbone_path_or_name=backbone,
        num_labels=config["model"].get("num_labels", 3),
        class_weights=class_weights,
        is_dapt_checkpoint=is_dapt,
        local_files_only=True
    )
    
    train_dataset = SentimentClassificationDataset(
        train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, max_length=max_seq_len
    )
    val_dataset = SentimentClassificationDataset(
        val_df["text"].tolist(), val_df["label"].tolist(), tokenizer, max_length=max_seq_len
    )
    test_dataset = SentimentClassificationDataset(
        test_df["text"].tolist(), test_df["label"].tolist(), tokenizer, max_length=max_seq_len
    )
    
    eval_batch_size = ft_cfg.get("per_device_eval_batch_size", 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=0)
    
    log = run_classifier_training(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        output_dir=output_dir,
        model_name=model_name,
        num_epochs=epochs,
        learning_rate=float(ft_cfg.get("learning_rate", 2e-5)),
        weight_decay=float(ft_cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(ft_cfg.get("warmup_ratio", 0.1)),
        early_stopping_patience=ft_cfg.get("early_stopping_patience", 3),
        use_fp16=ft_cfg.get("fp16", True) and device_info["cuda_available"]
    )
    return log

def main():
    parser = argparse.ArgumentParser(description="GeoFinBERT Stage 2: Supervised Sentiment Fine-Tuning")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--model_type", type=str, choices=["baseline", "adapted", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Max train samples")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Max val samples")
    parser.add_argument("--max_test_samples", type=int, default=None, help="Max test samples")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("project", {}).get("seed", 42))
    
    device_info = get_device_info()
    print(f"[Hardware Check] {device_info}")
    
    # Load strictly chronological labeled splits
    splits_dir = config["data"]["splits_dir"]
    train_df = pd.read_parquet(os.path.join(splits_dir, "sentiment_train.parquet"))
    val_df = pd.read_parquet(os.path.join(splits_dir, "sentiment_val.parquet"))
    test_df = pd.read_parquet(os.path.join(splits_dir, "sentiment_test.parquet"))
    
    if args.max_train_samples and len(train_df) > args.max_train_samples:
        train_df = train_df.iloc[:args.max_train_samples]
    if args.max_val_samples and len(val_df) > args.max_val_samples:
        val_df = val_df.iloc[:args.max_val_samples]
    if args.max_test_samples and len(test_df) > args.max_test_samples:
        test_df = test_df.iloc[:args.max_test_samples]
        
    print(f"[Data] Labeled Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    
    # Compute class weights on train partition
    class_weights = None
    if config["stage2_finetuning"].get("use_class_weights", True):
        class_weights = compute_class_weights(train_df["label"].tolist(), num_classes=3)
        
    epochs = args.epochs or config["stage2_finetuning"].get("num_train_epochs", 4)
    batch_size = args.batch_size or config["stage2_finetuning"].get("per_device_train_batch_size", 16)
    
    if args.model_type in ["baseline", "both"]:
        print("\n>>> TRAINING BASELINE FINBERT CLASSIFIER <<<")
        train_single_classifier("baseline", config, train_df, val_df, test_df, class_weights, epochs, batch_size, device_info)
        
    if args.model_type in ["adapted", "both"]:
        print("\n>>> TRAINING ADAPTED GEOFINBERT CLASSIFIER <<<")
        train_single_classifier("adapted", config, train_df, val_df, test_df, class_weights, epochs, batch_size, device_info)
        
    print("\nStage 2 Fine-Tuning completed successfully.")

if __name__ == "__main__":
    main()
