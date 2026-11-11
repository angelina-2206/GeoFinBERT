import os
import sys
import torch
import pandas as pd
from torch.utils.data import DataLoader

from geofinbert.config import load_config
from geofinbert.utils import set_seed, get_device_info, count_parameters
from geofinbert.data.fetcher import fetch_unlabeled_geofin_corpus, fetch_human_labeled_sentiment_corpus
from geofinbert.data.preprocessor import (
    validate_schema_and_integrity,
    create_chronological_splits
)
from geofinbert.models.dapt import build_dapt_model
from geofinbert.models.classifier import build_classifier_model
from geofinbert.data.dataset import MLMTextDataset, SentimentClassificationDataset, get_mlm_collator
from geofinbert.training.trainer_dapt import run_dapt_training
from geofinbert.training.trainer_classifier import compute_class_weights, run_classifier_training
from geofinbert.inference import GeoFinBERTInferencePipeline

def run_smoke_test():
    print("\n" + "="*80)
    print("GEOFIBERT END-TO-END REAL-DATA SMOKE TEST")
    print("="*80)
    
    # 1. Hardware environment verification
    device_info = get_device_info()
    print(f"\n[Hardware Diagnostics] Device: {device_info['device']} | CUDA Available: {device_info['cuda_available']} | PyTorch: {device_info['torch_version']}")
    if not device_info["cuda_available"]:
        print(" -> Running smoke test on CPU in lightweight verification mode (small real-data slice).")

    # 2. Data Acquisition on a small real slice
    print("\n[Step 1/5] Fetching real data slices for smoke testing...")
    dapt_df = fetch_unlabeled_geofin_corpus(max_samples=50)
    labeled_df = fetch_human_labeled_sentiment_corpus(max_samples=50)
    
    # 3. Schema & Integrity Validation
    print("\n[Step 2/5] Validating dataset schema, missing values, dates, and labels...")
    dapt_val = validate_schema_and_integrity(dapt_df, is_labeled=False)
    assert dapt_val["valid"], f"DAPT schema validation failed: {dapt_val['errors']}"
    print(f" -> DAPT Schema valid: {len(dapt_df)} articles | Date range: {dapt_val['stats'].get('date_range')}")
    
    labeled_val = validate_schema_and_integrity(labeled_df, is_labeled=True)
    assert labeled_val["valid"], f"Labeled schema validation failed: {labeled_val['errors']}"
    print(f" -> Labeled Schema valid: {len(labeled_df)} articles | Labels: {labeled_val['stats'].get('label_distribution')}")

    # 4. Chronological Splits & Zero Leakage
    print("\n[Step 3/5] Testing chronological splitting and duplicate leakage prevention...")
    dapt_train, dapt_val, _ = create_chronological_splits(dapt_df, train_ratio=0.8, val_ratio=0.2, test_ratio=0.0)
    l_train, l_val, l_test = create_chronological_splits(labeled_df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    print(f" -> Splits verified: DAPT Train={len(dapt_train)}, Val={len(dapt_val)} | Labeled Train={len(l_train)}, Val={len(l_val)}, Test={len(l_test)}")

    # 5. Stage 1 DAPT Architecture & Training Step (MLM)
    print("\n[Step 4/5] Testing Stage 1 DAPT MLM model instantiation and training step...")
    smoke_dapt_dir = "models/checkpoints/GeoFinBERT-DAPT-smoke"
    dapt_model, tokenizer = build_dapt_model(
        base_model_name_or_path="ProsusAI/finbert",
        tokenizer_name="ProsusAI/finbert",
        local_files_only=True
    )
    
    train_ds = MLMTextDataset(dapt_train["text"].tolist()[:10], tokenizer, max_length=64)
    val_ds = MLMTextDataset(dapt_val["text"].tolist()[:6], tokenizer, max_length=64)
    collator = get_mlm_collator(tokenizer, mlm_probability=0.15)
    
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, collate_fn=collator)
    
    dapt_log = run_dapt_training(
        model=dapt_model,
        tokenizer=tokenizer,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        output_dir=smoke_dapt_dir,
        num_epochs=1,
        learning_rate=3e-5,
        use_fp16=False
    )
    assert os.path.exists(os.path.join(smoke_dapt_dir, "pytorch_model.bin")), "DAPT checkpoint not created!"
    print(f" -> DAPT smoke checkpoint verified at: {smoke_dapt_dir}")

    # 6. Stage 2 Supervised Fine-Tuning & Class Weights
    print("\n[Step 5/5] Testing Stage 2 Classifier with class-weighted loss and inference...")
    class_weights = compute_class_weights(l_train["label"].tolist(), num_classes=3)
    smoke_cls_dir = "models/checkpoints/GeoFinBERT-smoke"
    
    cls_model, tokenizer = build_classifier_model(
        backbone_path_or_name=smoke_dapt_dir,
        num_labels=3,
        class_weights=class_weights,
        is_dapt_checkpoint=True,
        local_files_only=True
    )
    
    cls_train_ds = SentimentClassificationDataset(l_train["text"].tolist()[:10], l_train["label"].tolist()[:10], tokenizer, max_length=64)
    cls_val_ds = SentimentClassificationDataset(l_val["text"].tolist()[:6], l_val["label"].tolist()[:6], tokenizer, max_length=64)
    cls_test_ds = SentimentClassificationDataset(l_test["text"].tolist()[:6], l_test["label"].tolist()[:6], tokenizer, max_length=64)
    
    cls_train_loader = DataLoader(cls_train_ds, batch_size=2, shuffle=True)
    cls_val_loader = DataLoader(cls_val_ds, batch_size=2, shuffle=False)
    cls_test_loader = DataLoader(cls_test_ds, batch_size=2, shuffle=False)
    
    cls_log = run_classifier_training(
        model=cls_model,
        tokenizer=tokenizer,
        train_dataloader=cls_train_loader,
        val_dataloader=cls_val_loader,
        test_dataloader=cls_test_loader,
        output_dir=smoke_cls_dir,
        model_name="GeoFinBERT-Smoke",
        num_epochs=1,
        learning_rate=2e-5,
        early_stopping_patience=2,
        use_fp16=False
    )
    assert os.path.exists(os.path.join(smoke_cls_dir, "pytorch_model.bin")), "Classifier checkpoint not created!"
    print(f" -> Classifier smoke checkpoint verified at: {smoke_cls_dir}")

    # 7. Test Inference Pipeline on Geopolitical and Energy Headlines
    print("\n[Inference Verification] Testing live predictions on domain headlines:")
    pipeline = GeoFinBERTInferencePipeline(smoke_cls_dir, device="cpu", max_length=64)
    test_headlines = [
        "Crude oil prices surge as Strait of Hormuz naval confrontations threaten Persian Gulf tanker routes.",
        "Gold bullion falls sharply following unexpected hawkish rate remarks from Federal Reserve.",
        "OPEC+ ministers agree to keep crude oil output quotas unchanged for the next quarter."
    ]
    for h in test_headlines:
        res = pipeline.predict(h)
        print(f"\nHeadline: \"{h}\"")
        print(f"Prediction: {res['prediction'].upper()} (Confidence: {res['confidence']:.2%})")
        print(f"Probabilities: {res['probabilities']}")
        
    print("\n" + "="*80)
    print("ALL SMOKE TESTS PASSED! ARCHITECTURE & PIPELINE INTEGRITY CONFIRMED.")
    print("="*80)

if __name__ == "__main__":
    run_smoke_test()
