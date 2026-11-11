import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

# FinBERT standard label IDs
TARGET_NAMES = ["positive", "negative", "neutral"]

def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    target_names: List[str] = TARGET_NAMES
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics for academic reporting.
    Returns accuracy, macro F1, weighted F1, per-class metrics, and confusion matrix.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
    
    clf_report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    
    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "confusion_matrix": cm,
        "per_class": {
            cls_name: {
                "precision": round(clf_report[cls_name]["precision"], 4),
                "recall": round(clf_report[cls_name]["recall"], 4),
                "f1_score": round(clf_report[cls_name]["f1-score"], 4),
                "support": int(clf_report[cls_name]["support"])
            }
            for cls_name in target_names if cls_name in clf_report
        }
    }

def format_metrics_table(baseline_metrics: Dict[str, Any], adapted_metrics: Dict[str, Any]) -> str:
    """Generate Markdown comparison table for the research report."""
    md = []
    md.append("| Metric | Baseline FinBERT | GeoFinBERT (Adapted) | Absolute Δ | Relative Δ (%) |")
    md.append("| :--- | :---: | :---: | :---: | :---: |")
    
    scalar_keys = [
        ("Accuracy", "accuracy"),
        ("Macro F1", "macro_f1"),
        ("Weighted F1", "weighted_f1"),
        ("Macro Precision", "macro_precision"),
        ("Macro Recall", "macro_recall"),
    ]
    
    for label, key in scalar_keys:
        b_val = baseline_metrics.get(key, 0.0)
        a_val = adapted_metrics.get(key, 0.0)
        diff = a_val - b_val
        rel = (diff / b_val * 100) if b_val > 0 else 0.0
        sign = "+" if diff >= 0 else ""
        md.append(f"| **{label}** | {b_val:.4f} | {a_val:.4f} | {sign}{diff:.4f} | {sign}{rel:.2f}% |")
        
    md.append("")
    md.append("### Per-Class F1 Comparison")
    md.append("| Class | Baseline FinBERT F1 | GeoFinBERT F1 | Δ F1 |")
    md.append("| :--- | :---: | :---: | :---: |")
    
    for cls_name in TARGET_NAMES:
        b_f1 = baseline_metrics.get("per_class", {}).get(cls_name, {}).get("f1_score", 0.0)
        a_f1 = adapted_metrics.get("per_class", {}).get(cls_name, {}).get("f1_score", 0.0)
        diff = a_f1 - b_f1
        sign = "+" if diff >= 0 else ""
        md.append(f"| **{cls_name.capitalize()}** | {b_f1:.4f} | {a_f1:.4f} | {sign}{diff:.4f} |")
        
    return "\n".join(md)

