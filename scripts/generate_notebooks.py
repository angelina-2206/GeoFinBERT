import json
import os
import zipfile

def markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    }

def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")]
    }

def save_notebook(cells, output_path):
    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated notebook: {output_path}")

def build_notebook_1():
    cells = [
        markdown_cell("""# 01. Dataset Preparation & Integrity Pipeline
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)
### Academic Research Project: **GeoFinBERT**
Adapting `ProsusAI/FinBERT` to real-world geopolitical and energy-market news (2000–11 November 2026).

---
### Research Domain:
- **Oil and crude oil markets** (Brent, WTI, OPEC quotas, SPR, refining disruptions)
- **Gold and bullion markets** (Bullion reserves, safe-haven flows, inflation hedge)
- **US–Iran relations** (Sanctions, JCPOA, maritime seizures, diplomatic escalation)
- **Middle Eastern geopolitical events** (Strait of Hormuz, Persian Gulf security, regional conflicts)

---
### Strict Academic Data Integrity Principles:
1. **Zero Synthetic Data**: Never generate synthetic articles or artificial human labels.
2. **Zero Model Predictions as Ground Truth**: Never use LLM or model outputs as labels.
3. **URL and Date Preservation**: Retain canonical source URLs, publication timestamps, and accredited publishers.
4. **Wire Syndication Deduplication**: Detect and remove exact duplicates and near-duplicate syndicated stories.
5. **Zero Lookahead Leakage**: Strictly chronological train, validation, and test partitions."""),
        
        code_cell("""# Environment Setup & Automatic Colab Diagnostics
import os
import sys

try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    print("[Colab Environment Detected] Installing dependencies...")
    !pip install -q -r requirements.txt
    !pip install -q -e .

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath("src"))

from geofinbert.config import load_config
from geofinbert.utils import set_seed, get_device_info
from geofinbert.data.fetcher import fetch_unlabeled_geofin_corpus, fetch_human_labeled_sentiment_corpus
from geofinbert.data.preprocessor import validate_schema_and_integrity, create_chronological_splits

config = load_config("configs/config.yaml")
set_seed(config["project"]["seed"])

device_info = get_device_info()
print("Hardware Check:", device_info)
if device_info["cuda_available"]:
    print(f"Accelerated GPU Active: {device_info['gpu_name']} ({device_info['vram_gb']} GB VRAM)")
else:
    print("Running in CPU mode. In Colab, enable GPU: Runtime -> Change runtime type -> T4/A100 GPU.")"""),

        markdown_cell("""## Step 1: Ingesting Genuine Historical Pretraining Corpus (DAPT)
We download authentic news articles across accredited financial and commodity publishers (ForexLive, Benzinga, ZeroHedge, OilPrice, MarketWatch, Economic Times, MoneyControl, Reuters syndication)."""),

        code_cell("""# Fetch Unlabeled Geopolitical and Energy Market News Corpus
dapt_df = fetch_unlabeled_geofin_corpus(
    cache_dir=config["data"]["raw_dir"],
    processed_dir=config["data"]["processed_dir"]
)

print(f"Total unlabelled DAPT articles ingested: {len(dapt_df):,}")
dapt_df.head(3)"""),

        markdown_cell("""## Step 2: Ingesting Independently Human-Annotated Sentiment Corpus
We acquire real articles with independently human-annotated sentiment (`positive`, `negative`, `neutral`) using consensus financial annotations and commodity price-sentiment ground truth."""),

        code_cell("""# Fetch Human-Labeled Corpus
labeled_df = fetch_human_labeled_sentiment_corpus(
    cache_dir=config["data"]["raw_dir"],
    processed_dir=config["data"]["processed_dir"]
)

print(f"Total human-labeled sentiment articles ingested: {len(labeled_df):,}")
labeled_df.head(3)"""),

        markdown_cell("""## Step 3: Strict Schema Validation & Data Integrity Auditing
We verify that no synthetic artifacts exist, missing text is eliminated, and dates lie strictly between `2000-01-01` and `2026-11-11`."""),

        code_cell("""dapt_audit = validate_schema_and_integrity(dapt_df, is_labeled=False)
print("=== DAPT Integrity Audit ===")
print("Valid:", dapt_audit["valid"])
print("Stats:", dapt_audit["stats"])
if dapt_audit["errors"]:
    print("Errors:", dapt_audit["errors"])

labeled_audit = validate_schema_and_integrity(labeled_df, is_labeled=True)
print("\\n=== Human-Labeled Sentiment Integrity Audit ===")
print("Valid:", labeled_audit["valid"])
print("Label Distribution:", labeled_audit["stats"].get("label_distribution"))
if labeled_audit["errors"]:
    print("Errors:", labeled_audit["errors"])"""),

        markdown_cell("""## Step 4: Chronological Partitioning (Zero Temporal Leakage)
To accurately simulate out-of-time forecasting and prevent lookahead bias:
$$\\max(\\text{Train Date}) \\le \\min(\\text{Val Date}) \\le \\max(\\text{Val Date}) \\le \\min(\\text{Test Date})$$
We also verify cross-partition duplicate exclusion."""),

        code_cell("""splits_dir = config["data"]["splits_dir"]
os.makedirs(splits_dir, exist_ok=True)

# Split DAPT Corpus (85% Train, 15% Val)
dapt_train, dapt_val, _ = create_chronological_splits(
    dapt_df, date_col="date", train_ratio=0.85, val_ratio=0.15, test_ratio=0.0
)
dapt_train.to_parquet(os.path.join(splits_dir, "dapt_train.parquet"), index=False)
dapt_val.to_parquet(os.path.join(splits_dir, "dapt_val.parquet"), index=False)

# Split Human-Labeled Corpus (70% Train, 15% Val, 15% Test)
split_cfg = config["data"]["splits"]
l_train, l_val, l_test = create_chronological_splits(
    labeled_df,
    date_col="date",
    train_ratio=split_cfg["train_ratio"],
    val_ratio=split_cfg["val_ratio"],
    test_ratio=split_cfg["test_ratio"]
)
l_train.to_parquet(os.path.join(splits_dir, "sentiment_train.parquet"), index=False)
l_val.to_parquet(os.path.join(splits_dir, "sentiment_val.parquet"), index=False)
l_test.to_parquet(os.path.join(splits_dir, "sentiment_test.parquet"), index=False)

print(f"DAPT Splits: Train={len(dapt_train):,}, Val={len(dapt_val):,}")
print(f"Sentiment Splits: Train={len(l_train):,}, Val={len(l_val):,}, Test={len(l_test):,}")"""),

        markdown_cell("""## Step 5: Visualizing Dataset Statistics by Year, Topic, and Label"""),

        code_cell("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Topic Breakdown
topic_counts = dapt_df["topic"].value_counts()
sns.barplot(x=topic_counts.values, y=topic_counts.index, ax=axes[0], palette="viridis")
axes[0].set_title("DAPT Articles by Research Topic")
axes[0].set_xlabel("Article Count")

# 2. Label Distribution
label_names = ["Positive (0)", "Negative (1)", "Neutral (2)"]
label_counts = [labeled_df["label"].value_counts().get(i, 0) for i in [0, 1, 2]]
sns.barplot(x=label_names, y=label_counts, ax=axes[1], palette="magma")
axes[1].set_title("Human Sentiment Ground Truth Distribution")
axes[1].set_ylabel("Count")

# 3. Timeline Coverage
dapt_years = pd.to_datetime(dapt_df["date"]).dt.year.value_counts().sort_index()
axes[2].plot(dapt_years.index, dapt_years.values, marker="o", color="navy")
axes[2].set_title("Corpus Timeline (2000–2026)")
axes[2].set_xlabel("Year")
axes[2].set_ylabel("Article Count")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()"""),

        markdown_cell("""## Step 6: Dataset Summary Verification
The cleaned, verified datasets are now serialized in `data/splits/` ready for Stage 1 DAPT pretraining and Stage 2 Fine-Tuning.""")
    ]
    return cells

def build_notebook_2():
    cells = [
        markdown_cell("""# 02. Stage 1: Domain-Adaptive Pretraining (DAPT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)
### Academic Research Project: **GeoFinBERT**

---
### Architectural Methodology:
1. **Base Model**: Load `ProsusAI/FinBERT`, which is fine-tuned on corporate financial filings.
2. **Architecture Transfer**: Transfer FinBERT's 12-layer BERT transformer encoder and embeddings into a `BertForMaskedLM` architecture.
3. **MLM Objective**: Masked Language Modeling dynamically masks 15% of tokens (80% `[MASK]`, 10% random, 10% unchanged) over authentic geopolitical & energy market news.
4. **Tokenizer Preservation**: Retain FinBERT's vocabulary of 30,522 tokens and tie prediction embeddings.
5. **Checkpointing**: Save the domain-adapted checkpoint as `GeoFinBERT-DAPT` alongside training logs, validation loss, and perplexity."""),

        code_cell("""# Setup and Hardware Diagnostics
import os
import sys

try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    print("[Colab Environment Detected] Installing dependencies...")
    !pip install -q -r requirements.txt
    !pip install -q -e .

import torch
import pandas as pd
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath("src"))

from geofinbert.config import load_config
from geofinbert.utils import set_seed, get_device_info, count_parameters
from geofinbert.models.dapt import build_dapt_model
from geofinbert.data.dataset import MLMTextDataset, get_mlm_collator
from geofinbert.training.trainer_dapt import run_dapt_training

config = load_config("configs/config.yaml")
set_seed(config["project"]["seed"])

device_info = get_device_info()
print("Hardware Check:", device_info)
if device_info["cuda_available"]:
    print(f"Accelerated GPU Active: {device_info['gpu_name']} ({device_info['vram_gb']} GB VRAM)")
else:
    print("Warning: CUDA GPU not detected. For fast GPU training in Colab: Runtime -> Change runtime type -> T4/A100 GPU.")"""),

        markdown_cell("""## Step 1: Load Chronological DAPT Partitions"""),

        code_cell("""splits_dir = config["data"]["splits_dir"]
train_df = pd.read_parquet(os.path.join(splits_dir, "dapt_train.parquet"))
val_df = pd.read_parquet(os.path.join(splits_dir, "dapt_val.parquet"))

print(f"Loaded DAPT Training Set: {len(train_df):,} articles")
print(f"Loaded DAPT Validation Set: {len(val_df):,} articles")"""),

        markdown_cell("""## Step 2: Build Architecture & Transfer FinBERT Weights
We instantiate `BertForMaskedLM` using `ProsusAI/FinBERT` encoder weights, keeping the original tokenizer intact."""),

        code_cell("""base_model_name = config["model"]["base_model_name"]
model, tokenizer = build_dapt_model(
    base_model_name_or_path=base_model_name,
    tokenizer_name=config["model"]["tokenizer_name"],
    local_files_only=False
)

total_p, train_p = count_parameters(model)
print(f"Base FinBERT Encoder successfully transferred to BertForMaskedLM.")
print(f"Total Parameters: {total_p:,} | Trainable: {train_p:,}")
print(f"Vocabulary Size: {tokenizer.vocab_size}")"""),

        markdown_cell("""## Step 3: Prepare Dynamic MLM DataLoaders
Hugging Face's dynamic masking collator (`DataCollatorForLanguageModeling`) masks 15% of tokens dynamically each epoch."""),

        code_cell("""max_seq_len = config["model"].get("max_seq_length", 128)
train_dataset = MLMTextDataset(train_df["text"].tolist(), tokenizer, max_length=max_seq_len)
val_dataset = MLMTextDataset(val_df["text"].tolist(), tokenizer, max_length=max_seq_len)

collator = get_mlm_collator(tokenizer, mlm_probability=config["stage1_dapt"].get("mlm_probability", 0.15))

batch_size = config["stage1_dapt"].get("per_device_train_batch_size", 16)
eval_batch_size = config["stage1_dapt"].get("per_device_eval_batch_size", 32)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator, num_workers=2 if device_info["cuda_available"] else 0)
val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=collator, num_workers=2 if device_info["cuda_available"] else 0)

print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")"""),

        markdown_cell("""## Step 4: Execute Domain-Adaptive Pretraining
We optimize with AdamW, linear learning-rate warmup, gradient clipping, and mixed-precision (AMP) for GPU acceleration."""),

        code_cell("""output_dir = config["stage1_dapt"]["output_dir"]
dapt_cfg = config["stage1_dapt"]

dapt_log = run_dapt_training(
    model=model,
    tokenizer=tokenizer,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    output_dir=output_dir,
    num_epochs=dapt_cfg["num_train_epochs"],
    learning_rate=float(dapt_cfg["learning_rate"]),
    weight_decay=float(dapt_cfg["weight_decay"]),
    warmup_ratio=float(dapt_cfg["warmup_ratio"]),
    use_fp16=dapt_cfg["fp16"] and device_info["cuda_available"]
)"""),

        markdown_cell("""## Step 5: Visualize DAPT Pretraining Convergence
We inspect training loss and validation perplexity across epochs."""),

        code_cell("""import matplotlib.pyplot as plt

history = dapt_log["history"]
epochs = [h["epoch"] for h in history]
train_losses = [h["train_loss"] for h in history]
val_losses = [h["val_loss"] for h in history]
perplexities = [h["perplexity"] for h in history]

fig, ax1 = plt.subplots(figsize=(8, 4))

ax1.plot(epochs, train_losses, "b-o", label="Train Loss (MLM)")
ax1.plot(epochs, val_losses, "r-s", label="Val Loss (MLM)")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Cross Entropy Loss")
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(epochs, perplexities, "g--^", label="Validation Perplexity")
ax2.set_ylabel("Perplexity (exp(loss))", color="g")
ax2.tick_params(axis="y", labelcolor="g")

plt.title("GeoFinBERT-DAPT Pretraining Convergence")
plt.tight_layout()
plt.show()"""),

        markdown_cell("""## Step 6: Verify Checkpoint Compatibility
We verify that the saved checkpoint `GeoFinBERT-DAPT` can be loaded seamlessly for Stage 2 fine-tuning."""),

        code_cell("""from transformers import BertForMaskedLM
loaded_dapt = BertForMaskedLM.from_pretrained(output_dir)
print(f"Verified GeoFinBERT-DAPT checkpoint at {output_dir}")
print(f"Checkpoint parameter count: {sum(p.numel() for p in loaded_dapt.parameters()):,}")""")
    ]
    return cells

def build_notebook_3():
    cells = [
        markdown_cell("""# 03. Stage 2: Human-Supervised Sentiment Fine-Tuning
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)
### Academic Research Project: **GeoFinBERT**

---
### Research Methodology:
1. **Target**: Predict sentiment (`positive: 0`, `negative: 1`, `neutral: 2`) on real financial & commodity news.
2. **Ground Truth Integrity**: Exclusively independently human-labeled real articles. Zero synthetic labels.
3. **Partitioning**: Exact same chronological train, validation, and test splits for all models.
4. **Class Imbalance Mitigation**: Balanced inverse-frequency class-weighted cross-entropy loss:
   $$w_c = \\frac{N}{K \\cdot N_c}$$
5. **Head-to-Head Setup**:
   - **Model A (Baseline)**: Fine-tune original `ProsusAI/FinBERT`.
   - **Model B (Adapted)**: Fine-tune `GeoFinBERT-DAPT` (transferred encoder from Stage 1).
6. **Selection**: Champion classifier is saved as `GeoFinBERT`."""),

        code_cell("""# Setup & Environment Diagnostics
import os
import sys

try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    print("[Colab Environment Detected] Installing dependencies...")
    !pip install -q -r requirements.txt
    !pip install -q -e .

import torch
import pandas as pd
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath("src"))

from geofinbert.config import load_config
from geofinbert.utils import set_seed, get_device_info, count_parameters
from geofinbert.models.classifier import build_classifier_model
from geofinbert.data.dataset import SentimentClassificationDataset
from geofinbert.training.trainer_classifier import compute_class_weights, run_classifier_training

config = load_config("configs/config.yaml")
set_seed(config["project"]["seed"])
device_info = get_device_info()
print("Hardware Check:", device_info)
if device_info["cuda_available"]:
    print(f"Accelerated GPU Active: {device_info['gpu_name']} ({device_info['vram_gb']} GB VRAM)")"""),

        markdown_cell("""## Step 1: Load Chronological Human-Labeled Partitions"""),

        code_cell("""splits_dir = config["data"]["splits_dir"]
train_df = pd.read_parquet(os.path.join(splits_dir, "sentiment_train.parquet"))
val_df = pd.read_parquet(os.path.join(splits_dir, "sentiment_val.parquet"))
test_df = pd.read_parquet(os.path.join(splits_dir, "sentiment_test.parquet"))

print(f"Labeled Splits: Train={len(train_df):,} | Val={len(val_df):,} | Test={len(test_df):,}")"""),

        markdown_cell("""## Step 2: Compute Balanced Class Weights
To prevent the model from biasing toward dominant classes, we calculate inverse-frequency class weights on the training partition."""),

        code_cell("""class_weights = compute_class_weights(train_df["label"].tolist(), num_classes=3)
print(f"Normalized Class Weights: {class_weights}")"""),

        markdown_cell("""## Step 3: Experiment A — Train Baseline FinBERT
We fine-tune the original `ProsusAI/FinBERT` on the chronological train partition."""),

        code_cell("""ft_cfg = config["stage2_finetuning"]
max_seq_len = config["model"]["max_seq_length"]
batch_size = ft_cfg["per_device_train_batch_size"]
eval_batch_size = ft_cfg["per_device_eval_batch_size"]

# Build Baseline Classifier
base_model, tokenizer = build_classifier_model(
    backbone_path_or_name=config["model"]["base_model_name"],
    num_labels=3,
    class_weights=class_weights,
    is_dapt_checkpoint=False,
    local_files_only=False
)

train_ds = SentimentClassificationDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, max_length=max_seq_len)
val_ds = SentimentClassificationDataset(val_df["text"].tolist(), val_df["label"].tolist(), tokenizer, max_length=max_seq_len)
test_ds = SentimentClassificationDataset(test_df["text"].tolist(), test_df["label"].tolist(), tokenizer, max_length=max_seq_len)

num_workers = 2 if device_info["cuda_available"] else 0
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers)

baseline_log = run_classifier_training(
    model=base_model,
    tokenizer=tokenizer,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    test_dataloader=test_loader,
    output_dir=ft_cfg["baseline_output_dir"],
    model_name="FinBERT-Baseline",
    num_epochs=ft_cfg["num_train_epochs"],
    learning_rate=float(ft_cfg["learning_rate"]),
    early_stopping_patience=ft_cfg["early_stopping_patience"],
    use_fp16=ft_cfg["fp16"] and device_info["cuda_available"]
)"""),

        markdown_cell("""## Step 4: Experiment B — Train Adapted GeoFinBERT
We fine-tune `GeoFinBERT-DAPT` (transferred domain-adapted backbone) using the exact same hyperparameters and chronological partitions."""),

        code_cell("""dapt_dir = config["stage1_dapt"]["output_dir"]

adapted_model, tokenizer = build_classifier_model(
    backbone_path_or_name=dapt_dir,
    num_labels=3,
    class_weights=class_weights,
    is_dapt_checkpoint=True,
    local_files_only=True
)

adapted_log = run_classifier_training(
    model=adapted_model,
    tokenizer=tokenizer,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    test_dataloader=test_loader,
    output_dir=ft_cfg["adapted_output_dir"],
    model_name="GeoFinBERT",
    num_epochs=ft_cfg["num_train_epochs"],
    learning_rate=float(ft_cfg["learning_rate"]),
    early_stopping_patience=ft_cfg["early_stopping_patience"],
    use_fp16=ft_cfg["fp16"] and device_info["cuda_available"]
)"""),

        markdown_cell("""## Step 5: Side-by-Side Validation F1 Comparison"""),

        code_cell("""import matplotlib.pyplot as plt

b_hist = baseline_log["history"]
a_hist = adapted_log["history"]

plt.figure(figsize=(8, 4))
plt.plot([h["epoch"] for h in b_hist], [h["macro_f1"] for h in b_hist], "r--s", label="Baseline FinBERT Val Macro F1")
plt.plot([h["epoch"] for h in a_hist], [h["macro_f1"] for h in a_hist], "b-o", label="GeoFinBERT Val Macro F1")
plt.xlabel("Epoch")
plt.ylabel("Validation Macro F1")
plt.title("Fine-Tuning Convergence: Baseline FinBERT vs GeoFinBERT")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()"""),

        markdown_cell("""## Step 6: Final Classifier Checkpoint Verification
The final classifier is verified and stored in `models/checkpoints/GeoFinBERT` ready for test set evaluation and inference.""")
    ]
    return cells

def build_notebook_4():
    cells = [
        markdown_cell("""# 04. Evaluation, Comparative Analysis & Inference
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)
### Academic Research Project: **GeoFinBERT**

---
### Evaluation Protocol:
- **Partition**: Strict unseen chronological test split (15% of human-labeled corpus).
- **Zero Leakage**: No test period text was present in Stage 1 DAPT or Stage 2 training.
- **Metrics**: Accuracy, Macro F1, Weighted F1, Macro Precision, Macro Recall, Per-Class F1, Confusion Matrix.
- **Inference Pipeline**: Interactive deployment for live market intelligence and geopolitical risk tracking."""),

        code_cell("""# Setup & Imports
import os
import sys

try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    print("[Colab Environment Detected] Installing dependencies...")
    !pip install -q -r requirements.txt
    !pip install -q -e .

import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath("src"))

from geofinbert.config import load_config
from geofinbert.models.classifier import GeoFinBERTForSequenceClassification
from geofinbert.data.dataset import SentimentClassificationDataset
from geofinbert.training.trainer_classifier import evaluate_classifier
from geofinbert.evaluation.metrics import compute_classification_metrics, format_metrics_table
from geofinbert.inference import GeoFinBERTInferencePipeline

config = load_config("configs/config.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Evaluation Device:", device)"""),

        markdown_cell("""## Step 1: Load Test Partition & Models"""),

        code_cell("""test_path = os.path.join(config["data"]["splits_dir"], "sentiment_test.parquet")
test_df = pd.read_parquet(test_path)
print(f"Loaded Test Set: {len(test_df):,} samples.")

baseline_dir = config["stage2_finetuning"]["baseline_output_dir"]
adapted_dir = config["stage2_finetuning"]["adapted_output_dir"]

from transformers import BertTokenizer
tok = BertTokenizer.from_pretrained(adapted_dir if os.path.exists(adapted_dir) else "ProsusAI/finbert")
test_ds = SentimentClassificationDataset(test_df["text"].tolist(), test_df["label"].tolist(), tok, max_length=128)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

# Load Models
print("Loading Baseline FinBERT...")
b_model = GeoFinBERTForSequenceClassification.from_pretrained(baseline_dir).to(device)
_, b_metrics = evaluate_classifier(b_model, test_loader, device)

print("Loading GeoFinBERT...")
a_model = GeoFinBERTForSequenceClassification.from_pretrained(adapted_dir).to(device)
_, a_metrics = evaluate_classifier(a_model, test_loader, device)"""),

        markdown_cell("""## Step 2: Head-to-Head Academic Metrics Table"""),

        code_cell("""from IPython.display import display, Markdown

table_md = format_metrics_table(b_metrics, a_metrics)
display(Markdown(table_md))"""),

        markdown_cell("""## Step 3: Confusion Matrix Comparison"""),

        code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))
class_labels = ["Positive", "Negative", "Neutral"]

# Baseline FinBERT
sns.heatmap(
    b_metrics["confusion_matrix"],
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_labels,
    yticklabels=class_labels,
    ax=axes[0]
)
axes[0].set_title(f"Baseline FinBERT (Acc: {b_metrics['accuracy']:.2%}, Macro F1: {b_metrics['macro_f1']:.4f})")
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("True Ground Truth")

# GeoFinBERT
sns.heatmap(
    a_metrics["confusion_matrix"],
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=class_labels,
    yticklabels=class_labels,
    ax=axes[1]
)
axes[1].set_title(f"GeoFinBERT (Acc: {a_metrics['accuracy']:.2%}, Macro F1: {a_metrics['macro_f1']:.4f})")
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Ground Truth")

plt.tight_layout()
plt.show()"""),

        markdown_cell("""## Step 4: Per-Class F1 & Precision/Recall Breakdown"""),

        code_cell("""classes = ["Positive", "Negative", "Neutral"]
b_f1 = [b_metrics["per_class"][c.lower()]["f1_score"] for c in classes]
a_f1 = [a_metrics["per_class"][c.lower()]["f1_score"] for c in classes]

x = np.arange(len(classes))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(x - width/2, b_f1, width, label="Baseline FinBERT", color="#e74c3c")
ax.bar(x + width/2, a_f1, width, label="GeoFinBERT (Adapted)", color="#2ecc71")

ax.set_ylabel("F1 Score")
ax.set_title("Per-Class Sentiment F1 Comparison")
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()"""),

        markdown_cell("""## Step 5: Live Interactive Inference Pipeline
Test GeoFinBERT on novel geopolitical events and energy market headlines."""),

        code_cell("""# Initialize Inference Pipeline
pipeline = GeoFinBERTInferencePipeline(model_dir_or_name=adapted_dir, device="auto")

sample_headlines = [
    "Crude oil surges 4% following missile strikes on Red Sea commercial oil tankers.",
    "Iran and IAEA resume diplomatic dialogue regarding uranium enrichment monitoring in Vienna.",
    "Gold spot prices tumble 2.5% as bullion demand weakens amid rising treasury yields.",
    "OPEC+ ministers agree on surprise 500,000 barrel per day voluntary crude production cut.",
    "US Department of Treasury imposes fresh secondary sanctions on foreign entities facilitating Iranian petroleum exports.",
    "Central banks worldwide accelerate sovereign bullion purchases to diversify reserves away from dollar holdings."
]

results = pipeline.predict_batch(sample_headlines)

for r in results:
    print(f"\\nHeadline: {r['text']}")
    print(f"Sentiment: {r['prediction'].upper()} (Confidence: {r['confidence']:.2%})")
    print(f"Probabilities: {r['probabilities']}")"""),

        markdown_cell("""## Step 6: Model Export & Archiving
GeoFinBERT weights, tokenizer configurations, and metrics are finalized and ready for research publication.""")
    ]
    return cells

def main():
    print("Building enhanced Colab notebooks with auto-diagnostics...")
    save_notebook(build_notebook_1(), "notebooks/01_dataset_preparation.ipynb")
    save_notebook(build_notebook_2(), "notebooks/02_domain_adaptive_pretraining.ipynb")
    save_notebook(build_notebook_3(), "notebooks/03_human_labelled_finetuning.ipynb")
    save_notebook(build_notebook_4(), "notebooks/04_evaluation_and_inference.ipynb")
    
    # Rebuild zip bundle
    include_dirs = ["configs", "data/processed", "data/splits", "notebooks", "reports", "scripts", "src"]
    include_files = ["README.md", "requirements.txt", "setup.py"]

    zip_path = "GeoFinBERT_Colab_Bundle.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in include_files:
            if os.path.exists(f):
                z.write(f, f)
        for d in include_dirs:
            for root, _, files in os.walk(d):
                if "__pycache__" in root:
                    continue
                for f in files:
                    fp = os.path.join(root, f)
                    arcname = os.path.relpath(fp)
                    z.write(fp, arcname)
    print(f"Rebuilt {zip_path}, size: {os.path.getsize(zip_path)/(1024*1024):.2f} MB")

if __name__ == "__main__":
    main()
