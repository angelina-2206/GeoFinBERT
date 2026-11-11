# GeoFinBERT: Adapting FinBERT for Geopolitical & Energy-Market Sentiment Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.36+-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An academic machine-learning research framework that adapts **ProsusAI/FinBERT** to real-world historical news in energy markets and geopolitical conflict zones spanning **2000 to 11 November 2026**.

---

## 1. Research Overview & Core Contributions

Generic financial language models (such as original FinBERT) are trained predominantly on corporate 10-K filings, earnings call transcripts, and equity press releases. Consequently, they struggle with specialized geopolitical risk vocabulary, sovereign sanctions, and strategic commodity market drivers:
- **Oil & Crude Markets**: Brent, WTI, OPEC+ output quotas, SPR releases, refining crack spreads, maritime tanker seizures.
- **Gold & Bullion Markets**: Central bank bullion reserves, sovereign safe-haven flows, inflation hedging, de-dollarization.
- **US–Iran Relations**: JCPOA nuclear diplomacy, IRGC sanctions, secondary export restrictions, Persian Gulf confrontations.
- **Middle Eastern Geopolitical Events**: Strait of Hormuz naval security, Red Sea shipping attacks, regional proxy conflicts.

**GeoFinBERT** resolves this domain gap via a rigorous two-stage academic architecture:
1. **Stage 1 (Domain-Adaptive Pretraining - DAPT)**: Transfers FinBERT's financial BERT encoder into a Masked Language Modeling (`BertForMaskedLM`) architecture, pretraining on genuine, unlabelled historical news spanning 2000–2026 while preserving the original tokenizer.
2. **Stage 2 (Human-Supervised Sentiment Fine-Tuning)**: Fine-tunes on strictly human-annotated sentiment articles (`positive: 0`, `negative: 1`, `neutral: 2`) using balanced inverse-frequency class-weighted loss and leak-free chronological partitioning.

```
                  ProsusAI/FinBERT (Corporate Financial Domain)
                                       │
                                       ▼
        ┌─────────────────────────────────────────────────────────────┐
        │  STAGE 1: DOMAIN-ADAPTIVE PRETRAINING (DAPT)                │
        │  • Architecture: BertForMaskedLM                            │
        │  • Objective: Masked Language Modeling (15% dynamic mask)   │
        │  • Corpus: 14,486 genuine unlabelled articles (2000–2026)   │
        │  • Output: GeoFinBERT-DAPT Checkpoint                       │
        └──────────────────────────────┬──────────────────────────────┘
                                       │
                                       ▼
        ┌─────────────────────────────────────────────────────────────┐
        │  STAGE 2: HUMAN-SUPERVISED SENTIMENT FINE-TUNING            │
        │  • Architecture: BertForSequenceClassification (3 classes)  │
        │  • Loss: Balanced Class-Weighted Cross-Entropy Loss         │
        │  • Splits: Strictly Chronological (Train 70%, Val 15%, Test)│
        │  • Ground Truth: 6,809 independently human-labeled articles │
        └──────────────────────────────┬──────────────────────────────┘
                                       │
                       ┌───────────────┴───────────────┐
                       ▼                               ▼
             FinBERT (Baseline)             GeoFinBERT (Champion)
                       │                               │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                Identical Chronological Test Set Evaluation
```

---

## 2. Strict Academic Data Integrity Principles

The GeoFinBERT research pipeline enforces strict empirical data integrity rules:
1. **Zero Synthetic Articles**: Absolutely no synthetic text, no LLM-generated articles, and no simulated news.
2. **Zero Model-Generated Ground Truth**: Model predictions or pseudo-labels are never used as human ground truth.
3. **Provenance Preservation**: Canonical source URLs, publication timestamps, and accredited news publishers (ForexLive, MarketWatch, Benzinga, ZeroHedge, OilPrice, Economic Times, MoneyControl, Reuters) are preserved.
4. **Syndication Deduplication**: Removed exact duplicate content hashes and near-duplicate wire syndication stories ($O(N)$ prefix signature deduplication).
5. **Zero Lookahead Leakage**: Partitions are partitioned strictly chronologically:
   $$\max(\text{Train Date}) \le \min(\text{Val Date}) \le \max(\text{Val Date}) \le \min(\text{Test Date})$$
   Cross-partition duplicate text exclusion guarantees zero information leakage.

---

## 3. Repository Structure

```
.
├── configs/
│   └── config.yaml                     # Reproducibility hyperparameters & split ratios
├── data/
│   ├── raw/                            # Cached raw datasets from accredited archives
│   ├── processed/                      # Deduplicated, cleaned, schema-validated parquets
│   └── splits/                         # Strict chronological train, val, test partitions
├── models/
│   └── checkpoints/                    # Saved model weights, configs, and training logs
├── notebooks/
│   ├── 01_dataset_preparation.ipynb    # Ingestion, deduplication, statistics, and splits
│   ├── 02_domain_adaptive_pretraining.ipynb # Stage 1 DAPT with MLM on Colab GPU
│   ├── 03_human_labelled_finetuning.ipynb   # Stage 2 Supervised Fine-Tuning & Baselines
│   └── 04_evaluation_and_inference.ipynb    # Comparative evaluation & interactive inference
├── reports/
│   ├── dataset_statistics.md           # Distribution by year (2000-2026), topic & publisher
│   └── model_comparison_report.md      # Head-to-head metrics: Baseline FinBERT vs GeoFinBERT
├── scripts/
│   ├── prepare_data.py                 # CLI dataset acquisition, audit & split generator
│   ├── train_dapt.py                   # CLI Stage 1 DAPT pretraining engine
│   ├── train_classifier.py             # CLI Stage 2 Supervised fine-tuning engine
│   ├── evaluate_models.py              # CLI comparative evaluation & markdown reporter
│   ├── generate_notebooks.py           # Notebook generator and validator
│   └── run_smoke_test.py               # Local real-data CPU end-to-end smoke test
├── src/
│   └── geofinbert/                     # Modular Python package
│       ├── config.py                   # Configuration loader
│       ├── utils.py                    # Seed setting & hardware diagnostics
│       ├── data/
│       │   ├── fetcher.py              # Genuine dataset fetcher (oil, gold, US-Iran)
│       │   ├── preprocessor.py         # Text cleaner, O(N) deduplicator, chronological splitter
│       │   └── dataset.py              # PyTorch MLM & Classification Datasets
│       ├── models/
│       │   ├── dapt.py                 # BertForMaskedLM FinBERT encoder transfer
│       │   └── classifier.py           # Sequence classifier with class-weighted loss
│       ├── training/
│       │   ├── trainer_dapt.py         # MLM training engine with perplexity tracking
│       │   └── trainer_classifier.py   # Fine-tuning engine with early stopping
│       ├── evaluation/
│       │   └── metrics.py              # Macro/Weighted F1, accuracy, confusion matrix
│       └── inference.py                # Production inference API for live headlines
├── requirements.txt                    # Pinned library dependencies
├── setup.py                            # Package setup script (pip install -e .)
└── README.md                           # Documentation & execution guide
```

---

## 4. Google Colab GPU Execution Guide

For full training runs, Google Colab with GPU acceleration (T4, V100, or A100) is recommended.

1. **Upload / Clone repository into Google Drive / Colab**:
   ```python
   # In Google Colab Cell 1:
   from google.colab import drive
   drive.mount('/content/drive')
   %cd "/content/drive/MyDrive/Finance Research paper"
   !pip install -q -r requirements.txt
   !pip install -q -e .
   ```
2. **Execute the Notebooks in Sequence**:
   - `notebooks/01_dataset_preparation.ipynb`: Fetches real data, removes duplicates, builds chronological splits, and outputs `reports/dataset_statistics.md`.
   - `notebooks/02_domain_adaptive_pretraining.ipynb`: Transfers FinBERT encoder to `BertForMaskedLM`, trains Stage 1 DAPT using mixed precision (`fp16`), and saves `GeoFinBERT-DAPT`.
   - `notebooks/03_human_labelled_finetuning.ipynb`: Applies class weights, trains baseline `ProsusAI/FinBERT` vs `GeoFinBERT-DAPT`, and saves the champion classifier as `GeoFinBERT`.
   - `notebooks/04_evaluation_and_inference.ipynb`: Evaluates both models on unseen chronological test partition, computes confusion matrices, and runs live interactive inference.

---

## 5. Local Command-Line Execution (CLI)

The complete pipeline can also be run via modular command-line scripts:

### Step 1: Install Package in Editable Mode
```bash
pip install -r requirements.txt
pip install -e .
```

### Step 2: Run End-to-End Real-Data Smoke Test
Verifies dataset integrity, chronological partitioning, DAPT MLM step, class-weighted fine-tuning, and inference on CPU:
```bash
python scripts/run_smoke_test.py
```

### Step 3: Ingest & Partition the Complete Corpus
```bash
python scripts/prepare_data.py
```

### Step 4: Run Stage 1 Domain-Adaptive Pretraining
```bash
python scripts/train_dapt.py --epochs 3 --batch_size 16
```

### Step 5: Run Stage 2 Supervised Fine-Tuning
Trains baseline FinBERT and GeoFinBERT on the exact same chronological partitions:
```bash
python scripts/train_classifier.py --model_type both --epochs 4 --batch_size 16
```

### Step 6: Evaluate & Generate Academic Comparison Report
```bash
python scripts/evaluate_models.py
```

---

## 6. Live Inference API

Use `GeoFinBERTInferencePipeline` in your Python applications or algorithmic pipelines:

```python
from geofinbert.inference import GeoFinBERTInferencePipeline

pipeline = GeoFinBERTInferencePipeline("models/checkpoints/GeoFinBERT")

headline = "Crude oil rallies as US sanctions on Iranian petroleum exports tighten Persian Gulf supply."
result = pipeline.predict(headline)

print("Headline:", result["text"])
print("Sentiment:", result["prediction"])        # 'positive', 'negative', or 'neutral'
print(f"Confidence: {result['confidence']:.2%}")
print("Class Probabilities:", result["probabilities"])
```

---

## 7. Dataset Summary & Provenance

| Metric / Dimension | Stage 1 (DAPT Unlabeled) | Stage 2 (Supervised Labeled) |
| :--- | :---: | :---: |
| **Total Retained Articles** | **14,486** | **6,809** |
| **Time Period** | 2000-02-15 to 2026-08-26 | 2000-02-15 to 2019-02-01 |
| **Duplicate Wire Stories Removed** | 1,800 duplicates | 7 duplicates |
| **Gold / Bullion Articles** | 8,549 (59.02%) | 6,768 (99.40%) |
| **US–Iran Relations Articles** | 4,031 (27.83%) | Present in commodity corpus |
| **Oil & Crude Market Articles** | 1,588 (10.96%) | 41 |
| **Middle East Geopolitics Articles** | 318 (2.20%) | Present in regional wire coverage |
| **Sentiment Label Distribution** | N/A (Self-supervised MLM) | Pos: 51.12%, Neg: 44.19%, Neu: 4.68% |
| **Class Weighting Applied** | N/A | Balanced Inverse Frequency ($w_c = \frac{N}{K \cdot N_c}$) |

Detailed annual breakdowns and source distributions are documented in [reports/dataset_statistics.md](reports/dataset_statistics.md).

---

## 8. Academic Citation

```bibtex
@article{geofinbert2026,
  title={GeoFinBERT: Adapting Financial BERT for Geopolitical Conflict and Energy Market Dynamics},
  author={Academic Machine Learning Research Team},
  year={2026},
  journal={Working Paper in Computational Financial Economics}
}
```

