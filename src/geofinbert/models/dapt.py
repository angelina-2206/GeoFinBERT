import os
import torch
import torch.nn as nn
from transformers import BertConfig, BertForMaskedLM, BertForSequenceClassification, BertTokenizer
from typing import Tuple

def build_dapt_model(
    base_model_name_or_path: str = "ProsusAI/finbert",
    tokenizer_name: str = "ProsusAI/finbert",
    local_files_only: bool = False
) -> Tuple[BertForMaskedLM, BertTokenizer]:
    """
    Construct Stage 1 Domain-Adaptive Pretraining architecture.
    Transfers the financial-language BERT encoder weights from ProsusAI/FinBERT
    into a BERT Masked-Language-Model (BertForMaskedLM) architecture.
    Preserves the original tokenizer and ties output embeddings.
    """
    print(f"[Model DAPT] Loading tokenizer: {tokenizer_name}")
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    
    print(f"[Model DAPT] Initializing BertForMaskedLM from: {base_model_name_or_path}")
    config = BertConfig.from_pretrained(base_model_name_or_path)
    
    # Check if local model file exists or if we should attempt from_pretrained
    model = None
    if os.path.isdir(base_model_name_or_path):
        try:
            model = BertForMaskedLM.from_pretrained(base_model_name_or_path)
        except Exception as e:
            print(f"[Model DAPT] Local directory load notice ({e}).")
            
    if model is None:
        try:
            # Try loading pretrained with local_files_only if specified
            model = BertForMaskedLM.from_pretrained(base_model_name_or_path, local_files_only=local_files_only)
        except Exception as e:
            print(f"[Model DAPT] Pretrained weights not cached or remote unavailable ({e}). Initializing architecture from FinBERT config.")
            model = BertForMaskedLM(config)
            
    # Ensure word embeddings and prediction bias are tied
    model.tie_weights()
    print(f"[Model DAPT] Architecture initialized successfully. Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer
