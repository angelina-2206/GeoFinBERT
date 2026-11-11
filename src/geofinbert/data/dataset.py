import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional
from transformers import BertTokenizer, DataCollatorForLanguageModeling

class MLMTextDataset(Dataset):
    """
    PyTorch Dataset for Masked Language Modeling (DAPT).
    Tokenizes raw text without static masking; dynamic masking is applied by the collator.
    """
    def __init__(self, texts: List[str], tokenizer: BertTokenizer, max_length: int = 128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding.get("token_type_ids", torch.zeros_like(encoding["attention_mask"])).squeeze(0)
        }

class SentimentClassificationDataset(Dataset):
    """
    PyTorch Dataset for Human-Supervised Sentiment Fine-Tuning.
    Encodes text and integer labels (0: positive, 1: negative, 2: neutral).
    """
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: BertTokenizer,
        max_length: int = 128
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding.get("token_type_ids", torch.zeros_like(encoding["attention_mask"])).squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def get_mlm_collator(tokenizer: BertTokenizer, mlm_probability: float = 0.15) -> DataCollatorForLanguageModeling:
    """Return Hugging Face dynamic MLM data collator."""
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability
    )

