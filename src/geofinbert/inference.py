import os
import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from typing import List, Dict, Any, Union

from geofinbert.models.classifier import GeoFinBERTForSequenceClassification

ID2LABEL = {0: "positive", 1: "negative", 2: "neutral"}
LABEL2ID = {"positive": 0, "negative": 1, "neutral": 2}

class GeoFinBERTInferencePipeline:
    """
    Production inference engine for GeoFinBERT sentiment analysis on geopolitical
    and energy-market news.
    """
    def __init__(self, model_dir_or_name: str, device: str = "auto", max_length: int = 128):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"[Inference] Loading GeoFinBERT model from: {model_dir_or_name} onto {self.device}")
        self.tokenizer = BertTokenizer.from_pretrained(model_dir_or_name)
        self.model = GeoFinBERTForSequenceClassification.from_pretrained(model_dir_or_name)
        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length

    @torch.no_grad()
    def predict(self, text: str) -> Dict[str, Any]:
        """Classify single headline or article snippet."""
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        token_type_ids = encoding.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)
            
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        logits = outputs["logits"]
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        pred_id = int(probs.argmax())
        
        return {
            "text": text,
            "prediction": ID2LABEL[pred_id],
            "label_id": pred_id,
            "confidence": float(round(probs[pred_id], 4)),
            "probabilities": {
                "positive": float(round(probs[0], 4)),
                "negative": float(round(probs[1], 4)),
                "neutral": float(round(probs[2], 4)),
            }
        }

    @torch.no_grad()
    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
        """Batch inference for high-throughput news pipelines."""
        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encodings = self.tokenizer(
                batch_texts,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)
            token_type_ids = encodings.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)
                
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            logits = outputs["logits"]
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            
            for j, t in enumerate(batch_texts):
                pred_id = int(probs[j].argmax())
                results.append({
                    "text": t,
                    "prediction": ID2LABEL[pred_id],
                    "label_id": pred_id,
                    "confidence": float(round(probs[j][pred_id], 4)),
                    "probabilities": {
                        "positive": float(round(probs[j][0], 4)),
                        "negative": float(round(probs[j][1], 4)),
                        "neutral": float(round(probs[j][2], 4)),
                    }
                })
        return results

