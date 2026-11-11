import os
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertForSequenceClassification, BertTokenizer
from typing import Optional, Dict, Any, Tuple

class GeoFinBERTForSequenceClassification(nn.Module):
    """
    Sequence Classification model supporting both baseline FinBERT
    and Domain-Adapted GeoFinBERT-DAPT encoder backbones.
    Includes class-weighted CrossEntropyLoss to handle label imbalance.
    """
    def __init__(
        self,
        config: BertConfig,
        num_labels: int = 3,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.num_labels = num_labels
        self.config = config
        self.config.num_labels = num_labels
        
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        
        # Always register class_weights as a buffer (defaults to ones)
        if class_weights is not None:
            weights = class_weights.float()
        else:
            weights = torch.ones(num_labels, dtype=torch.float)
        self.register_buffer("class_weights", weights)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
            
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": outputs.hidden_states
        }

    def save_pretrained(self, save_directory: str):
        """Save model weights and configuration."""
        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)
        state_dict_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), state_dict_path)
        print(f"[Model Classifier] Saved checkpoint to {save_directory}")

    @classmethod
    def from_pretrained(
        cls,
        load_directory_or_name: str,
        class_weights: Optional[torch.Tensor] = None,
        num_labels: int = 3
    ):
        """Instantiate classifier from a saved checkpoint or pretrained directory."""
        config = BertConfig.from_pretrained(load_directory_or_name)
        model = cls(config=config, num_labels=num_labels, class_weights=class_weights)
        state_dict_path = os.path.join(load_directory_or_name, "pytorch_model.bin")
        if os.path.exists(state_dict_path):
            state = torch.load(state_dict_path, map_location="cpu")
            model.load_state_dict(state, strict=False)
        return model

def build_classifier_model(
    backbone_path_or_name: str = "ProsusAI/finbert",
    num_labels: int = 3,
    class_weights: Optional[torch.Tensor] = None,
    is_dapt_checkpoint: bool = False,
    local_files_only: bool = False
) -> Tuple[GeoFinBERTForSequenceClassification, BertTokenizer]:
    """
    Factory function to initialize a sentiment classifier:
    - Baseline: initialized with original FinBERT weights.
    - Adapted: initialized with GeoFinBERT-DAPT encoder weights.
    """
    print(f"[Model Classifier] Initializing classifier from backbone: {backbone_path_or_name} (DAPT={is_dapt_checkpoint})")
    tokenizer = BertTokenizer.from_pretrained(
        backbone_path_or_name if os.path.isdir(backbone_path_or_name) else "ProsusAI/finbert"
    )
    config = BertConfig.from_pretrained(backbone_path_or_name)
    config.num_labels = num_labels
    
    model = GeoFinBERTForSequenceClassification(
        config=config,
        num_labels=num_labels,
        class_weights=class_weights
    )
    
    # Load encoder weights
    if is_dapt_checkpoint and os.path.exists(os.path.join(backbone_path_or_name, "pytorch_model.bin")):
        print(f"[Model Classifier] Loading DAPT encoder weights from {backbone_path_or_name}...")
        dapt_state = torch.load(os.path.join(backbone_path_or_name, "pytorch_model.bin"), map_location="cpu")
        bert_keys = {k[5:]: v for k, v in dapt_state.items() if k.startswith("bert.")}
        missing, unexpected = model.bert.load_state_dict(bert_keys, strict=False)
        print(f"[Model Classifier] DAPT encoder loaded. (Missing keys: {len(missing)}, Unexpected: {len(unexpected)})")
    else:
        # Load baseline FinBERT
        print(f"[Model Classifier] Loading baseline FinBERT weights from {backbone_path_or_name}...")
        try:
            hf_model = BertForSequenceClassification.from_pretrained(
                backbone_path_or_name, local_files_only=local_files_only
            )
            model.bert.load_state_dict(hf_model.bert.state_dict())
            if hasattr(hf_model, "classifier"):
                model.classifier.load_state_dict(hf_model.classifier.state_dict())
        except Exception as e:
            print(f"[Model Classifier] Pretrained weights not cached or remote unavailable ({e}). Initializing from config.")
            
    return model, tokenizer
