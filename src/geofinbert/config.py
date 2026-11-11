import os
import yaml
from typing import Any, Dict

def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config

