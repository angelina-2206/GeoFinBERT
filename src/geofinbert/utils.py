import os
import random
import logging
import time
from typing import Dict, Any, Tuple
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """Set random seed across all libraries for deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def get_device_info() -> Dict[str, Any]:
    """Inspect and return hardware environment diagnostics."""
    cuda_available = torch.cuda.is_available()
    device_name = "cpu"
    gpu_name = None
    vram_gb = 0.0
    
    if cuda_available:
        device_name = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = round(vram_bytes / (1024 ** 3), 2)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_name = "mps"
        gpu_name = "Apple Silicon MPS"
    
    return {
        "device": device_name,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "torch_version": torch.__version__,
    }

def format_time(seconds: float) -> str:
    """Convert elapsed seconds into a readable string (HH:MM:SS)."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hrs > 0:
        return f"{hrs:02d}h {mins:02d}m {secs:02d}s"
    elif mins > 0:
        return f"{mins:02d}m {secs:02d}s"
    else:
        return f"{secs}s"

def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """Return total and trainable parameter counts."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def setup_logger(name: str = "geofinbert", log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """Configure standard logger with console and optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger

