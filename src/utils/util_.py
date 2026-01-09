import torch
import numpy as np
import random
import os

def seed_everything(seed=42):
    """
    Seeds all random number generators for reproducibility.
    
    Args:
        seed (int): The seed value.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def get_device():
    """
    Returns the appropriate device (cuda or cpu).
    
    Returns:
        str: 'cuda' if available, else 'cpu'.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"