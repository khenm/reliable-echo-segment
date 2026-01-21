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

def find_latest_latent_profile(root_dir="runs"):
    """
    Recursively finds the most recently modified latent_profile.joblib file in the root_dir.

    Args:
        root_dir (str): Root directory to search (default: "runs").

    Returns:
        str | None: Absolute path to the latest latent_profile.joblib, or None if not found.
    """
    latest_file = None
    latest_time = 0

    if not os.path.exists(root_dir):
        return None

    for root, _, files in os.walk(root_dir):
        for file in files:
            if file == "latent_profile.joblib":
                full_path = os.path.join(root, file)
                try:
                    mtime = os.path.getmtime(full_path)
                    if mtime > latest_time:
                        latest_time = mtime
                        latest_file = full_path
                except OSError:
                    continue
    
    if latest_file:
        return os.path.abspath(latest_file)
    return None

def load_checkpoint_dict(ckpt_path, device):
    """
    Loads the raw checkpoint dictionary or object from a file.
    
    Args:
        ckpt_path (str): Path to the checkpoint file.
        device (torch.device): Device to load the checkpoint onto.
        
    Returns:
        dict or object: The loaded checkpoint content.
    """
    return torch.load(ckpt_path, map_location=device, weights_only=False)

def load_model_weights(model, checkpoint, strict=True):
    """
    Loads model weights from a checkpoint, handling both state_dict and full checkpoint dicts.
    
    Args:
        model (torch.nn.Module): The model to load weights into.
        checkpoint (dict or object): The loaded checkpoint object/dict.
        strict (bool): Whether to enforce strict key matching.
        
    Returns:
        dict: The loaded state dict (for reference).
    """
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict, strict=strict)
    return state_dict

def save_checkpoint(path, state_dict):
    """
    Saves a checkpoint dictionary to the specified path.
    
    Args:
        path (str): Path to save the checkpoint.
        state_dict (dict): The dictionary containing model/optim state.
    """
    torch.save(state_dict, path)

def load_checkpoint(model, ckpt_path, device):
    """
    Legacy wrapper for backward compatibility. Loads model state.
    
    Args:
        model (torch.nn.Module): The model to load weights into.
        ckpt_path (str): Path to the checkpoint file.
        device (torch.device): Device to load the checkpoint onto.
    """
    ckpt = load_checkpoint_dict(ckpt_path, device)
    load_model_weights(model, ckpt)