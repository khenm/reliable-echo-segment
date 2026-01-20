import yaml
import os
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads a configuration file and resolves component references (models, datasets).
    
    Args:
        config_path (str): Path to the experiment configuration file.
        
    Returns:
        Dict[str, Any]: The fully resolved configuration dictionary.
    """
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    base_dir = os.path.dirname(config_path)
    
    # Resolve Model
    if isinstance(cfg.get('model'), str):
        model_name = cfg['model']
        model_config_path = os.path.join(base_dir, 'models', f"{model_name}.yaml")
        if not os.path.exists(model_config_path):
             raise FileNotFoundError(f"Model config not found at {model_config_path}")
             
        with open(model_config_path, 'r') as f:
            model_cfg = yaml.safe_load(f)
        cfg['model'] = model_cfg
        
    # Resolve Data
    if isinstance(cfg.get('data'), str):
        dataset_name = cfg['data']
        dataset_config_path = os.path.join(base_dir, 'datasets', f"{dataset_name}.yaml")
        if not os.path.exists(dataset_config_path):
             raise FileNotFoundError(f"Dataset config not found at {dataset_config_path}")
             
        with open(dataset_config_path, 'r') as f:
            dataset_cfg = yaml.safe_load(f)
            
        cfg['data'] = dataset_cfg
    
    # Handle nested data.name format
    elif isinstance(cfg.get('data'), dict) and 'name' in cfg['data']:
        dataset_name = cfg['data']['name']
        dataset_config_path = os.path.join(base_dir, 'datasets', f"{dataset_name}.yaml")
        if not os.path.exists(dataset_config_path):
            raise FileNotFoundError(f"Dataset config not found at {dataset_config_path}")
        
        with open(dataset_config_path, 'r') as f:
            dataset_cfg = yaml.safe_load(f)
        
        # Merge: dataset file as base, inline overrides take precedence
        inline_overrides = cfg['data']
        dataset_cfg.update(inline_overrides)
        cfg['data'] = dataset_cfg
    
    return cfg
