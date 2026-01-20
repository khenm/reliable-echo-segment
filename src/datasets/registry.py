import torch.nn as nn
import os 
from src.utils.logging import get_logger

class DatasetRegistry:
    """
    Central registry for dataset classes.
    """
    _registry = {}

    @classmethod
    def register(cls, name):
        """
        Decorator to register a dataset class with a specific name.
        """
        def decorator(dataset_class):
            if name in cls._registry:
                raise ValueError(f"Dataset '{name}' is already registered.")
            cls._registry[name] = dataset_class
            return dataset_class
        return decorator

    @classmethod
    def get(cls, name):
        """
        Retrieve a dataset class by name.
        """
        if name not in cls._registry:
            raise ValueError(f"Dataset '{name}' is not registered. Available datasets: {list(cls._registry.keys())}")
        return cls._registry[name]

    @classmethod
    def get_dataloaders(cls, cfg):
        """
        Instantiate dataloaders based on the configuration.

        Args:
            cfg (dict): Configuration dictionary.

        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        data_name = cfg['data'].get('name', 'CAMUS').upper()
        dataset_class = cls.get(data_name)
        
        if hasattr(dataset_class, 'get_dataloaders'):
            return dataset_class.get_dataloaders(cfg)
        else:
             raise NotImplementedError(f"Dataset '{data_name}' does not implement 'get_dataloaders'.")

def register_dataset(name):
    """
    Decorator to register a dataset class with a specific name.
    """
    return DatasetRegistry.register(name)

def _read_ids(txt_path):
    """
    Reads patient IDs from a text file.
    
    Args:
        txt_path (str): Path to the text file.
        
    Returns:
        list: List of patient IDs.
    """
    with open(txt_path) as f:
        return [l.strip() for l in f if l.strip()]

def _get_files(ids, data_nii_dir):
    """
    Scans the directory for image and label files for given patient IDs.
    
    Args:
        ids (list): List of patient IDs.
        data_nii_dir (str): Root directory of NIfTI data.
        
    Returns:
        list: List of dictionaries containing paths and metadata.
    """
    items = []
    for pid in ids:
        pdir = os.path.join(data_nii_dir, pid)
        for view in ("2CH", "4CH"):
            for ph in ("ED", "ES"):
                img = os.path.join(pdir, f"{pid}_{view}_{ph}.nii.gz")
                lbl = os.path.join(pdir, f"{pid}_{view}_{ph}_gt.nii.gz")
                if os.path.exists(img) and os.path.exists(lbl):
                    items.append({
                        "image": img,
                        "label": lbl,
                        "case":  pid,
                        "view":  view,
                        "phase": ph
                    })
    return items
