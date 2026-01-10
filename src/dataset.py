import os
import math
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRangePercentilesd,
    ResizeWithPadOrCropd, RandFlipd, RandRotate90d, RandAffined
)
from monai.data import CacheDataset, DataLoader, list_data_collate
from src.utils.logging import get_logger

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

def get_dataloaders(cfg):
    """
    Creates DataLoaders for training, validation, and testing.
    
    Args:
        cfg (dict): Configuration dictionary containing data paths and training params.
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    logger = get_logger()
    
    data_name = cfg['data'].get('name', 'CAMUS').upper()
    if data_name == 'ECHONET':
        from src.dataset_echonet import get_echonet_dataloaders
        logger.info("Using EchoNet-Dynamic dataset.")
        return get_echonet_dataloaders(cfg)
    
    split_dir = cfg['data']['split_dir']
    nii_dir = cfg['data']['nifti_dir']
    
    logger.info(f"Reading splits from {split_dir}...")
    ids_tr = _read_ids(os.path.join(split_dir, "subgroup_training.txt"))
    ids_val = _read_ids(os.path.join(split_dir, "subgroup_validation.txt"))
    ids_ts = _read_ids(os.path.join(split_dir, "subgroup_testing.txt"))
    
    logger.info(f"IDs found -> train {len(ids_tr)} · val {len(ids_val)} · test {len(ids_ts)}")

    train_files = _get_files(ids_tr, nii_dir)
    val_files = _get_files(ids_val, nii_dir)
    test_files = _get_files(ids_ts, nii_dir)

    logger.info(f"Files found -> train {len(train_files)} · val {len(val_files)} · test {len(test_files)}")

    img_size = tuple(cfg['data']['img_size'])
    _common = [
        LoadImaged(("image", "label")),
        EnsureChannelFirstd(("image", "label")),
        ScaleIntensityRangePercentilesd("image", 1, 99, 0, 1, clip=True),
        ResizeWithPadOrCropd(("image", "label"), img_size),
    ]
    
    _aug = [
        RandFlipd(("image", "label"), prob=0.5, spatial_axis=1),
        RandRotate90d(("image", "label"), prob=0.5, max_k=3),
        RandAffined(("image", "label"), prob=0.3,
                    rotate_range=math.pi/18,
                    shear_range=0.05,
                    scale_range=0.05,
                    mode=("bilinear", "nearest")),
    ]

    tf_tr = Compose(_common + _aug)
    tf_val = Compose(_common)

    num_workers = cfg['training'].get('num_workers', 4)
    persistent_workers = cfg['training'].get('persistent_workers', True)

    ds_tr = CacheDataset(train_files, tf_tr, 1.0, num_workers=num_workers)
    ds_va = CacheDataset(val_files, tf_val, 1.0, num_workers=num_workers)
    ds_ts = CacheDataset(test_files, tf_val, 1.0, num_workers=num_workers)

    ld_tr = DataLoader(ds_tr, batch_size=cfg['training']['batch_size_train'], shuffle=True, 
                       num_workers=num_workers, pin_memory=True, collate_fn=list_data_collate,
                       persistent_workers=persistent_workers)
    ld_va = DataLoader(ds_va, batch_size=cfg['training']['batch_size_val'], shuffle=False, 
                       num_workers=num_workers, pin_memory=True, collate_fn=list_data_collate,
                       persistent_workers=persistent_workers)
    ld_ts = DataLoader(ds_ts, batch_size=cfg['training']['batch_size_val'], shuffle=False, 
                       num_workers=num_workers, pin_memory=True, collate_fn=list_data_collate,
                       persistent_workers=persistent_workers)

    return ld_tr, ld_va, ld_ts