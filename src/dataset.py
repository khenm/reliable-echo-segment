import os
import math
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRangePercentilesd,
    ResizeWithPadOrCropd, RandFlipd, RandRotate90d, RandAffined
)
from monai.data import CacheDataset, DataLoader, list_data_collate

def _read_ids(txt_path):
    with open(txt_path) as f:
        return [l.strip() for l in f if l.strip()]

def _get_files(ids, data_nii_dir):
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
    # Reads IDs
    split_dir = cfg['data']['split_dir']
    nii_dir = cfg['data']['nifti_dir']
    
    ids_tr = _read_ids(os.path.join(split_dir, "subgroup_training.txt"))
    ids_val = _read_ids(os.path.join(split_dir, "subgroup_validation.txt"))
    ids_ts = _read_ids(os.path.join(split_dir, "subgroup_testing.txt"))

    train_files = _get_files(ids_tr, nii_dir)
    val_files = _get_files(ids_val, nii_dir)
    test_files = _get_files(ids_ts, nii_dir)

    print(f"Files -> train {len(train_files)} · val {len(val_files)} · test {len(test_files)}")

    # Transforms
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

    # Datasets
    ds_tr = CacheDataset(train_files, tf_tr, 1.0, num_workers=4)
    ds_va = CacheDataset(val_files, tf_val, 1.0, num_workers=4)
    ds_ts = CacheDataset(test_files, tf_val, 1.0, num_workers=4)

    # Loaders
    ld_tr = DataLoader(ds_tr, batch_size=cfg['training']['batch_size_train'], shuffle=True, 
                       num_workers=4, pin_memory=True, collate_fn=list_data_collate)
    ld_va = DataLoader(ds_va, batch_size=cfg['training']['batch_size_val'], shuffle=False, 
                       num_workers=4, pin_memory=True, collate_fn=list_data_collate)
    ld_ts = DataLoader(ds_ts, batch_size=cfg['training']['batch_size_val'], shuffle=False, 
                       num_workers=4, pin_memory=True, collate_fn=list_data_collate)

    return ld_tr, ld_va, ld_ts