import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from monai.data import CacheDataset, DataLoader, list_data_collate
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRangePercentilesd, ResizeWithPadOrCropd

# Import from src
from src.utils.logging import get_logger
from src.dataset import _read_ids, _get_files
from src.models.model import get_model
from src.core.conformal import ConformalCalibrator
from src.calibration_split import split_validation_set
from src.inference.wrapper import predict_with_guarantee

def get_rcps_dataloaders(cfg):
    """
    Creates dataloaders for the RCPS calibration and test sets.
    
    Args:
        cfg (dict): Configuration dictionary containing data paths.
        
    Returns:
        tuple: (DataLoader, DataLoader) for calibration and test sets.
        
    Raises:
        FileNotFoundError: If the split files do not exist.
    """
    split_dir = cfg['data']['split_dir']
    nii_dir = cfg['data']['nifti_dir']
    
    cal_txt = os.path.join(split_dir, "subgroup_val_calibration.txt")
    test_txt = os.path.join(split_dir, "subgroup_val_test.txt")
    
    if not os.path.exists(cal_txt) or not os.path.exists(test_txt):
        raise FileNotFoundError(f"RCPS split files not found at {split_dir}. Run calibration_split.py first.")
        
    ids_cal = _read_ids(cal_txt)
    ids_test = _read_ids(test_txt)
    
    cal_files = _get_files(ids_cal, nii_dir)
    test_files = _get_files(ids_test, nii_dir)
    
    # Transforms
    img_size = tuple(cfg['data']['img_size'])
    tf_val = Compose([
        LoadImaged(("image", "label")),
        EnsureChannelFirstd(("image", "label")),
        ScaleIntensityRangePercentilesd("image", 1, 99, 0, 1, clip=True),
        ResizeWithPadOrCropd(("image", "label"), img_size),
    ])
    
    ds_cal = CacheDataset(cal_files, tf_val, 1.0, num_workers=4)
    ds_test = CacheDataset(test_files, tf_val, 1.0, num_workers=4)
    
    ld_cal = DataLoader(ds_cal, batch_size=cfg['training']['batch_size_val'], shuffle=False, 
                       num_workers=4, pin_memory=True, collate_fn=list_data_collate)
    ld_test = DataLoader(ds_test, batch_size=cfg['training']['batch_size_val'], shuffle=False, 
                        num_workers=4, pin_memory=True, collate_fn=list_data_collate)
                        
    return ld_cal, ld_test

def run_rcps_pipeline(cfg):
    """
    Executes the RCPS pipeline: Calibration and Evaluation.

    1. Loads stratified calibration and test data.
    2. Loads the trained model.
    3. Calibrates the conformal predictor to find optimal lambda.
    4. Evaluates the guaranteed predictor on the test set.
    5. Saves risk plots and metrics.

    Args:
        cfg (dict): Configuration dictionary.
    """
    logger = get_logger()
    logger.info("Starting RCPS Pipeline...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Check for split files, generate if missing
    split_dir = cfg['data']['split_dir']
    cal_txt = os.path.join(split_dir, "subgroup_val_calibration.txt")
    test_txt = os.path.join(split_dir, "subgroup_val_test.txt")
    
    if not os.path.exists(cal_txt) or not os.path.exists(test_txt):
        logger.info("RCPS split files not found. Generating now...")
        bins = cfg['data'].get('stratify_bins', None)
        split_validation_set(cfg, bins=bins)
    
    # Load Data
    try:
        ld_cal, ld_test = get_rcps_dataloaders(cfg)
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    logger.info(f"Calibration batches: {len(ld_cal)} | Test batches: {len(ld_test)}")
    
    # Load Model
    model = get_model(cfg, device)
    ckpt_path = cfg['training']['ckpt_save_path']
    
    if not os.path.exists(ckpt_path):
        if os.path.exists(os.path.join("checkpoints", ckpt_path)):
            ckpt_path = os.path.join("checkpoints", ckpt_path)
            
    if not os.path.exists(ckpt_path):
        logger.error(f"Checkpoint not found at {ckpt_path}. Cannot run RCPS.")
        return
        
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    # Collect Calibration Scores
    logger.info("Collecting calibration scores...")
    
    all_scores = []
    all_masks = []
    
    LV_LABEL = 1
    
    with torch.no_grad():
        for batch in ld_cal:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)
            
            # Scores for target class (LV)
            scores = probs[:, LV_LABEL, :, :]
            
            # Binary Mask for LV
            masks = (labels == LV_LABEL).float().squeeze(1)
            
            all_scores.append(scores.cpu())
            all_masks.append(masks.cpu())
            
    all_scores = torch.cat(all_scores, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    
    logger.info(f"Calibration Set Size: {all_scores.shape[0]} images")
    
    # Calibrate
    alpha = 0.1 
    logger.info(f"Calibrating for Risk <= {alpha} (Dice >= {1-alpha})...")
    
    calibrator = ConformalCalibrator(all_scores, all_masks)
    best_lambda = calibrator.calibrate(all_scores, all_masks, alpha=alpha)
    
    # Plot Risk Curve
    risks = calibrator.get_risk_curve()
    
    lambdas = [x['lambda'] for x in risks]
    empirical_risks = [x['empirical_risk'] for x in risks]
    upper_bounds = [x['upper_bound'] for x in risks]
    
    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, empirical_risks, label='Empirical Risk', marker='.')
    plt.plot(lambdas, upper_bounds, label='Hoeffding Upper Bound', linestyle='--')
    plt.axhline(y=alpha, color='r', linestyle=':', label=f'Alpha ({alpha})')
    if best_lambda is not None:
        plt.axvline(x=best_lambda, color='g', linestyle='-', label=f'Optimal Lambda ({best_lambda:.3f})')
    
    plt.xlabel('Threshold $\lambda$')
    plt.ylabel('Risk ($1 - Dice$)')
    plt.title('RCPS Calibration: Risk vs Threshold')
    plt.legend()
    plt.grid(True)
    os.makedirs("runs", exist_ok=True)
    plt.savefig("runs/rcps_risk.png")
    logger.info("Saved risk plot to runs/rcps_risk.png")
    
    if best_lambda is None:
        logger.warning("Calibration failed to find a valid lambda satisfying the bound.")
    
    # Evaluate on Test Set
    logger.info("Evaluating on Test Set...")
    
    pass_cnt = 0
    total_cnt = 0
    test_dices = []
    
    with torch.no_grad():
        for batch in ld_test:
             imgs = batch["image"].to(device)
             labels = batch["label"].to(device) 
             
             pred_masks = predict_with_guarantee(model, imgs, calibrator)
             
             for i in range(len(imgs)):
                 gt = (labels[i] == LV_LABEL).float()
                 pr = (pred_masks[i] == LV_LABEL).float()
                 
                 # Compute Dice
                 inter = (gt * pr).sum()
                 union = gt.sum() + pr.sum()
                 dice = (2.0 * inter) / (union + 1e-8)
                 dice = dice.item()
                 
                 test_dices.append(dice)
                 
                 # Check pass (Risk <= alpha)
                 if (1.0 - dice) <= alpha:
                     pass_cnt += 1
                 total_cnt += 1
                 
    pass_rate = pass_cnt / total_cnt if total_cnt > 0 else 0
    mean_dice = np.mean(test_dices) if test_dices else 0
    
    logger.info("Test Evaluation Complete.")
    logger.info(f"Mean Dice: {mean_dice:.4f}")
    logger.info(f"Pass Rate (Dice >= {1-alpha:.2f}): {pass_rate*100:.2f}%")
    
    # Save results
    with open(os.path.join("runs", "rcps_results.txt"), "w") as f:
        f.write(f"Optimal Lambda: {best_lambda}\n")
        f.write(f"Alpha: {alpha}\n")
        f.write(f"Mean Test Dice: {mean_dice:.4f}\n")
        f.write(f"Pass Rate: {pass_rate*100:.2f}%\n")
