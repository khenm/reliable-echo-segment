import os
import torch
import numpy as np
import pandas as pd
from monai.metrics import DiceMetric

from src.utils.logging import get_logger
from src.dataset import get_dataloaders
from src.models.model import get_model
from src.eval_rcps import get_rcps_dataloaders
from src.analysis.latent_profile import LatentProfiler
from src.analysis.adaptive_calibration import AdaptiveScaler
import joblib

def run_adaptive_pipeline(cfg):
    """
    Executes the Adaptive Calibration pipeline.
    
    1. Load Model & Latent Profiler.
    2. Load Calibration Data.
    3. Compute Distances (d) and True Errors (s).
    4. Fit AdaptiveScaler: s_hat(d) = alpha * d + beta.
    5. Compute Scaled Non-Conformity Scores: s' = s / s_hat(d).
    6. Find Conformal Quantile (Q) for s'.
    7. Evaluate on Test Set: Bound(d) = Q * s_hat(d).
    """
    logger = get_logger()
    logger.info("Starting Adaptive Calibration Pipeline...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths
    run_dir = cfg['training'].get('run_dir', 'runs')
    profiler_path = os.path.join(run_dir, "latent_profile.joblib")
    ckpt_path = cfg['training']['ckpt_save_path']
    
    if not os.path.exists(ckpt_path):
        if os.path.exists(os.path.join("checkpoints", ckpt_path)):
             ckpt_path = os.path.join("checkpoints", ckpt_path)
             
    if not os.path.exists(ckpt_path):
        logger.error("Checkpoint not found.")
        return
        
    if not os.path.exists(profiler_path):
        logger.warning(f"Latent profile not found at {profiler_path}. Please run --profile first.")
        # Optional: could auto-run profiling, but let's error for clarity
        return 

    # Load Model
    model = get_model(cfg, device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Load Profiler
    profiler = LatentProfiler()
    profiler.load(profiler_path)
    
    # Load Data (Use RCPS splits for consistency)
    try:
        ld_cal, ld_test = get_rcps_dataloaders(cfg)
    except FileNotFoundError:
        logger.error("Split files not found. Run RCPS first to generate splits or ensure paths correct.")
        return

    # --- Phase 1: Calibration ---
    logger.info("Phase 1: Calibration - Fitting Adaptive Scaler")
    
    d_cal = [] # Distances
    s_cal = [] # Errors (1-Dice)
    
    LV_LABEL = 1
    
    with torch.no_grad():
        for batch in ld_cal:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            # Forward pass: get logit and mu
            logits, mu, _ = model(imgs)
            
            # Compute Dice/Error
            pred_labels = torch.argmax(logits, dim=1)
            
            for i in range(len(imgs)):
                # Get latent vector for this sample
                vec = mu[i].cpu().numpy()
                dist = profiler.get_mahalanobis_distance(vec)
                d_cal.append(dist)
                
                # Compute Dice
                gt = (labels[i] == LV_LABEL).float()
                pr = (pred_labels[i] == LV_LABEL).float()
                
                inter = (gt * pr).sum()
                union = gt.sum() + pr.sum()
                dice = (2.0 * inter) / (union + 1e-8)
                err = 1.0 - dice.item()
                s_cal.append(err)
                
    d_cal = np.array(d_cal)
    s_cal = np.array(s_cal)
    
    # Fit Linear Model
    scaler = AdaptiveScaler()
    scaler.fit(d_cal, s_cal)
    
    # Save Plot
    scaler.plot_calibration(d_cal, s_cal, save_path=os.path.join(run_dir, "adaptive_fit.png"))
    
    # --- Phase 2: Conformal Scaling ---
    logger.info("Phase 2: Conformal Calibration on Scaled Scores")
    
    # Predict s_hat for calibration data
    s_hat_cal = scaler.predict(d_cal)
    # Avoid division by zero/negative (though s_hat should be positive if correlation exists)
    s_hat_cal = np.maximum(s_hat_cal, 1e-6) 
    
    # Compute scaled scores
    s_prime = s_cal / s_hat_cal
    
    # Find Quantile
    alpha = 0.1 # 90% coverage target
    n = len(s_cal)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(1.0, max(0.0, q_level))
    
    Q = np.quantile(s_prime, q_level, method='higher') # 'higher' is standard for CP
    
    logger.info(f"Conformal Quantile Q (90% target): {Q:.4f}")
    
    # --- Phase 3: Testing ---
    logger.info("Phase 3: Testing Adaptive Coverage")
    
    d_test = []
    s_test = []
    bounds = []
    covered = []
    
    with torch.no_grad():
        for batch in ld_test:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            logits, mu, _ = model(imgs)
            pred_labels = torch.argmax(logits, dim=1)
            
            for i in range(len(imgs)):
                vec = mu[i].cpu().numpy()
                dist = profiler.get_mahalanobis_distance(vec)
                d_test.append(dist)
                
                gt = (labels[i] == LV_LABEL).float()
                pr = (pred_labels[i] == LV_LABEL).float()
                inter = (gt * pr).sum()
                union = gt.sum() + pr.sum()
                dice = (2.0 * inter) / (union + 1e-8)
                err = 1.0 - dice.item()
                s_test.append(err)
                
                # Calculate Adaptive Bound
                # Bound(d) = Q * s_hat(d)
                s_hat = scaler.predict(np.array([dist]))[0]
                s_hat = max(s_hat, 1e-6)
                bound = Q * s_hat
                
                bounds.append(bound)
                covered.append(err <= bound)
                
    coverage = np.mean(covered)
    avg_width = np.mean(bounds)
    
    logger.info("Adaptive Evaluation Results:")
    logger.info(f"Target Coverage: {1-alpha:.2f}")
    logger.info(f"Actual Coverage: {coverage:.4f}")
    logger.info(f"Average Bound Width (Error): {avg_width:.4f}")
    
    # Save Results
    res_path = os.path.join(run_dir, "adaptive_results.txt")
    with open(res_path, "w") as f:
        f.write(f"Alpha: {alpha}\n")
        f.write(f"Quantile Q: {Q}\n")
        f.write(f"Coverage: {coverage:.4f}\n")
        f.write(f"Avg Bound Width: {avg_width:.4f}\n")
        
    # Save Calibration State for Inference
    state_path = os.path.join(run_dir, "calibration_state.joblib")
    state = {
        'scaler': scaler,
        'Q': Q,
        'alpha': alpha
    }
    joblib.dump(state, state_path)
    logger.info(f"Saved calibration state to {state_path}")
