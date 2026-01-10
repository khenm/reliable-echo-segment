import os
import torch
import numpy as np
import pandas as pd
import argparse
import yaml

from src.utils.logging import get_logger
from src.dataset import get_dataloaders
from src.models.model import get_model
from src.eval_rcps import get_rcps_dataloaders
from src.analysis.latent_profile import LatentProfiler
from src.analysis.adaptive_calibration import AdaptiveScaler
from src.utils.plot import plot_coverage_by_difficulty

def run_stratified_analysis(cfg):
    """
    Executes the Stratified Coverage Analysis (Proof on Hard Cases).
    """
    logger = get_logger()
    logger.info("Starting Stratified Coverage Analysis...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = cfg['training'].get('run_dir', 'runs')
    os.makedirs(run_dir, exist_ok=True)
    
    # 1. Load Resources
    # -----------------
    ckpt_path = cfg['training']['ckpt_save_path']
    profiler_path = os.path.join(run_dir, "latent_profile.joblib")
    
    # Fallback for checkpoint
    if not os.path.exists(ckpt_path):
        if os.path.exists(os.path.join("checkpoints", ckpt_path)):
             ckpt_path = os.path.join("checkpoints", ckpt_path)
    if not os.path.exists(ckpt_path):
        logger.error(f"Checkpoint not found at {ckpt_path}")
        return

    # Check Profiler
    if not os.path.exists(profiler_path):
        logger.error(f"Latent profile not found at {profiler_path}. Run --profile first.")
        return

    # Model
    model = get_model(cfg, device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    # Profiler
    profiler = LatentProfiler()
    profiler.load(profiler_path)

    # Data
    try:
        ld_cal, ld_test = get_rcps_dataloaders(cfg)
    except FileNotFoundError:
        logger.error("Split files not found. Run RCPS first.")
        return

    LV_LABEL = 1

    # 2. Calibration Phase
    # --------------------
    logger.info("Collecting Calibration Data...")
    
    cal_dists = []
    cal_errs = []
    
    with torch.no_grad():
        for batch in ld_cal:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            logits, mu, _ = model(imgs)
            probs = torch.softmax(logits, dim=1)
            pred_labels = torch.argmax(logits, dim=1)
            
            for i in range(len(imgs)):
                # Distance
                vec = mu[i].cpu().numpy()
                dist = profiler.get_mahalanobis_distance(vec)
                cal_dists.append(dist)
                
                # Error (1 - Dice)
                gt = (labels[i] == LV_LABEL).float()
                pr = (pred_labels[i] == LV_LABEL).float()
                dice = (2.0 * (gt * pr).sum()) / (gt.sum() + pr.sum() + 1e-8)
                err = 1.0 - dice.item()
                cal_errs.append(err)

    cal_dists = np.array(cal_dists)
    cal_errs = np.array(cal_errs)
    
    # Calibrate Baseline (Constant Bound)
    # Find Q_base such that P(err <= Q_base) = 0.90
    alpha = 0.1
    n = len(cal_errs)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(1.0, max(0.0, q_level))
    Q_base = np.quantile(cal_errs, q_level, method='higher')
    logger.info(f"Baseline Quantile (Constant): {Q_base:.4f}")

    # Calibrate MACS (Adaptive)
    scaler = AdaptiveScaler()
    scaler.fit(cal_dists, cal_errs)
    
    s_hat_cal = scaler.predict(cal_dists)
    s_hat_cal = np.maximum(s_hat_cal, 1e-6)
    s_prime = cal_errs / s_hat_cal
    
    Q_macs = np.quantile(s_prime, q_level, method='higher')
    logger.info(f"MACS Quantile (Adaptive): {Q_macs:.4f}")

    # 3. Evaluation Phase
    # -------------------
    logger.info("Evaluating on Test Set...")
    
    results = []
    
    with torch.no_grad():
        for batch in ld_test:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            logits, mu, _ = model(imgs)
            pred_labels = torch.argmax(logits, dim=1)
            
            for i in range(len(imgs)):
                vec = mu[i].cpu().numpy()
                dist = profiler.get_mahalanobis_distance(vec)
                
                gt = (labels[i] == LV_LABEL).float()
                pr = (pred_labels[i] == LV_LABEL).float()
                dice = (2.0 * (gt * pr).sum()) / (gt.sum() + pr.sum() + 1e-8)
                err = 1.0 - dice.item()
                
                # Baseline Check
                cov_base = (err <= Q_base)
                
                # MACS Check
                s_hat = max(scaler.predict(np.array([dist]))[0], 1e-6)
                bound_macs = Q_macs * s_hat
                cov_macs = (err <= bound_macs)
                
                results.append({
                    "dist": dist,
                    "error": err,
                    "cov_base": cov_base,
                    "cov_macs": cov_macs,
                    "s_hat": s_hat
                })
                
    df = pd.DataFrame(results)
    
    # 4. Binning and Analysis
    # -----------------------
    # Sort by Difficulty (Distance)
    df = df.sort_values("dist")
    
    # Create 3 bins by quantile
    df["bin"] = pd.qcut(df["dist"], 3, labels=["Easy", "Medium", "Hard"])
    
    # Calculate Coverage per Bin
    summary = df.groupby("bin")[["cov_base", "cov_macs"]].mean()
    summary.columns = ["Baseline", "MACS"]
    
    logger.info("\nStratified Coverage Results:")
    logger.info(summary)
    
    # Save Metrics
    csv_path = os.path.join(run_dir, "stratified_metrics.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved metrics to {csv_path}")

    # 5. Plotting
    # -----------
    plot_path = os.path.join(run_dir, "stratified_coverage.png")
    plot_coverage_by_difficulty(summary, save_path=plot_path)
    logger.info(f"Analysis Complete. Plot saved to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        
    run_stratified_analysis(cfg)
