import argparse
import os
import torch
import pandas as pd
from tqdm import tqdm
from src.utils.config_loader import load_config
from src.utils.dist import setup_dist, cleanup_dist
from src.registry import get_dataloaders
from src.tta.engine import CloudAdaptiveEngine
from src.tta.auditor import AdaptiveGatekeeper
from src.utils.logging import get_logger
from src.utils.runner import run_init, load_model_for_inference

logger = get_logger()

def run_tta(cfg, device, simulate_cloud=False):
    mode_name = "Cloud-Guided" if simulate_cloud else "Standard"
    logger.info(f"Starting Test-Time Adaptation ({mode_name})...")
    
    # 1. Load Data
    original_bs = cfg['training'].get('batch_size')
    cfg['training']['batch_size'] = 1
    loaders = get_dataloaders(cfg)
    cfg['training']['batch_size'] = original_bs
    _, _, ld_ts = loaders
    
    # 2. Load Model
    model = load_model_for_inference(cfg, device)
    if model is None: return
    
    # 3. Load Golden Stats & Init Gatekeeper
    vault_dir = cfg['training']['vault_dir']
    gl_path = os.path.join(vault_dir, "golden_landmarks.pt")
    ga_path = os.path.join(vault_dir, "golden_amplitude.pt")
    
    gl = torch.load(gl_path, map_location=device) if os.path.exists(gl_path) else None
    ga = torch.load(ga_path, map_location=device) if os.path.exists(ga_path) else None
    
    if gl is None: logger.warning("Golden Landmarks not found. Adaption may be limited.")
    if ga is None: logger.warning("Golden Amplitude not found. Adaption may be limited.")
    
    gatekeeper = AdaptiveGatekeeper(golden_landmarks=gl, golden_amplitude=ga)
    
    # 4. Init Engine
    tta_cfg = cfg.get('tta', {})
    threshold = float(tta_cfg.get('uncertainty_threshold', 0.5))
    engine = CloudAdaptiveEngine(model, gatekeeper, uncertainty_threshold=threshold)
    
    results = []
    logger.info(f"Running TTA on {len(ld_ts)} videos...")
    
    for batch in tqdm(ld_ts, desc="TTA Inference"):
        gatekeeper.reset()
        
        video = batch["video"].to(device)
        target = batch["target"].to(device)
        case = batch["case"][0]
        
        # Prepare batch data for Cloud Simulation
        batch_data = None
        if simulate_cloud:
             batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        output, meta = engine.predict_step(video, batch_data=batch_data, simulate_cloud=simulate_cloud)
        
        # Parse Output (Assumes Regression or scalar EF for now based on previous code)
        pred_ef = 0.0
        if isinstance(output, dict):
            pred_ef = output.get('pred_ef', output.get('mask_logits', 0.0)).item()
        else:
            pred_ef = output.item()
            
        res_entry = {
            "FileName": case,
            "Actual_EF": target.item(),
            "Predicted_EF": pred_ef,
            "Uncertainty": meta['uncertainty'],
            "Cloud_Triggered": meta['cloud_triggered'],
            "Adaptation_Step": meta['adaptation_step']
        }
        results.append(res_entry)
        
    df_results = pd.DataFrame(results)
    save_path = cfg['training']['clinical_pairs_csv'].replace(".csv", "_tta.csv")
    df_results.to_csv(save_path, index=False)
    logger.info(f"TTA Results saved to {save_path}")
    
    mae = (df_results["Actual_EF"] - df_results["Predicted_EF"]).abs().mean()
    logger.info(f"TTA MAE: {mae:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Reliable Echo Segmentation - TTA")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--cloud_simulation", action="store_true", help="Simulate Cloud Oracle during TTA")

    args = parser.parse_args()
    cfg = load_config(args.config)
    
    setup_dist()
    device = run_init(cfg, args_resume=False)
    
    try:
        run_tta(cfg, device, simulate_cloud=args.cloud_simulation)
    finally:
        cleanup_dist()

if __name__ == "__main__":
    main()
