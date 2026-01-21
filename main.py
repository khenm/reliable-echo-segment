import yaml
from src.utils.config_loader import load_config
import argparse
import os
import torch
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from src.utils.util_ import seed_everything, get_device, load_checkpoint
from src.datasets.registry import DatasetRegistry
import src.datasets
from src.models.registry import get_model
import src.models.r2plus1d # Register R2Plus1D
import src.models.vae_unet # Register VAEUNet
from src.trainer import Trainer
from src.utils.logging import get_logger
from src.losses import KLLoss, DifferentiableEFLoss
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, MAEMetric
from src.utils.postprecessing import generate_clinical_pairs
from src.utils.plot import (
    plot_metrics_summary, 
    plot_clinical_bland_altman, 
    plot_reliability_curves, 
    plot_ef_category_roc
)
from src.analysis.latent_profile import LatentProfiler
from src.tta.engine import TTA_Engine

# Setup logging (stdout only initially)
logger = get_logger()

def resolve_resume_path(resume_mode, runs_root, model_name):
    """
    Resolves the checkpoint path for resuming based on the mode.
    
    Args:
        resume_mode (str|bool): 'auto', path string, or True (treated as 'auto').
        runs_root (str): Root directory for runs (e.g., 'runs').
        model_name (str): Name of the model (for directory scoping).
        
    Returns:
        str | None: Path to the checkpoint to resume from, or None if fresh run.
    """
    # If mode is None or False, fresh run
    if not resume_mode:
        return None
        
    # If mode is a specific path, verify and return
    if isinstance(resume_mode, str) and resume_mode != 'auto' and "/" in resume_mode:
        if os.path.exists(resume_mode):
            return resume_mode
        else:
            logger.warning(f"Explicit resume path {resume_mode} not found. Starting fresh.")
            return None
            
    # Auto-Resume Logic
    model_runs_dir = os.path.join(runs_root, model_name)
    if not os.path.exists(model_runs_dir):
        logger.info(f"No previous runs found for {model_name}. Starting fresh.")
        return None
        
    # List all run directories, sorted by creation time (descending)
    runs = sorted([
        os.path.join(model_runs_dir, d) for d in os.listdir(model_runs_dir) 
        if os.path.isdir(os.path.join(model_runs_dir, d))
    ], key=os.path.getmtime, reverse=True)
    
    for run in runs:
        last_ckpt = os.path.join(run, "last.ckpt")
        if os.path.exists(last_ckpt):
            logger.info(f"Auto-Discovery: Found resume point at {last_ckpt}")
            return last_ckpt
            
    logger.info("Auto-Discovery: No 'last.ckpt' found in recent history. Starting fresh.")
    return None

def run_init(cfg, args_resume):
    """
    Initializes the environment, logging, and output directories.
    Enforces Split-Storage:
    - Workspace: runs/{model_name}/{timestamp}/ (Logs, Config, last.ckpt)
    - Vault: checkpoints/{model_name}/ (best artifacts)

    Args:
        cfg (dict): Configuration dictionary.
        args_resume (bool): Command line resume flag.

    Returns:
        torch.device: The device (CPU or CUDA) to be used.
    """
    model_name = cfg['model'].get('name', 'VAEUNet')
    
    # 1. Directory Setup
    # Workspace
    runs_root = cfg['training'].get('save_dir', 'runs')
    os.makedirs(runs_root, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Vault 
    ckpt_root = cfg['training'].get('checkpoint_dir', 'checkpoints')
    os.makedirs(ckpt_root, exist_ok=True)

    vault_dir = os.path.join(ckpt_root, model_name)
    os.makedirs(vault_dir, exist_ok=True)
    
    # Determine Resume State
    resume_mode = 'auto' if args_resume else cfg['training'].get('resume_mode', None)
    resume_path = resolve_resume_path(resume_mode, runs_root, model_name)
    
    if resume_path:
        # Resume from existing run directory
        # The run_dir is the parent of the checkpoint
        run_dir = os.path.dirname(resume_path)
        logger.info(f"Resuming workspace: {run_dir}")
        cfg['training']['resume_path'] = resume_path
    else:
        # Create NEW Workspace
        run_dir = os.path.join(runs_root, model_name, timestamp)
        os.makedirs(run_dir, exist_ok=True)
        logger.info(f"Created new workspace: {run_dir}")
    
    cfg['training']['run_dir'] = run_dir
    cfg['training']['vault_dir'] = vault_dir
    
    # Setup Logging
    log_file = os.path.join(run_dir, "run.log")
    _ = get_logger(log_file=log_file)
    
    # 2. Dump Configuration (Immutable Snapshot)
    with open(os.path.join(run_dir, "config.yaml"), 'w') as f:
        yaml.dump(cfg, f)
    
    logger.info("Initializing...")
    logger.info(f"Workspace: {run_dir}")
    logger.info(f"Vault: {vault_dir}")
    
    seed_everything(cfg['training']['seed'])
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Update paths in config for other components
    cfg['training']['ckpt_save_path'] = os.path.join(run_dir, "last.ckpt") # Workspace path
    
    metrics_csv = cfg['training'].get('test_metrics_csv', 'test_metrics.csv')
    if "/" in metrics_csv: metrics_csv = os.path.basename(metrics_csv)
    cfg['training']['test_metrics_csv'] = os.path.join(run_dir, metrics_csv)
    
    clinical_csv = cfg['training'].get('clinical_pairs_csv', 'clinical_pairs.csv')
    if "/" in clinical_csv: clinical_csv = os.path.basename(clinical_csv)
    cfg['training']['clinical_pairs_csv'] = os.path.join(run_dir, clinical_csv)

    return device

def run_preprocess(cfg):
    """
    Verifies data availability and loading by realizing the DataLoaders.

    Args:
        cfg (dict): Configuration dictionary.
    """
    logger.info("Checking data loading...")
    loaders = DatasetRegistry.get_dataloaders(cfg)
    
    # Iterate through all loaders to ensure data is readable
    for i, loader in enumerate(loaders):
        if loader is None: continue
        name = ["Train", "Val", "Test"][i]
        for _ in tqdm(loader, desc=f"Verifying {name}", mininterval=1.0):
            pass
            
    logger.info("Data verification complete.")

def _load_model_for_inference(cfg, device):
    """
    Instantiates model and loads weights from configured checkpoint path.
    Handles existence checks and logging.
    
    Args:
        cfg (dict): Configuration dictionary.
        device (torch.device): Computation device.
        
    Returns:
        torch.nn.Module | None: Loaded model or None if checkpoint missing.
    """
    ckpt_path = cfg['training']['ckpt_save_path']
    
    # Robust check (handle relative path in checkpoints dir if not found absolute)
    if not os.path.exists(ckpt_path):
        alt_path = os.path.join("checkpoints", ckpt_path)
        if os.path.exists(alt_path):
             ckpt_path = alt_path
    
    if not os.path.exists(ckpt_path):
        logger.error(f"Checkpoint not found at {ckpt_path}. Cannot proceed.")
        return None
        
    model = get_model(cfg, device)
    logger.info(f"Loading checkpoint: {ckpt_path}")
    load_checkpoint(model, ckpt_path, device)
    model.eval()
    return model

def run_train(cfg, device):
    """
    Instantiates the model and trainer, then starts the training loop.

    Args:
        cfg (dict): Configuration dictionary.
        device (torch.device): The computation device.
    """
    logger.info("Starting Training...")
    loaders = DatasetRegistry.get_dataloaders(cfg)
    model = get_model(cfg, device)
    
    # Define Criterions based on Model Type
    model_name = cfg['model'].get('name', 'VAEUNet')
    is_regression = (model_name.lower() == "r2plus1d")
    
    criterions = {}
    weights = cfg.get('loss', {}).get('weights', {})

    if is_regression:
        if weights.get('ef', 0.0) > 0:
            criterions['ef'] = DifferentiableEFLoss(pixel_spacing=1.0, weight=weights.get('ef'))
             
        num_classes = cfg['data'].get('num_classes', 1)
        criterions['seg'] = DiceCELoss(sigmoid=True, 
                                       lambda_dice=weights.get('dice', 0.7), 
                                       lambda_ce=weights.get('ce', 0.3))
    else:
        criterions['dice'] = DiceCELoss(to_onehot_y=True, softmax=True)
        kl_weight = weights.get('kl', 1e-4)
        criterions['kl'] = KLLoss(weight=kl_weight)

    # Define Metrics based on Model Type
    num_classes = cfg['data'].get('num_classes', 1)
    include_bg = (num_classes == 1)
    
    metrics = {}
    if is_regression:
        metrics['mae'] = MAEMetric(reduction="mean")
        metrics['dice'] = DiceMetric(include_background=include_bg, reduction="mean")
    else:
        metrics['dice'] = DiceMetric(include_background=include_bg, reduction="mean")

    trainer = Trainer(model, loaders, cfg, device, criterions=criterions, metrics=metrics)
    trainer.train()

def run_eval(cfg, device):
    """
    Runs evaluation on the test set involving both technical (Dice/HD95) 
    and clinical (EF/Area) metrics.

    Args:
        cfg (dict): Configuration dictionary.
        device (torch.device): The computation device.
    """
    logger.info("Starting Evaluation...")
    loaders = DatasetRegistry.get_dataloaders(cfg)
    _, _, ld_ts = loaders
    
    model = _load_model_for_inference(cfg, device)
    if model is None:
        logger.error("Skipping evaluation due to missing checkpoint.")
        return
    
    # Standard Dice/HD statistics
    trainer = Trainer(model, loaders, cfg, device)
    trainer.evaluate_test()
    
    # Clinical Metrics (EF, Volumes)
    ef_save_path = cfg['training']['clinical_pairs_csv']
    generate_clinical_pairs(model, ld_ts, device, ef_save_path)

def run_tta(cfg, device):
    """
    Runs Test-Time Adaptation on the test set.
    """
    logger.info("Starting Test-Time Adaptation (TTA)...")
    loaders = DatasetRegistry.get_dataloaders(cfg)
    _, _, ld_ts = loaders
    
    
    model = _load_model_for_inference(cfg, device)
    if model is None:
        logger.error("Skipping TTA due to missing checkpoint.")
        return
    
    # Configure TTA Engine
    tta_cfg = cfg.get('tta', {})
    engine = TTA_Engine(
        model, 
        lr=float(tta_cfg.get('lr', 1e-4)), 
        n_augments=int(tta_cfg.get('n_augments', 4)),
        steps=int(tta_cfg.get('steps', 1)),
        optimizer_name=tta_cfg.get('optimizer', 'SGD')
    )
    
    engine.model.to(device)
    
    # Run TTA Loop
    results = []
    
    logger.info(f"Running TTA on {len(ld_ts)} videos...")
    
    for batch in tqdm(ld_ts, desc="TTA Inference"):
        # Reset model state before each video
        engine.reset()
        
        # batch is dict: "video", "target", "case"
        # video: (B, C, T, H, W) -> B=1 usually
        video = batch["video"].to(device)
        target = batch["target"].to(device)
        case = batch["case"][0] # Assume batch size 1
        
        # Adapt and Predict
        pred = engine.forward_and_adapt(video)
        
        results.append({
            "FileName": case,
            "Actual_EF": target.item(),
            "Predicted_EF": pred.item()
        })
        
    # Save Results
    df_results = pd.DataFrame(results)
    save_path = cfg['training']['clinical_pairs_csv'].replace(".csv", "_tta.csv")
    df_results.to_csv(save_path, index=False)
    logger.info(f"TTA Results saved to {save_path}")
    
    # Simple Metrics
    mae = (df_results["Actual_EF"] - df_results["Predicted_EF"]).abs().mean()
    logger.info(f"TTA MAE: {mae:.2f}")

def run_plot(cfg):
    """
    Generates summary plots from validation and testing results.

    Args:
        cfg (dict): Configuration dictionary.
    """
    logger.info("Generating plots...")
    metrics_path = cfg['training']['test_metrics_csv']
    ef_path = cfg['training']['clinical_pairs_csv']
    
    if not os.path.exists(metrics_path):
        logger.warning(f"Metrics file {metrics_path} not found. Skipping metrics plots.")
    else:
        df_metrics = pd.read_csv(metrics_path)
        plot_metrics_summary(df_metrics, save_path=os.path.join(cfg['training']['run_dir'], "plot_metrics_summary.png"))
        logger.info("Generated plot_metrics_summary.png")

    if not os.path.exists(ef_path):
         logger.warning(f"Clinical file {ef_path} not found. Skipping clinical plots.")
    else:
        df_ef = pd.read_csv(ef_path)
        if os.path.exists(metrics_path):
            df_metrics = pd.read_csv(metrics_path) 
            plot_reliability_curves(df_metrics, df_ef, save_path=os.path.join(cfg['training']['run_dir'], "plot_reliability.png"))
        
        plot_clinical_bland_altman(df_ef, save_path=os.path.join(cfg['training']['run_dir'], "plot_bland_altman.png"))
        plot_ef_category_roc(df_ef, save_path=os.path.join(cfg['training']['run_dir'], "plot_ef_roc.png"))
        logger.info("Generated clinical plots.")

def run_profile(cfg, device):
    """
    Runs the latent space profiling on the training set.

    Args:
        cfg (dict): Configuration dictionary.
        device (torch.device): The computation device.
    """
    logger.info("Starting Latent Profiling...")
    
    model = _load_model_for_inference(cfg, device)
    if model is None:
        logger.error("Skipping profiling due to missing checkpoint.")
        return
    
    loaders = DatasetRegistry.get_dataloaders(cfg)
    ld_tr, _, _ = loaders
    
    latent_dim = cfg['model']['latent_dim']
    profiler = LatentProfiler(latent_dim=latent_dim)
    
    profiler.fit(model, ld_tr, device)
    
    run_dir = cfg['training']['run_dir']
    save_path = os.path.join(run_dir, "latent_profile.joblib")
    profiler.save(save_path)
    logger.info(f"Latent profile saved to {save_path}")

def main():
    """
    Main entry point for the Reliable Echo Segmentation Pipeline.
    Parses arguments and orchestrates the execution of pipeline steps.
    """
    parser = argparse.ArgumentParser(description="Reliable Echo Segmentation Pipeline")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--init", action="store_true", help="Run initialization check")
    parser.add_argument("--preprocess", action="store_true", help="Run data loading verification")
    parser.add_argument("--train", action="store_true", help="Run training loop")
    parser.add_argument("--eval", action="store_true", help="Run evaluation (metrics + EF)")
    parser.add_argument("--rcps", action="store_true", help="Run RCPS calibration and evaluation")
    parser.add_argument("--tta", action="store_true", help="Run Test-Time Adaptation")
    parser.add_argument("--adaptive", action="store_true", help="Run Adaptive Calibration")
    parser.add_argument("--profile", action="store_true", help="Run Latent Profiling")
    parser.add_argument("--plot", action="store_true", help="Generate all plots")
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    
    args = parser.parse_args()

    cfg = load_config(args.config)

    run_all = args.all or not (args.init or args.preprocess or args.train or args.eval or args.plot or args.rcps or args.profile or args.adaptive or args.tta)

    device = run_init(cfg, args.resume)
    
    if args.resume and 'resume_path' not in cfg['training']:
        logger.warning("Resume requested but no previous run found. Starting fresh (unless specific path provided later).")

    if run_all or args.preprocess:
        run_preprocess(cfg)
    
    if run_all or args.train:
        run_train(cfg, device)

    if run_all or args.profile:
        run_profile(cfg, device)
        
    if run_all or args.tta:
        run_tta(cfg, device)
        
    if run_all or args.eval:
        run_eval(cfg, device)
        
    if run_all or args.plot:
        run_plot(cfg)

if __name__ == "__main__":
    main()