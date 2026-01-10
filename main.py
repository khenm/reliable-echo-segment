import yaml
import argparse
import os
import torch
import pandas as pd
from datetime import datetime
from src.utils.util_ import seed_everything, get_device
from src.dataset import get_dataloaders
from src.models.model import get_model
from src.trainer import Trainer
from src.utils.logging import get_logger
from src.utils.postprecessing import generate_clinical_pairs
from src.utils.plot import (
    plot_metrics_summary, 
    plot_clinical_bland_altman, 
    plot_reliability_curves, 
    plot_ef_category_roc
)
from src.eval_rcps import run_rcps_pipeline
from src.analysis.latent_profile import LatentProfiler
from src.eval_adaptive import run_adaptive_pipeline

# Setup logging (stdout only initially)
logger = get_logger()

def run_init(cfg):
    """
    Initializes the environment, logging, and output directories.

    Args:
        cfg (dict): Configuration dictionary.

    Returns:
        torch.device: The device (CPU or CUDA) to be used.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_root = cfg['training'].get('save_dir', 'runs')
    run_dir = os.path.join(save_root, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    cfg['training']['run_dir'] = run_dir
    
    log_file = os.path.join(run_dir, "run.log")
    _ = get_logger(log_file=log_file)
    
    logger.info("Initializing...")
    logger.info(f"Run Directory: {run_dir}")
    logger.info("Settings: " + str(cfg['training']))
    seed_everything(cfg['training']['seed'])
    device = get_device()
    logger.info(f"Using device: {device}")
    
    if "/" in ckpt_name:
        ckpt_name = os.path.basename(ckpt_name)
    
    ckpt_dir = cfg['training'].get('checkpoint_dir', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    
    cfg['training']['ckpt_save_path'] = os.path.join(ckpt_dir, f"run_{timestamp}_{ckpt_name}")
    
    metrics_csv = cfg['training']['test_metrics_csv']
    if "/" in metrics_csv:
        metrics_csv = os.path.basename(metrics_csv)
    cfg['training']['test_metrics_csv'] = os.path.join(run_dir, metrics_csv)
    
    cfg['training']['clinical_pairs_csv'] = os.path.join(run_dir, "camus_clinical_pairs.csv")

    return device

def run_preprocess(cfg):
    """
    Verifies data availability and loading by realizing the DataLoaders.

    Args:
        cfg (dict): Configuration dictionary.
    """
    logger.info("Checking data loading...")
    get_dataloaders(cfg)
    logger.info("Data verification complete.")

def run_train(cfg, device):
    """
    Instantiates the model and trainer, then starts the training loop.

    Args:
        cfg (dict): Configuration dictionary.
        device (torch.device): The computation device.
    """
    logger.info("Starting Training...")
    loaders = get_dataloaders(cfg)
    model = get_model(cfg, device)
    
    trainer = Trainer(model, loaders, cfg, device)
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
    loaders = get_dataloaders(cfg)
    _, _, ld_ts = loaders
    
    model = get_model(cfg, device)
    ckpt_path = cfg['training']['ckpt_save_path']
    if not os.path.exists(ckpt_path):
        logger.error(f"Checkpoint not found at {ckpt_path}. Skipping evaluation.")
        return

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    
    # Standard Dice/HD statistics
    trainer = Trainer(model, loaders, cfg, device)
    trainer.evaluate_test()
    
    # Clinical Metrics (EF, Volumes)
    ef_save_path = cfg['training']['clinical_pairs_csv']
    generate_clinical_pairs(model, ld_ts, device, ef_save_path)

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
    
    model = get_model(cfg, device)
    ckpt_path = cfg['training']['ckpt_save_path']
    if not os.path.exists(ckpt_path):
        if os.path.exists(os.path.join("checkpoints", ckpt_path)):
             ckpt_path = os.path.join("checkpoints", ckpt_path)
             
    if not os.path.exists(ckpt_path):
        logger.error(f"Checkpoint not found at {ckpt_path}. Skipping profiling.")
        return

    logger.info(f"Loading checkpoint: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    
    loaders = get_dataloaders(cfg)
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
    parser.add_argument("--adaptive", action="store_true", help="Run Adaptive Calibration")
    parser.add_argument("--profile", action="store_true", help="Run Latent Profiling")
    parser.add_argument("--plot", action="store_true", help="Generate all plots")
    parser.add_argument("--all", action="store_true", help="Run full pipeline")
    
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    run_all = args.all or not (args.init or args.preprocess or args.train or args.eval or args.plot or args.rcps or args.profile or args.adaptive)

    device = run_init(cfg)

    needs_checkpoint = args.eval or args.rcps or args.profile or args.adaptive or args.plot
    is_training = args.train or args.all
    
    if needs_checkpoint and not is_training:
        current_ckpt = cfg['training']['ckpt_save_path']
        if not os.path.exists(current_ckpt):
            ckpt_dir = os.path.dirname(current_ckpt)
            logger.info(f"Checkpoint {current_ckpt} not found. Searching for latest in {ckpt_dir}...")
            
            latest_ckpt = find_latest_checkpoint(ckpt_dir)
            if latest_ckpt:
                logger.info(f"Using latest checkpoint: {latest_ckpt}")
                cfg['training']['ckpt_save_path'] = latest_ckpt
            else:
                logger.warning(f"No checkpoint found in {ckpt_dir}. Subsequent steps may fail.")

    if run_all or args.preprocess:
        run_preprocess(cfg)
    
    if run_all or args.train:
        run_train(cfg, device)
        
    if run_all or args.eval:
        run_eval(cfg, device)
        
    if run_all or args.plot:
        run_plot(cfg)
        
    if run_all or args.rcps:
        run_rcps_pipeline(cfg)
        
    if run_all or args.profile:
        run_profile(cfg, device)
        
    if run_all or args.adaptive:
        run_adaptive_pipeline(cfg)

def find_latest_checkpoint(ckpt_dir):
    """
    Finds the most recently modified checkpoint file in the directory.

    Args:
        ckpt_dir (str): Directory path content checkpoint files.

    Returns:
        str | None: Path to the latest checkpoint file, or None if not found.
    """
    if not os.path.exists(ckpt_dir):
        return None
    files = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    if not files:
        return None
    return max(files, key=os.path.getmtime)

if __name__ == "__main__":
    main()