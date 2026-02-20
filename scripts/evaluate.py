import os
os.environ['MPLBACKEND'] = 'Agg'
import argparse

from src.utils.config_loader import load_config
from src.utils.dist import setup_dist, cleanup_dist
from src.registry import get_dataloaders
from src.trainer import Trainer
from src.utils.plot import plot_clinical_comparison
from src.utils.logging import get_logger
from src.utils.runner import run_init, load_model_for_inference, get_metrics

logger = get_logger()

def run_eval(cfg, device):
    logger.info("Starting Evaluation...")
    loaders = get_dataloaders(cfg)
    # loaders is (train_loader, val_loader, test_loader)
    
    model = load_model_for_inference(cfg, device)
    if model is None:
        logger.error("Skipping evaluation due to missing checkpoint.")
        return
    
    metrics = get_metrics(cfg)
    trainer = Trainer(model, loaders, cfg, device, metrics=metrics)
    trainer.evaluate_test()

    logger.info("Generating visualization examples...")
    try:
        samples = trainer.get_examples(num_examples=3)
        for s in samples:
            save_name = os.path.join(cfg['training']['run_dir'], f"vis_{s['fname']}_f{s['frame_idx']}.png")
            plot_clinical_comparison(s, save_path=save_name)
    except Exception as e:
        logger.error(f"Failed to plot visualization examples: {e}")

def main():
    parser = argparse.ArgumentParser(description="Reliable Echo Segmentation - Evaluation")
    parser.add_argument("--config", type=str, default="configs/config.yaml")

    args = parser.parse_args()
    cfg = load_config(args.config)
    
    setup_dist()
    device = run_init(cfg, args_resume=False) 
    
    try:
        run_eval(cfg, device)
    finally:
        cleanup_dist()

if __name__ == "__main__":
    main()
