import argparse
import matplotlib
matplotlib.use('Agg')
import wandb
from src.utils.config_loader import load_config
from src.utils.dist import setup_dist, cleanup_dist
from src.registry import get_dataloaders, build_model
from src.trainer import Trainer
from src.utils.logging import get_logger
from src.utils.runner import run_init, get_criterions, get_metrics

logger = get_logger()

def run_train(cfg, device):
    logger.info("Starting Training...")
    loaders = get_dataloaders(cfg)
    model = build_model(cfg, device)
    
    if cfg.get('wandb', {}).get('enable', True):
        if wandb.run is not None:
            wandb.watch(model, log="gradients", log_freq=100)
    
    criterions = get_criterions(cfg)
    metrics = get_metrics(cfg)

    trainer = Trainer(model, loaders, cfg, device, criterions=criterions, metrics=metrics)
    trainer.train()

def main():
    parser = argparse.ArgumentParser(description="Reliable Echo Segmentation - Training")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint")
    parser.add_argument("--pretrain", action="store_true", help="Run in pretraining mode (filter clips to contain both ED and ES)")

    args = parser.parse_args()

    cfg = load_config(args.config)
    
    # Override wandb enable status based on flag
    if 'wandb' not in cfg: cfg['wandb'] = {}
    cfg['wandb']['enable'] = args.wandb
    
    # Set pretrain mode
    if 'training' not in cfg: cfg['training'] = {}
    cfg['training']['pretrain'] = args.pretrain

    setup_dist()
    device = run_init(cfg, args.resume)
    
    try:
        run_train(cfg, device)
    finally:
        cleanup_dist()

if __name__ == "__main__":
    main()
