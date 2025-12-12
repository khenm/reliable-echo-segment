import yaml
import argparse
from src.utils.util_ import seed_everything, get_device
from src.dataset import get_dataloaders
from src.models.model import get_model
from src.trainer import Trainer

def main(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg['training']['seed'])
    device = get_device()
    print(f"Using device: {device}")

    # Data
    print("Initializing DataLoaders...")
    loaders = get_dataloaders(cfg)

    # Model
    print("Initializing Model...")
    model = get_model(cfg, device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training
    trainer = Trainer(model, loaders, cfg, device)
    trainer.train()

    # Evaluation
    print("Starting Evaluation...")
    trainer.evaluate_test()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)