import argparse
import yaml
import torch
from src.utils.util_ import seed_everything
from src.dataset import get_loaders
from src.models.base_model import MyModel
from src.trainer import Trainer

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main(args):
    # 1. Load Config
    cfg = load_config(args.config)
    print(f"Starting experiment: {cfg['experiment_name']}")

    # 2. Reproducibility
    seed_everything(cfg['seed'])

    # 3. Data
    train_loader, val_loader = get_loaders(cfg['data'])

    # 4. Model
    device = torch.device(cfg['device'] if torch.cuda.is_available() else "cpu")
    model = MyModel(cfg['model']).to(device)

    # 5. Training
    trainer = Trainer(model, train_loader, val_loader, cfg['training'], device)
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    args = parser.parse_args()
    main(args)