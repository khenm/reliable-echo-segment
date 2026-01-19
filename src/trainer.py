import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric, MAEMetric
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from src.utils.logging import get_logger

logger = get_logger()

class Trainer:
    """
    Handles the training, validation, and testing of the segmentation model.
    """
    def __init__(self, model, loaders, cfg, device):
        """
        Initializes the Trainer.
        
        Args:
            model (torch.nn.Module): The model to train.
            loaders (tuple): (train_loader, val_loader, test_loader).
            cfg (dict): Configuration dictionary.
            device (torch.device): Device to run on.
        """
        self.model = model
        self.ld_tr, self.ld_va, self.ld_ts = loaders
        self.cfg = cfg
        self.device = device
        self.num_classes = cfg['data']['num_classes']
        self.ckpt_path = cfg['training']['ckpt_save_path']
        
        self.model_name = cfg['model'].get('name', 'VAEUNet')
        self.is_regression = (self.model_name == "R2Plus1D")

        if self.is_regression:
            self.loss_fn = nn.MSELoss()
            self.metr_mae_val = MAEMetric(reduction="mean")
        else:
            self.loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
            self.metr_dice_val = DiceMetric(include_background=False, reduction="mean")
            
        self.opt = torch.optim.AdamW(model.parameters(), 
                                     lr=cfg['training']['lr'], 
                                     weight_decay=cfg['training']['weight_decay'])
        
        dev_type = device.type if hasattr(device, 'type') else str(device)
        self.scaler = torch.amp.GradScaler(device=dev_type, enabled=(dev_type == 'cuda'))
        self.kl_weight = cfg['training'].get('kl_weight', 1e-4) # Beta parameter

    def _load_checkpoint(self, path, load_optimizer=False):
        """
        Loads a checkpoint, handling both legacy (weights only) and new (full dict) formats.
        
        Args:
            path (str): Path to the checkpoint file.
            load_optimizer (bool): Whether to load optimizer state.
            
        Returns:
            tuple: (start_epoch, best_metric)
        """
        logger.info(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        start_epoch = 1
        best_metric = 0.0
        
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            # New format
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if load_optimizer and "optimizer_state_dict" in checkpoint:
                self.opt.load_state_dict(checkpoint["optimizer_state_dict"])
            if "epoch" in checkpoint:
                start_epoch = checkpoint["epoch"] + 1
            if "best_metric" in checkpoint:
                best_metric = checkpoint["best_metric"]
            logger.info(f"Loaded checkpoint (epoch {checkpoint.get('epoch')}, best_metric {best_metric:.4f})")
        else:
            # Legacy format (assuming it's just state_dict)
            self.model.load_state_dict(checkpoint)
            logger.info("Loaded legacy checkpoint (model weights only). Resetting optimizer/epoch.")
            
        return start_epoch, best_metric

    def train(self):
        """
        Executes the training loop with early stopping.
        """
        epochs = self.cfg['training']['epochs']
        patience = self.cfg['training']['patience']
        best_metric = 0.0
        wait = 0
        stop_ep = epochs

        start_tr = time.time()
        
        start_epoch = 1
        if 'resume_path' in self.cfg['training']:
            start_epoch, best_metric = self._load_checkpoint(self.cfg['training']['resume_path'], load_optimizer=True)

        for ep in range(start_epoch, epochs + 1):
            self.model.train()
            run_loss = 0.0
            
            for batch in self.ld_tr:
                self.opt.zero_grad(set_to_none=True)
                
                dev_type = self.device.type if hasattr(self.device, 'type') else str(self.device)
                
                if self.is_regression:
                    # Generic handling for Video Dataset (returns 'video', 'target')
                    imgs = batch.get("video", batch.get("image")).to(self.device)
                    targets = batch["target"].to(self.device)
                    
                    with torch.amp.autocast(device_type=dev_type):
                        preds = self.model(imgs)
                        loss = self.loss_fn(preds.squeeze(), targets)
                else:
                    imgs = batch["image"].to(self.device)
                    labs = batch["label"].to(self.device)
                
                    with torch.amp.autocast(device_type=dev_type):
                        logits, mu, log_var = self.model(imgs)
                        dice_ce_loss = self.loss_fn(logits, labs)
                        
                        # KL Divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
                        # Sum over latent dim, mean over batch usually
                        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
                        kl_loss = torch.mean(kl_div)
                        
                        loss = dice_ce_loss + self.kl_weight * kl_loss
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()
                run_loss += loss.item()

            val_dice = self._validate()
            val_dice = self._validate()
            logger.info(f"E{ep:03d} trainLoss={run_loss/len(self.ld_tr):.4f} (KL={kl_loss.item():.6f}) valDice={val_dice:.4f}")

            if val_dice > best_metric:
                logger.info(f"Saving BEST checkpoint to {self.ckpt_path}...")
                torch.save({
                    'epoch': ep,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.opt.state_dict(),
                    'best_metric': val_dice
                }, self.ckpt_path)
                logger.info("Checkpoint saved.")
                best_metric = val_dice
                wait = 0
            else:
                wait += 1
            
            # Save Latest Checkpoint
            last_ckpt_path = self.ckpt_path.replace(".pt", "_last.pt")
            torch.save({
                'epoch': ep,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.opt.state_dict(),
                'best_metric': best_metric
            }, last_ckpt_path)
            
            if wait >= patience:
                logger.info("⏹ Early stop")
                stop_ep = ep
                break
        
        train_time = time.time() - start_tr
        logger.info(f"🏁 Finished at epoch {stop_ep} (best={best_metric:.4f}) in {train_time/60:.1f} min")

    def _validate(self):
        """
        Runs validation pass and returns the mean Dice score.
        """
        self.model.eval()
        
        if self.is_regression:
             self.metr_mae_val.reset()
        else:
             self.metr_dice_val.reset()
        
        dev_type = self.device.type if hasattr(self.device, 'type') else str(self.device)
        with torch.no_grad(), torch.amp.autocast(device_type=dev_type):
            for vb in self.ld_va:
                if self.is_regression:
                    v_img = vb.get("video", vb.get("image")).to(self.device)
                    v_target = vb["target"].to(self.device)
                    
                    v_preds = self.model(v_img)
                    self.metr_mae_val(v_preds.squeeze(), v_target)
                else:
                    v_img = vb["image"].to(self.device)
                    v_lab = vb["label"].to(self.device)

                    v_logits, _, _ = self.model(v_img)
                    v_pred_labels = torch.argmax(v_logits, dim=1)

                    if v_lab.ndim == 4 and v_lab.shape[1] == 1:
                        v_gt_labels = v_lab[:, 0].long()
                    else:
                        raise RuntimeError(f"Unexpected val GT shape {v_lab.shape}")

                    v_y_pred = F.one_hot(v_pred_labels, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
                    v_y_true = F.one_hot(v_gt_labels, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

                    self.metr_dice_val(v_y_pred, v_y_true)
        
        if self.is_regression:
            # For MAE, lower is better. Trainer logic assumes higher is better (saving best checkpoint).
            # We can return negative MAE or invert the logic in train loop.
            # Let's return negative MAE so higher is better (closer to 0).
            return -float(self.metr_mae_val.aggregate().cpu())
        else:
            return float(self.metr_dice_val.aggregate().cpu())

    def evaluate_test(self):
        """
        Evaluates the best model on the test set.
        For Segmentation: Dice and HD95 metrics.
        For Regression: MAE and R2 (or just saves predictions).
        
        Returns:
            pd.DataFrame: DataFrame containing per-sample metrics.
        """
        logger.info(f"Loading best checkpoint from: {self.ckpt_path}")
        self._load_checkpoint(self.ckpt_path, load_optimizer=False)
        self.model.eval()

        records = []
        
        if self.is_regression:
             MAE_metric = MAEMetric(reduction="none")
             
             with torch.no_grad():
                for batch in self.ld_ts:
                    imgs = batch.get("video", batch.get("image")).to(self.device)
                    targets = batch["target"].to(self.device)
                    cases = batch["case"]
                    
                    preds = self.model(imgs).squeeze()
                    
                    MAE_metric(preds, targets)
                    mae_vals = torch.abs(preds - targets).cpu().numpy()
                    
                    preds_np = preds.cpu().numpy()
                    targets_np = targets.cpu().numpy()
                    
                    for i in range(len(imgs)):
                         row = {
                             "case": cases[i],
                             "target_EF": targets_np[i],
                             "pred_EF": preds_np[i],
                             "MAE": mae_vals[i]
                         }
                         records.append(row)
             
             df = pd.DataFrame(records)
             overall_mae = df["MAE"].mean()
             logger.info(f"Test Set Overall MAE: {overall_mae:.4f}")

        else:
            dice_metric = DiceMetric(include_background=False, reduction="none")
            hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="none")

            with torch.no_grad():
                for batch in self.ld_ts:
                    imgs = batch["image"].to(self.device)
                    gts = batch["label"].to(self.device)
                    cases, views, phases = batch["case"], batch["view"], batch["phase"]

                    logits, _, _ = self.model(imgs)
                    pred_labels = torch.argmax(logits, dim=1)

                    if gts.ndim == 4 and gts.shape[1] == 1:
                        gt_labels = gts[:, 0].long()
                    else:
                        gt_labels = gts.long()

                    y_pred_all = F.one_hot(pred_labels, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
                    y_true_all = F.one_hot(gt_labels, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

                    for i in range(len(imgs)):
                        y_p, y_t = y_pred_all[i:i+1], y_true_all[i:i+1]
                        
                        dice_metric.reset()
                        hd95_metric.reset()
                        dice_metric(y_p, y_t)
                        hd95_metric(y_p, y_t)
                        
                        dice_vals = dice_metric.aggregate().cpu().numpy().flatten()
                        hd95_vals = hd95_metric.aggregate().cpu().numpy().flatten()

                        row = {
                            "case": cases[i], "view": views[i], "phase": phases[i],
                            "dice_LV": dice_vals[0] if len(dice_vals) > 0 else 0.0,
                            "hd95_LV": hd95_vals[0] if len(hd95_vals) > 0 else 0.0,
                        }
                        
                        if len(dice_vals) >= 3:
                            row.update({
                                "dice_MYO": dice_vals[1], "dice_LA": dice_vals[2],
                                "hd95_MYO": hd95_vals[1], "hd95_LA": hd95_vals[2],
                            })
                        else:
                             row.update({
                                "dice_MYO": 0.0, "dice_LA": 0.0,
                                "hd95_MYO": 0.0, "hd95_LA": 0.0,
                            })

                        records.append(row)

            df = pd.DataFrame(records)
            
        df.to_csv(self.cfg['training']['test_metrics_csv'], index=False)
        logger.info(f"Saved metrics to {self.cfg['training']['test_metrics_csv']}")
        return df