import os
import time
import numpy as np
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric, MAEMetric
from tqdm import tqdm
from src.utils.logging import get_logger

logger = get_logger()

class Trainer:
    """
    Handles training, validation, and testing for both segmentation (VAE-U-Net) and 
    dual-task regression/segmentation (R2+1D) models.
    """
    def __init__(self, model, loaders, cfg, device, criterions=None):
        """
        Args:
            model (torch.nn.Module): The model to train.
            loaders (tuple): Tuple containing (train_loader, val_loader, test_loader).
            cfg (dict): Configuration dictionary defining hyperparameters and paths.
            device (torch.device): Device to execute computation on.
            criterions (dict): Dictionary of loss functions. e.g. {'dice': DiceLoss(), 'kl': KLLoss()}
        """
        self.model = model
        self.ld_tr, self.ld_va, self.ld_ts = loaders
        self.cfg = cfg
        self.device = device
        self.criterions = criterions if criterions is not None else {}
        self.num_classes = cfg['data']['num_classes']
        self.ckpt_path = cfg['training']['ckpt_save_path'] # Workspace: .../last.ckpt
        self.vault_dir = cfg['training'].get('vault_dir', None) # Vault: checkpoints/{model_name}/
        
        self.model_name = cfg['model'].get('name', 'VAEUNet')
        self.is_regression = (self.model_name.lower() == "r2plus1d")

        # Metric initialization
        if self.is_regression:
            self.metr_mae_val = MAEMetric(reduction="mean")
            self.metr_dice_val = DiceMetric(include_background=False, reduction="mean")
        else:
            self.metr_dice_val = DiceMetric(include_background=False, reduction="mean")
            
        self.opt = torch.optim.AdamW(model.parameters(), 
                                     lr=cfg['training']['lr'], 
                                     weight_decay=cfg['training']['weight_decay'])
        
        dev_type = device.type if hasattr(device, 'type') else str(device)
        self.scaler = torch.amp.GradScaler(device=dev_type, enabled=(dev_type == 'cuda'))

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
            if load_optimizer:
                if "optimizer_state_dict" in checkpoint:
                    self.opt.load_state_dict(checkpoint["optimizer_state_dict"])
                else:
                    logger.warning("Optimizer state not found in checkpoint. Starting optimizer fresh.")
                
                # Restore RNG State
                if "rng_state" in checkpoint:
                    rng_state = checkpoint["rng_state"]
                    torch.set_rng_state(rng_state["torch"])
                    torch.cuda.set_rng_state(rng_state["cuda"])
                    np.random.set_state(rng_state["numpy"])
                    random.setstate(rng_state["python"])
                    logger.info("Restored RNG states.")
                    
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
            
            # Track individual losses for logging
            epoch_loss_components = {key: 0.0 for key in self.criterions.keys()}

            pbar = tqdm(self.ld_tr, desc=f"Epoch {ep}/{epochs}", mininterval=2.0)
            for batch in pbar:
                self.opt.zero_grad(set_to_none=True)
                
                dev_type = self.device.type if hasattr(self.device, 'type') else str(self.device)
                
                loss = 0.0
                loss_dict = {}

                if self.is_regression:
                    # Generic handling for Video Dataset (returns 'video', 'target', 'label')
                    imgs = batch.get("video", batch.get("image")).to(self.device)
                    targets = batch["target"].to(self.device)
                    mask_targets = batch.get("label")
                    
                    if mask_targets is not None:
                        mask_targets = mask_targets.to(self.device)
                    
                    with torch.amp.autocast(device_type=dev_type):
                        preds, seg_logits = self.model(imgs)
                        
                        # Calculate losses based on keys
                        if 'ef' in self.criterions:
                            # DifferentiableEFLoss expects seg logits and returns (loss, pred_ef)
                            l_ef, _ = self.criterions['ef'](seg_logits, targets)
                            loss += l_ef
                            loss_dict['ef'] = l_ef.item()
                        
                        if 'seg' in self.criterions and mask_targets is not None:
                             l_seg = self.criterions['seg'](seg_logits, mask_targets.long())
                             loss += l_seg
                             loss_dict['seg'] = l_seg.item()

                else:
                    imgs = batch["image"].to(self.device)
                    labs = batch["label"].to(self.device)
                
                    with torch.amp.autocast(device_type=dev_type):
                        logits, mu, log_var = self.model(imgs)
                        
                        if 'dice' in self.criterions:
                            l_dice = self.criterions['dice'](logits, labs)
                            loss += l_dice
                            loss_dict['dice'] = l_dice.item()
                        
                        if 'kl' in self.criterions:
                            l_kl = self.criterions['kl'](mu, log_var)
                            loss += l_kl
                            loss_dict['kl'] = l_kl.item()
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()
                run_loss += loss.item()

                # Update accumulated component losses
                for k, v in loss_dict.items():
                    if k in epoch_loss_components:
                         epoch_loss_components[k] += v
                
                # Pbar update
                pbar_postfix = {"loss": f"{loss.item():.4f}"}
                for k, v in loss_dict.items():
                    pbar_postfix[k] = f"{v:.4f}"
                pbar.set_postfix(pbar_postfix)

            val_dice = self._validate()
            
            # Prepare log message
            avg_run_loss = run_loss / len(self.ld_tr)
            log_msg = f"E{ep:03d} trainLoss={avg_run_loss:.4f} "
            
            for k, v in epoch_loss_components.items():
                avg_comp = v / len(self.ld_tr)
                log_msg += f"{k}={avg_comp:.4f} "
            
            if self.is_regression:
                 log_msg += f"valMAE={-val_dice:.4f}"
            else:
                 log_msg += f"valDice={val_dice:.4f}"
            
            logger.info(log_msg)

            if val_dice > best_metric:
                best_metric = val_dice
                wait = 0
                
                # Save BEST to Vault
                if self.vault_dir:
                    best_ckpt_name = f"{self.cfg['training']['run_dir'].split('/')[-1]}_best.ckpt"
                    best_ckpt_path = os.path.join(self.vault_dir, best_ckpt_name)
                    logger.info(f"Saving BEST checkpoint to Vault: {best_ckpt_path}...")
                    torch.save({
                        'epoch': ep,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.opt.state_dict(),
                        'best_metric': val_dice
                    }, best_ckpt_path)
            else:
                wait += 1
            
            # Save Latest Checkpoint to Workspace (THE STATE DUMP)
            # This is overwriting 'last.ckpt' at every epoch
            torch.save({
                'epoch': ep,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.opt.state_dict(),
                'best_metric': best_metric,
                'rng_state': {
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state(),
                    "numpy": np.random.get_state(),
                    "python": random.getstate()
                }
            }, self.ckpt_path)
            
            if wait >= patience:
                logger.info("⏹ Early stop")
                stop_ep = ep
                break
        
        train_time = time.time() - start_tr
        logger.info(f"Finished at epoch {stop_ep} (best={best_metric:.4f}) in {train_time/60:.1f} min")

    def _validate(self):
        """
        Runs validation pass and returns the mean Dice score.
        """
        self.model.eval()
        
        if self.is_regression:
             self.metr_mae_val.reset()
             self.metr_dice_val.reset()
        else:
             self.metr_dice_val.reset()
        
        dev_type = self.device.type if hasattr(self.device, 'type') else str(self.device)
        with torch.no_grad(), torch.amp.autocast(device_type=dev_type):
            for vb in tqdm(self.ld_va, desc="Validating", mininterval=2.0, leave=False):
                if self.is_regression:
                    v_img = vb.get("video", vb.get("image")).to(self.device)
                    v_target = vb["target"].to(self.device)
                    
                    v_preds, v_seg = self.model(v_img)
                    
                    if v_target.ndim == 1:
                        v_target = v_target.view(-1, 1)
                    self.metr_mae_val(v_preds, v_target)
                    
                    # Dice Metric
                    v_mask_target = vb.get("label")
                    if v_mask_target is not None:
                        v_mask_target = v_mask_target.to(self.device).long()
                        # Convert logits to one-hot for DiceMetric
                        v_pred_labels = torch.argmax(v_seg, dim=1)
                        v_y_pred = F.one_hot(v_pred_labels, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()
                        v_y_true = F.one_hot(v_mask_target.squeeze(1), num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()
                        self.metr_dice_val(v_y_pred, v_y_true)
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
            mae = float(self.metr_mae_val.aggregate().cpu())
            dice = float(self.metr_dice_val.aggregate().cpu()) if self.metr_dice_val.get_buffer() is not None else 0.0
            logger.info(f"Validation Stats: MAE={mae:.4f} Dice={dice:.4f}")
            return -mae
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
             with torch.no_grad():
                for batch in tqdm(self.ld_ts, desc="Testing (MAE)", mininterval=2.0):
                    imgs = batch.get("video", batch.get("image")).to(self.device)
                    targets = batch["target"].to(self.device)
                    cases = batch["case"]
                    
                    preds, _ = self.model(imgs)
                    
                    # Ensure (B,) shape for simple subtraction/logging
                    # Model output is (B, 1), targets might be (B,) or (B, 1)
                    if preds.ndim == 2:
                        preds = preds.view(-1)
                    if targets.ndim == 2:
                        targets = targets.view(-1)
                    
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
                for batch in tqdm(self.ld_ts, desc="Testing (Dice/HD)", mininterval=2.0):
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