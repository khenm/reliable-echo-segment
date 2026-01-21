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
from src.utils.util_ import load_checkpoint_dict, save_checkpoint

logger = get_logger()

class Trainer:
    """
    Handles training, validation, and testing for both segmentation (VAE-U-Net) and 
    dual-task regression/segmentation (R2+1D) models.
    """
    def __init__(self, model, loaders, cfg, device, criterions=None, metrics=None):
        """
        Args:
            model (torch.nn.Module): The model to train.
            loaders (tuple): Tuple containing (train_loader, val_loader, test_loader).
            cfg (dict): Configuration dictionary defining hyperparameters and paths.
            device (torch.device): Device to execute computation on.
            criterions (dict): Dictionary of loss functions. e.g. {'dice': DiceLoss(), 'kl': KLLoss()}
            metrics (list): List of metrics to evaluate.
        """
        self.model = model
        self.ld_tr, self.ld_va, self.ld_ts = loaders
        self.cfg = cfg
        self.device = device
        self.criterions = criterions if criterions is not None else {}
        self.metrics = metrics if metrics is not None else {}
        self.num_classes = cfg['data']['num_classes']
        # Checkpoint paths derived from config
        checkpoint_dir = cfg['training']['checkpoint_dir']
        self.ckpt_path = os.path.join(checkpoint_dir, 'last.ckpt') # Workspace: .../last.ckpt
        self.vault_dir = checkpoint_dir  # Vault for best checkpoints
        
        self.model_name = cfg['model'].get('name', 'VAEUNet')
        self.is_regression = (self.model_name.lower() == "r2plus1d")
            
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
        time.sleep(0.1) # small delay before load
        checkpoint = load_checkpoint_dict(path, self.device)
        
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
                    if rng_state["cuda"] is not None and torch.cuda.is_available():
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
                    frame_mask = batch.get("frame_mask") # (B, T)
                    
                    if mask_targets is not None:
                        mask_targets = mask_targets.to(self.device)
                    
                    if frame_mask is not None:
                        frame_mask = frame_mask.to(self.device)
                    
                    with torch.amp.autocast(device_type=dev_type):
                        preds, seg_logits = self.model(imgs)
                        
                        # Calculate losses based on keys
                        if 'ef' in self.criterions:
                            # DifferentiableEFLoss expects seg logits and returns (loss, pred_ef)
                            l_ef, _ = self.criterions['ef'](seg_logits, targets)
                            loss += l_ef
                            loss_dict['ef'] = l_ef.item()
                        
                        if 'seg' in self.criterions and mask_targets is not None:
                             if frame_mask is not None:
                                 # Temporal Indexing: only calc loss on labeled frames
                                 # seg_logits: (B, 1, T, H, W) -> need to transpose to (B, T, 1, H, W) to flatten T
                                 B, C, T, H, W = seg_logits.shape
                                 
                                 # Reshape to (B*T, ...)
                                 sl_flat = seg_logits.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
                                 mt_flat = mask_targets.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
                                 fm_flat = frame_mask.view(-1) # (B*T)
                                 
                                 # Filter
                                 valid_indices = torch.nonzero(fm_flat).squeeze()
                                 
                                 if valid_indices.numel() > 0:
                                     sl_valid = sl_flat[valid_indices]
                                     mt_valid = mt_flat[valid_indices]
                                     l_seg = self.criterions['seg'](sl_valid, mt_valid.long())
                                 else:
                                     l_seg = torch.tensor(0.0, device=self.device, requires_grad=True)
                                     
                             else:
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

            val_result = self._validate()
            
            # Prepare log message
            avg_run_loss = run_loss / len(self.ld_tr)
            log_msg = f"E{ep:03d} trainLoss={avg_run_loss:.4f} "
            
            for k, v in epoch_loss_components.items():
                avg_comp = v / len(self.ld_tr)
                log_msg += f"{k}={avg_comp:.4f} "
            
            if self.is_regression:
                val_score, val_mae, val_dice = val_result
                log_msg += f"valMAE={val_mae:.4f} valDice={val_dice:.4f}"
            else:
                val_score = val_result
                log_msg += f"valDice={val_score:.4f}"
            
            logger.info(log_msg)

            if val_score > best_metric:
                best_metric = val_score
                wait = 0
                
                # Save BEST to Vault
                if self.vault_dir:
                    best_ckpt_name = f"{self.cfg['training']['save_dir'].split('/')[-1]}_best.ckpt"
                    best_ckpt_path = os.path.join(self.vault_dir, best_ckpt_name)
                    logger.info(f"Saving BEST checkpoint to Vault: {best_ckpt_path}...")
                    save_checkpoint(best_ckpt_path, {
                        'epoch': ep,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.opt.state_dict(),
                        'best_metric': best_metric
                    })
            else:
                wait += 1
            
            # Save Latest Checkpoint to Workspace (THE STATE DUMP)
            # This is overwriting 'last.ckpt' at every epoch
            save_checkpoint(self.ckpt_path, {
                'epoch': ep,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.opt.state_dict(),
                'best_metric': best_metric,
                'rng_state': {
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                    "numpy": np.random.get_state(),
                    "python": random.getstate()
                }
            })
            
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
        
        # Reset all metrics
        for metric in self.metrics.values():
            metric.reset()
        
        dev_type = self.device.type if hasattr(self.device, 'type') else str(self.device)
        with torch.no_grad(), torch.amp.autocast(device_type=dev_type):
            for vb in tqdm(self.ld_va, desc="Validating", mininterval=2.0, leave=False):
                if self.is_regression:
                    v_img = vb.get("video", vb.get("image")).to(self.device)
                    v_target = vb["target"].to(self.device)
                    
                    v_preds, v_seg = self.model(v_img)
                    
                    # MAE Metric
                    if 'mae' in self.metrics:
                        if v_target.ndim == 1:
                            v_target = v_target.view(-1, 1)
                        self.metrics['mae'](v_preds, v_target)
                    
                    # Dice Metric
                    if 'dice' in self.metrics:
                        v_mask_target = vb.get("label")
                        frame_mask = vb.get("frame_mask")
                        
                        if v_mask_target is not None:
                            v_mask_target = v_mask_target.to(self.device).long()
                            v_pred_labels = (torch.sigmoid(v_seg) > 0.5).float() 
                            
                            if frame_mask is not None:
                                frame_mask = frame_mask.to(self.device)
                                B, C, T, H, W = v_pred_labels.shape
                                
                                # Reshape and filter
                                vl_flat = v_pred_labels.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
                                vt_flat = v_mask_target.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
                                fm_flat = frame_mask.view(-1)
                                
                                valid_indices = torch.nonzero(fm_flat).squeeze()
                                if valid_indices.numel() > 0:
                                    self.metrics['dice'](vl_flat[valid_indices], vt_flat[valid_indices])
                            else:
                                self.metrics['dice'](v_pred_labels, v_mask_target)
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

                    if 'dice' in self.metrics:
                        self.metrics['dice'](v_y_pred, v_y_true)
        
        if self.is_regression:
            mae = float(self.metrics['mae'].aggregate().cpu()) if 'mae' in self.metrics else 0.0
            dice = float(self.metrics['dice'].aggregate().cpu()) if 'dice' in self.metrics and self.metrics['dice'].get_buffer() is not None else 0.0
            # Return tuple: (combined_score, mae, dice)
            # Use negative MAE as score since lower MAE is better
            return (-mae, mae, dice)
        else:
            return float(self.metrics['dice'].aggregate().cpu()) if 'dice' in self.metrics else 0.0

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