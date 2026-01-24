import os
import time
import numpy as np
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric, MAEMetric
from monai.transforms import AsDiscrete
from tqdm import tqdm
from src.utils.logging import get_logger
from src.utils.util_ import load_checkpoint_dict, save_checkpoint, load_full_checkpoint
from src.models.temporal import TemporalGate

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
        self.run_dir = cfg['training']['run_dir']
        self.vault_dir = cfg['training']['vault_dir']
        self.run_id = cfg['training'].get('run_id', 'unknown')
        
        self.ckpt_path = os.path.join(self.run_dir, 'last.ckpt') # Workspace: .../last.ckpt
        # Vault path template: checkpoints/{model_name}/{timestamp}_best.ckpt
        self.vault_path = os.path.join(self.vault_dir, f"{self.run_id}_best.ckpt")
        
        self.model_name = cfg['model'].get('name', 'VAEUNet')
        self.is_regression = (self.model_name.lower() == "r2plus1d")
        self.is_unet_2d = (self.model_name in ["unet_tcm"])
        self.is_dual_stream = (self.model_name == "dual_stream")
        
        # Initialize Temporal Gate if enabled
        if self.cfg.get('losses', {}).get('temporal', {}).get('enable'):
            tc_cfg = self.cfg['losses']['temporal']
            self.temporal_gate = TemporalGate(
                threshold=tc_cfg.get('gate_threshold', 0.7),
                scale=tc_cfg.get('gate_scale', 10.0)
            ).to(device)
        else:
            self.temporal_gate = None
            
        self.opt = torch.optim.AdamW(model.parameters(), 
                                     lr=cfg['training']['lr'], 
                                     weight_decay=cfg['training']['weight_decay'])
        
        dev_type = device.type if hasattr(device, 'type') else str(device)
        self.scaler = torch.amp.GradScaler(device=dev_type, enabled=(dev_type == 'cuda'))

        # Fix for Binary Segmentation
        if self.num_classes == 1:
            self.post_pred = AsDiscrete(threshold=0.5)
            # Ensure DiceMetric includes background for binary case (index 0 is the class)
            if 'dice' in self.metrics:
                self.metrics['dice'] = DiceMetric(include_background=True, reduction="mean")
        else:
            self.post_pred = AsDiscrete(argmax=True, to_onehot=self.num_classes)

    def _safe_one_hot_nchw(self, target, num_classes):
        """
        Robustly converts target to one-hot (N, C, H, W).
        Handles invalid indices (>= num_classes or < 0) by creating zero vectors.
        """
        # Create blank (N, C, H, W)
        shape = list(target.shape) # (N, H, W)
        shape.insert(1, num_classes)
        out = torch.zeros(shape, device=target.device, dtype=torch.float32)
        
        # Valid mask
        valid_mask = (target >= 0) & (target < num_classes)
        
        # We can use scatter_ to fill the ones
        # target needs to be (N, 1, H, W) for scatter
        # out needs to be (N, C, H, W)
        
        # Expand target dims
        target_unsqueezed = target.unsqueeze(1).long() # (N, 1, H, W)
        
        # Only scatter where valid
        # To do this safely with scatter, we need to ensure indices are valid or masked.
        # But scatter writes typically.
        # Alternative: use F.one_hot on clamped/masked data and then zero out invalid.
        
        safe_target = target.clone()
        safe_target[~valid_mask] = 0 # Temporarily point to 0
        
        one_hot = F.one_hot(safe_target.long(), num_classes=num_classes) # (N, H, W, C)
        one_hot = one_hot.permute(0, 3, 1, 2).float() # (N, C, H, W)
        
        # Zero out where it was invalid
        # valid_mask is (N, H, W), expand to (N, C, H, W)
        valid_mask_expanded = valid_mask.unsqueeze(1).expand_as(one_hot)
        one_hot = one_hot * valid_mask_expanded.float()
        
        return one_hot

    def _load_checkpoint(self, path, load_optimizer=False):
        """
        Loads a checkpoint using the unified utility.
        """
        from src.utils.util_ import load_full_checkpoint
        return load_full_checkpoint(path, self.model, 
                                    optimizer=self.opt if load_optimizer else None, 
                                    device=self.device, 
                                    load_rng=load_optimizer)

    def train(self):
        """
        Executes the training loop with early stopping.
        """
        epochs = self.cfg['training']['epochs']
        patience = self.cfg['training']['patience']
        best_metric = -float('inf')
        wait = 0
        stop_ep = epochs

        start_tr = time.time()
        
        start_epoch = 1
        if self.cfg['training'].get('resume_path'):
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

                if self.is_regression or self.is_unet_2d:
                    # Generic handling for Video Dataset (returns 'video', 'target', 'label')
                    imgs = batch.get("video", batch.get("image")).to(self.device)
                    targets = batch["target"].to(self.device)
                    mask_targets = batch.get("label")
                    frame_mask = batch.get("frame_mask") # (B, T)
                    
                    if mask_targets is not None:
                        mask_targets = mask_targets.to(self.device)
                    
                    if frame_mask is not None:
                        frame_mask = frame_mask.to(self.device)
                    
                    if mask_targets is not None:
                        mask_targets = mask_targets.to(self.device)
                    
                    if frame_mask is not None:
                        frame_mask = frame_mask.to(self.device)
                    
                    with torch.amp.autocast(device_type=dev_type):
                        if self.is_unet_2d:
                            # Reshape for 2D processing: (B, C, T, H, W) -> (B*T, C, H, W)
                            B, C, T, H, W = imgs.shape
                            imgs_flat = imgs.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
                            
                            # Forward pass (2D)
                            logits_flat, ef_flat, features_flat = self.model(imgs_flat)
                            
                            # Reshape back to Video structure
                            # logits: (B*T, num_classes, H, W) -> (B, T, num_classes, H, W) -> (B, num_classes, T, H, W)
                            num_classes = logits_flat.shape[1]
                            seg_logits = logits_flat.view(B, T, num_classes, H, W).permute(0, 2, 1, 3, 4)
                            
                            # features: (B*T, D) -> (B, T, D)
                            features = features_flat.view(B, T, -1)
                            
                            # EF: (B*T, 1) -> (B, T, 1) -> Mean over time for Video EF
                            ef_seq = ef_flat.view(B, T)
                            preds = ef_seq.mean(dim=1).unsqueeze(1) # (B, 1)

                            # --- Calculate Losses for UNet_2D ---
                            
                            # 1. Segmentation Loss (Standard)
                            if 'seg' in self.criterions and mask_targets is not None:
                                # Flatten for Dice Loss as usual
                                sl_flat = seg_logits.permute(0, 2, 1, 3, 4).reshape(-1, num_classes, H, W)
                                mt_flat = mask_targets.permute(0, 2, 1, 3, 4).reshape(-1, 1, H, W)
                                if frame_mask is not None:
                                    fm_flat = frame_mask.view(-1)
                                    valid_idx = torch.nonzero(fm_flat).reshape(-1)
                                    if valid_idx.numel() > 0:
                                        l_seg = self.criterions['seg'](sl_flat[valid_idx], mt_flat[valid_idx])
                                    else:
                                        l_seg = torch.tensor(0.0, device=self.device, requires_grad=True)
                                else:
                                    l_seg = self.criterions['seg'](seg_logits, mask_targets)
                                    
                                loss += l_seg
                                loss_dict['seg'] = l_seg.item()

                            # 2. EF Regression Loss (Direct)
                            if 'ef_reg' in self.criterions:
                                l_ef = self.criterions['ef_reg'](preds, targets.view(-1, 1))
                                loss += l_ef
                                loss_dict['ef'] = l_ef.item()

                            # 3. Consistency Loss (Refinement Loop)
                            if 'consistency' in self.criterions:
                                # Input: seg_logits (B, C, T, H, W) and preds (B, 1)
                                l_cons = self.criterions['consistency'](seg_logits, preds)
                                loss += l_cons
                                loss_dict['cons'] = l_cons.item()
                                
                            # 4. Temporal Consistency
                            if 'temporal' in self.criterions and self.temporal_gate is not None:
                                # Features: (B, T, D)
                                f_t = features[:, 1:]
                                f_prev = features[:, :-1]
                                
                                # Gate
                                gate, _ = self.temporal_gate(f_t.reshape(-1, f_t.shape[-1]), f_prev.reshape(-1, f_prev.shape[-1]))
                                gate = gate.view(B, T-1, 1)
                                
                                # Logits for Temporal Loss
                                # (B, C, T, H, W) -> need (B, T, C, H, W) for slicing or just slice dim 2
                                log_t = seg_logits[:, :, 1:]
                                log_prev = seg_logits[:, :, :-1]
                                
                                # My TemporalConsistencyLoss expects (B, C, H, W).
                                # Here inputs are (B, C, T-1, H, W).
                                # Flatten T dimension
                                log_t_flat = log_t.permute(0, 2, 1, 3, 4).reshape(-1, num_classes, H, W)
                                log_prev_flat = log_prev.permute(0, 2, 1, 3, 4).reshape(-1, num_classes, H, W)
                                gate_flat = gate.view(-1, 1)
                                
                                l_temp = self.criterions['temporal'](log_t_flat, log_prev_flat, gate_flat)
                                loss += l_temp
                                loss_dict['temp'] = l_temp.item()

                        elif self.is_dual_stream:
                            # Dual Stream Model
                            # Returns: ef_seq, seg_logits, ef_simpson
                            ef_seq, seg_logits, ef_simpson = self.model(imgs)
                            
                            # 1. Segmentation Loss
                            if 'seg' in self.criterions and mask_targets is not None:
                                B, C, T, H, W = seg_logits.shape
                                # Flatten T
                                sl_flat = seg_logits.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
                                mt_flat = mask_targets.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
                                
                                if frame_mask is not None:
                                    fm_flat = frame_mask.view(-1)
                                    valid_idx = torch.nonzero(fm_flat).squeeze()
                                    if valid_idx.numel() > 0:
                                        l_seg = self.criterions['seg'](sl_flat[valid_idx], mt_flat[valid_idx])
                                    else:
                                        l_seg = torch.tensor(0.0, device=self.device, requires_grad=True)
                                else:
                                    l_seg = self.criterions['seg'](sl_flat, mt_flat)
                                
                                loss += l_seg
                                loss_dict['seg'] = l_seg.item()
                                
                            # 2. Main EF Loss (Sequence Stream vs GT)
                            if 'ef' in self.criterions:
                                l_ef = self.criterions['ef'](ef_seq, targets.view(-1, 1))
                                loss += l_ef
                                loss_dict['ef'] = l_ef.item()

                            # 3. Simpson Consistency Loss (Seq vs Simpson)
                            if 'simpson' in self.criterions:
                                l_simpson = self.criterions['simpson'](ef_seq, ef_simpson)
                                loss += l_simpson
                                loss_dict['simpson'] = l_simpson.item()

                        else: 
                            # R2Plus1D
                            preds, seg_logits = self.model(imgs)
                        
                            # Calculate losses based on keys
                            if 'reg' in self.criterions:
                                l_reg = self.criterions['reg'](preds.view(-1, 1), targets.view(-1, 1))
                                loss += l_reg
                                loss_dict['reg'] = l_reg.item()
    
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
                                     valid_indices = torch.nonzero(fm_flat).reshape(-1)
                                     
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
    
                            if 'semi_sup' in self.criterions and mask_targets is not None:
                                # EchoSemiSupervisedLoss handles masking internally
                                if frame_mask is not None:
                                    l_semi, comp_dict = self.criterions['semi_sup'](
                                        seg_logits, 
                                        mask_targets, 
                                        labeled_mask=frame_mask
                                    )
                                else:
                                    l_semi, comp_dict = self.criterions['semi_sup'](seg_logits, mask_targets)
                                
                                loss += l_semi
                                loss_dict['semi_sup'] = l_semi.item()
                                for k, v in comp_dict.items():
                                    loss_dict[k] = v.item()

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
                    logger.info(f"Saving BEST checkpoint to Vault: {self.vault_path}...")
                    save_checkpoint(self.vault_path, {
                        'epoch': ep,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.opt.state_dict(),
                        'best_metric': best_metric
                    })
                    
                    # Save metrics.json
                    import json
                    metrics_path = os.path.join(self.run_dir, "metrics.json")
                    with open(metrics_path, 'w') as f:
                        json.dump({"best_metric": best_metric, "epoch": ep}, f, indent=4)
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
                if self.is_regression or self.is_dual_stream:
                    v_img = vb.get("video", vb.get("image")).to(self.device)
                    v_target = vb["target"].to(self.device)
                    
                    if self.is_dual_stream:
                        # (ef_seq, seg_logits, ef_simpson)
                        v_preds, v_seg, _ = self.model(v_img)
                    else:
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
                elif self.is_unet_2d:
                     # UNet_2D Validation (Video Input: B, C, T, H, W)
                     v_img = vb.get("video", vb.get("image")).to(self.device)
                     # Treat label as (B, 1, T, H, W)
                     v_lab = vb["label"].to(self.device)
                     frame_mask = vb.get("frame_mask")
                     
                     # Forward (Video) -> (B, num_classes, T, H, W) if reshaped manually or model does it
                     # Currently model expects (B, C, H, W).
                     # Reshape: (B, C, T, H, W) -> (B*T, C, H, W)
                     B, C, T, H, W = v_img.shape
                     v_img_flat = v_img.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
                     
                     v_logits_flat, _, _ = self.model(v_img_flat)
                     # v_logits_flat: (B*T, num_classes, H, W)
                     
                     if self.num_classes == 1:
                         # Binary Logic
                         v_pred_labels = (torch.sigmoid(v_logits_flat) > 0.5).int() # (B*T, 1, H, W)
                         v_gt_labels = v_lab.permute(0, 2, 1, 3, 4).reshape(-1, 1, H, W).int() # (B*T, 1, H, W)
                         
                         v_y_pred = v_pred_labels
                         v_y_true = v_gt_labels
                     else:
                         # Multi-class Logic
                         v_pred_labels = torch.argmax(v_logits_flat, dim=1) # (B*T, H, W)
                         
                         # Targets: (B, 1, T, H, W) -> (B*T, H, W)
                         v_gt_labels = v_lab.permute(0, 2, 1, 3, 4).reshape(-1, H, W).long()
                     
                         v_y_pred = F.one_hot(v_pred_labels, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
                         v_y_true = self._safe_one_hot_nchw(v_gt_labels, num_classes=self.num_classes)

                     if 'dice' in self.metrics:
                         # For DiceMetric: if include_background=True/False is set in init.
                         # If binary (num_class=1), inputs are (B, 1, H, W).
                         
                         if frame_mask is not None:
                             fm_flat = frame_mask.to(self.device).view(-1)
                             valid_idx = torch.nonzero(fm_flat).squeeze()
                             
                             if valid_idx.numel() > 0:
                                 # Filter
                                 v_y_pred_valid = v_y_pred[valid_idx]
                                 v_y_true_valid = v_y_true[valid_idx]
                                 
                                 self.metrics['dice'](v_y_pred_valid, v_y_true_valid)
                         else:
                             self.metrics['dice'](v_y_pred, v_y_true)

                else:
                    v_img = vb["image"].to(self.device)
                    v_lab = vb["label"].to(self.device)

                    v_logits, _, _ = self.model(v_img)
                    
                    if self.num_classes == 1:
                        # Binary Segmentation Fix
                        # v_logits: (B, 1, H, W)
                        # Apply Sigmoid -> Threshold
                        v_probs = torch.sigmoid(v_logits)
                        v_pred_list = [self.post_pred(i) for i in v_probs]
                        
                        # Targets: (B, 1, H, W)
                        if v_lab.ndim == 4 and v_lab.shape[1] == 1:
                            pass
                        elif v_lab.ndim == 3:
                            v_lab = v_lab.unsqueeze(1)
                        else:
                             # Try to adapt
                             if v_lab.shape[1] != 1:
                                  # Maybe (B, H, W)?
                                  pass
                             
                        # v_lab is expected to be 0s and 1s
                        v_true_list = [i for i in v_lab]
                        
                        if 'dice' in self.metrics:
                             self.metrics['dice'](y_pred=v_pred_list, y=v_true_list)
                    else:
                        # Multi-class
                        v_pred_labels = torch.argmax(v_logits, dim=1)
    
                        if v_lab.ndim == 4 and v_lab.shape[1] == 1:
                            v_gt_labels = v_lab[:, 0].long()
                        else:
                            raise RuntimeError(f"Unexpected val GT shape {v_lab.shape}")
    
                        v_y_pred = F.one_hot(v_pred_labels, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
                        # Use the safe one hot here too for consistency, or standard if guaranteed valid
                        v_y_true = self._safe_one_hot_nchw(v_gt_labels, num_classes=self.num_classes)
    
                        if 'dice' in self.metrics:
                            self.metrics['dice'](v_y_pred, v_y_true)
        
        if self.is_regression or self.is_dual_stream:
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
        self.model.eval()

        records = []
        
        if self.is_regression or self.is_dual_stream:
             with torch.no_grad():
                for batch in tqdm(self.ld_ts, desc="Testing (MAE)", mininterval=2.0):
                    imgs = batch.get("video", batch.get("image")).to(self.device)
                    targets = batch["target"].to(self.device)
                    cases = batch["case"]
                    
                    if self.is_dual_stream:
                        preds, _, _ = self.model(imgs)
                    else:
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
                    if self.num_classes == 1:
                        # Binary
                        v_pred_labels = (torch.sigmoid(logits) > 0.5).int() # (B, 1, H, W)
                        if gts.ndim == 4 and gts.shape[1] == 1:
                            v_gt_labels = gts.int()
                        else:
                            v_gt_labels = gts.unsqueeze(1).int() # Assume (B, H, W) -> (B, 1, H, W)
                            
                        y_pred_all = v_pred_labels.float()
                        y_true_all = v_gt_labels.float()
                    else:
                        # Multi-class
                        pred_labels = torch.argmax(logits, dim=1)

                        if gts.ndim == 4 and gts.shape[1] == 1:
                            gt_labels = gts[:, 0].long()
                        else:
                            gt_labels = gts.long()

                        y_pred_all = F.one_hot(pred_labels, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
                        y_true_all = self._safe_one_hot_nchw(gt_labels, num_classes=self.num_classes)

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

    def get_examples(self, num_examples=3):
        """
        Runs inference on a few test samples and returns data for visualization.
        
        Args:
            num_examples (int): Number of examples to retrieve.
            
        Returns:
            list: List of sample dictionaries containing:
                  'img', 'gt_mask', 'pred_mask', 'gt_ef', 'pred_ef', 'title'.
        """
        self.model.eval()
        samples = []
        
        try:
            # Get a batch from test loader
            # We use an iterator to just get one batch
            batch = next(iter(self.ld_ts))
            
            # Move to device
            if self.is_regression:
                 videos = batch['video'].to(self.device)
            else:
                 videos = batch['image'].to(self.device).unsqueeze(2) # (B, C, 1, H, W) for compatibility if needed
            
            targets = batch['target'].to(self.device)
            labels = batch.get('label')
            if labels is not None: labels = labels.to(self.device)
            cases = batch['case']
            
            with torch.no_grad():
                if self.is_dual_stream:
                    ef_pred, seg_pred, _ = self.model(videos)
                else:
                    ef_pred, seg_pred = self.model(videos)
                
            # Iterate
            limit = min(num_examples, len(videos))
            for i in range(limit):
                vid = videos[i].cpu().numpy() # (C, T, H, W)
                
                # EF Handling
                if targets.ndim == 2: t_ef = targets[i, 0].item()
                else: t_ef = targets[i].item()
                
                if ef_pred.ndim == 2: p_ef = ef_pred[i, 0].item()
                else: p_ef = ef_pred[i].item()
                
                gt_ef = t_ef * 100.0
                pr_ef = p_ef * 100.0
                
                fname = cases[i]
                
                # Select best frame
                # If R2+1D, vid is (C, T, H, W)
                T = vid.shape[1]
                t_idx = T // 2
                gt_mask = None
                
                if labels is not None:
                    # labels: (B, C, T, H, W) -> ith sample: (C, T, H, W) -> (T, H, W)
                    lbl = labels[i, 0].cpu().numpy() 
                    mask_sums = lbl.sum(axis=(1, 2))
                    if mask_sums.max() > 0:
                        t_idx = mask_sums.argmax()
                        gt_mask = lbl[t_idx]
                
                # Extract Image Frame (C, H, W)
                frame_img = vid[:, t_idx]
                
                # Extract Pred Mask
                # seg_pred: (B, 1, T, H, W)
                pr_mask_t = torch.sigmoid(seg_pred[i]).cpu().numpy() # (1, T, H, W)
                pr_mask = (pr_mask_t[0, t_idx] > 0.5).astype(float)
                
                sample = {
                    "img": frame_img,
                    "gt_mask": gt_mask,
                    "pred_mask": pr_mask,
                    "gt_ef": gt_ef,
                    "pred_ef": pr_ef,
                    "title": f"Case: {fname} | Frame: {t_idx} | EF: {gt_ef:.1f}% vs {pr_ef:.1f}%",
                    "fname": fname,
                    "frame_idx": t_idx
                }
                samples.append(sample)
                
        except Exception as e:
            logger.error(f"Failed to get visualization examples: {e}")
            
        return samples