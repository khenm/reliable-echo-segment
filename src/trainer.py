import os
import time
import numpy as np
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric, RMSEMetric
from monai.transforms import AsDiscrete
from tqdm import tqdm
from src.utils.logging import get_logger
from src.utils.util_ import save_checkpoint, load_full_checkpoint
from src.models.temporal import TemporalGate
from src.utils.plot import plot_volume_debug
import wandb

logger = get_logger()

class Trainer:
    """
    Handles training, validation, and testing for all model types.
    Refactored for Clean Code principles: SRP, Small Functions usually < 20 lines.
    """
    def __init__(self, model, loaders, cfg, device, criterions=None, metrics=None):
        self.model = model
        self.ld_tr, self.ld_va, self.ld_ts = loaders
        self.cfg = cfg
        self.device = device
        self.criterions = criterions or {}
        self.metrics = metrics or {}
        
        self.num_classes = cfg['data']['num_classes']
        self._setup_paths()
        self._identify_model_type()
        self._setup_optimization()
        self._setup_metrics()

    def _setup_paths(self):
        self.run_dir = self.cfg['training']['run_dir']
        self.vault_dir = self.cfg['training']['vault_dir']
        self.run_id = self.cfg['training'].get('run_id', 'unknown')
        self.ckpt_path = os.path.join(self.run_dir, 'last.ckpt')
        self.vault_path = os.path.join(self.vault_dir, f"{self.run_id}_best.ckpt")

    def _identify_model_type(self):
        self.model_name = self.cfg['model'].get('name', 'VAEUNet')
        self.is_regression = (self.model_name.lower() == "r2plus1d")
        self.is_unet_2d = (self.model_name in ["unet_tcm"])
        self.is_dual_stream = (self.model_name == "dual_stream")
        self.is_skeletal = (self.model_name == "skeletal_tracker")
        self.is_segmentation = (self.model_name in ["segment_tracker", "temporal_segment_tracker"])

    def _setup_optimization(self):
        # Temporal Gate
        if self.cfg.get('losses', {}).get('temporal', {}).get('enable'):
            tc_cfg = self.cfg['losses']['temporal']
            self.temporal_gate = TemporalGate(
                threshold=tc_cfg.get('gate_threshold', 0.7),
                scale=tc_cfg.get('gate_scale', 10.0)
            ).to(self.device)
        else:
            self.temporal_gate = None

        # Gradient Accumulation
        micro_batch = self.cfg['training'].get('batch_size', 8)
        effective_train = self.cfg['training'].get('train_batch_size', micro_batch)
        self.accum_steps = max(1, effective_train // micro_batch)
        if self.accum_steps > 1:
            logger.info(f"Gradient Accumulation: {self.accum_steps} steps (effective batch size: {effective_train})")

        self.opt = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.cfg['training']['lr'], 
            weight_decay=self.cfg['training']['weight_decay']
        )
        
        dev_type = self.device.type if hasattr(self.device, 'type') else str(self.device)
        self.scaler = torch.amp.GradScaler(device=dev_type, enabled=(dev_type == 'cuda'))

    def _setup_metrics(self):
        if self.num_classes == 1:
            self.post_pred = AsDiscrete(threshold=0.5)
            if 'dice' in self.metrics:
                self.metrics['dice'] = DiceMetric(include_background=True, reduction="mean")
        else:
            self.post_pred = AsDiscrete(argmax=True, to_onehot=self.num_classes)

    def train(self):
        epochs = self.cfg['training']['epochs']
        patience = self.cfg['training']['patience']
        best_metric = -float('inf')
        wait = 0
        
        start_ep = 1
        if self.cfg['training'].get('resume_path'):
            start_ep, best_metric = self._load_checkpoint(self.cfg['training']['resume_path'])

        logger.info(f"Starting training from epoch {start_ep}")
        
        for ep in range(start_ep, epochs + 1):
            avg_loss, loss_comps = self._run_epoch(ep, epochs)
            val_result = self._validate()
            
            # Logging
            self._log_epoch(ep, avg_loss, loss_comps, val_result)
            
            # Checkpointing
            score = val_result if not isinstance(val_result, tuple) else val_result[0]
            is_best = False
            if score > best_metric:
                best_metric = score
                wait = 0
                is_best = True
            else:
                wait += 1
            
            self._save_checkpoint(ep, best_metric, is_best=is_best)
            
            if wait >= patience:
                logger.info("⏹ Early stop")
                break

    def _run_epoch(self, ep, max_ep):
        self.model.train()
        run_loss = 0.0
        loss_components = {key: 0.0 for key in self.criterions.keys()}
        
        pbar = tqdm(self.ld_tr, desc=f"Epoch {ep}/{max_ep}", mininterval=2.0)
        self.opt.zero_grad(set_to_none=True)
        
        for batch_idx, batch in enumerate(pbar):
            loss, batch_comps = self._process_batch(batch)
            
            # Scale loss for gradient accumulation
            scaled_loss = loss / self.accum_steps
            self.scaler.scale(scaled_loss).backward()
            
            # Step optimizer every accum_steps
            if (batch_idx + 1) % self.accum_steps == 0:
                self._debug_gradients(ep)
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad(set_to_none=True)
            
            run_loss += loss.item()
            for k, v in batch_comps.items():
                if k in loss_components:
                    loss_components[k] += v
            
            # Update Pbar
            postfix = {"loss": f"{loss.item():.4f}"}
            postfix.update({k: f"{v:.4f}" for k, v in batch_comps.items()})
            pbar.set_postfix(postfix)

            if wandb.run is not None:
                wandb.log({"train/loss": loss.item(), **{f"train/{k}": v for k, v in batch_comps.items()}})

        # Handle remaining gradients if total batches not divisible by accum_steps
        if len(self.ld_tr) % self.accum_steps != 0:
            self._debug_gradients(ep)
            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad(set_to_none=True)

        avg_loss = run_loss / len(self.ld_tr)
        avg_comps = {k: v / len(self.ld_tr) for k, v in loss_components.items()}
        return avg_loss, avg_comps

    def _process_batch(self, batch):
        dev_type = self.device.type if hasattr(self.device, 'type') else str(self.device)
        with torch.amp.autocast(device_type=dev_type):
            if self.is_segmentation:
                return self._train_segmentation(batch)
            elif self.is_skeletal:
                return self._train_skeletal(batch)
            elif self.is_dual_stream:
                return self._train_dual(batch)
            elif self.is_unet_2d:
                return self._train_unet_2d(batch)
            elif self.is_regression:
                return self._train_regression(batch)
            else:
                return self._train_seg_vae(batch)

    # --- Model Specific Training Steps ---

    def _train_segmentation(self, batch):
        imgs = batch["video"].to(self.device)
        lengths = batch.get("lengths")
        if lengths is not None:
            lengths = lengths.to(self.device)

        need_features = 'distillation' in self.criterions
        if need_features:
            outputs = self.model(imgs, lengths=lengths, return_features=True)
            mask_logits = outputs['mask_logits']
            pred_edv = outputs['pred_edv']
            pred_esv = outputs['pred_esv']
            pred_ef = outputs['pred_ef']
            features = outputs['features']
        else:
            outputs = self.model(imgs, lengths=lengths)
            mask_logits = outputs['mask_logits']
            pred_edv = outputs['pred_edv']
            pred_esv = outputs['pred_esv']
            pred_ef = outputs['pred_ef']

        loss = 0.0
        comps = {}

        target_masks = batch.get("label")
        frame_mask = batch.get("frame_mask")
        
        # Targets
        ef_target = batch.get("target_ef").to(self.device).view(-1, 1) if "target_ef" in batch else batch.get("target").to(self.device).view(-1, 1)
        edv_target = batch.get("target_edv").to(self.device).view(-1, 1)
        esv_target = batch.get("target_esv").to(self.device).view(-1, 1)

        # 1. Segmentation Loss
        if 'segmentation' in self.criterions and target_masks is not None:
            target_masks = target_masks.to(self.device)
            if frame_mask is not None:
                frame_mask = frame_mask.to(self.device)

            # Check if loss function accepts frames (TemporalWeakSegLoss)
            loss_fn = self.criterions['segmentation']
            # Pass pred_ef for supervision
            if hasattr(loss_fn, 'cycle_loss'):
                l_seg, c_dict = loss_fn(
                    mask_logits, target_masks, pred_ef, ef_target, frame_mask, imgs
                )
            else:
                l_seg, c_dict = loss_fn(
                    mask_logits, target_masks, pred_ef, ef_target, frame_mask
                )
            loss += l_seg
            comps['segmentation'] = l_seg.item()
            comps.update({k: v.item() for k, v in c_dict.items()})
            
        # 2. Volume Regression Loss
        if 'volume' in self.criterions:
             # Ignore dummy targets (-1)
             # Start with simple MSE, mask out dummies if necessary (though usually they are filtered in dataset or valid)
             # EchoNet dataset fills with -1.0 if missing.
             
             valid_vol = (edv_target >= 0) & (esv_target >= 0)
             if valid_vol.any():
                 l_vol = self.criterions['volume'](pred_edv[valid_vol], edv_target[valid_vol]) + \
                         self.criterions['volume'](pred_esv[valid_vol], esv_target[valid_vol])
                 loss += l_vol
                 comps['volume'] = l_vol.item()
                 
        # 3. EF Regression Loss (Direct on physical EF)
        if 'ef' in self.criterions:
             l_ef = self.criterions['ef'](pred_ef, ef_target)
             loss += l_ef
             comps['ef'] = l_ef.item()

        if 'distillation' in self.criterions:
            l_distill = self.criterions['distillation'](features, imgs)
            loss += l_distill
            comps['distillation'] = l_distill.item()

        return loss, comps

    def _train_skeletal(self, batch):
        imgs = batch["video"].to(self.device)
        lengths = batch.get("lengths")
        if lengths is not None: lengths = lengths.to(self.device)
        
        pred_kps, vol, ef = self.model(imgs, lengths=lengths) if lengths is not None else self.model(imgs)
        
        loss = 0.0
        comps = {}
        
        # 1. Skeletal Loss
        targets = batch.get("keypoints")
        frame_mask = batch.get("frame_mask")
        
        if 'skeletal' in self.criterions and targets is not None:
            targets = targets.to(self.device)
            if frame_mask is not None: frame_mask = frame_mask.to(self.device)
            
            l_skel, c_dict = self.criterions['skeletal'](pred_kps, targets, frame_mask)
            loss += l_skel
            comps['skeletal'] = l_skel.item()
            comps.update({k: v.item() for k, v in c_dict.items()})

        # 2. EF Loss (Auxiliary)
        if 'ef' in self.criterions:
            ef_target = batch.get("target").to(self.device).view(-1, 1)
            l_ef = self.criterions['ef'](ef, ef_target)
            loss += l_ef
            comps['ef'] = l_ef.item()
            
        return loss, comps

    def _train_unet_2d(self, batch):
        # Input (B, C, T, H, W) -> Flatten -> (B*T, C, H, W)
        imgs = batch.get("video", batch.get("image")).to(self.device)
        B, C, T, H, W = imgs.shape
        imgs_flat = imgs.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        
        logits_flat, ef_flat, features_flat = self.model(imgs_flat)
        
        # Reshape Back
        seg_logits = logits_flat.view(B, T, -1, H, W).permute(0, 2, 1, 3, 4) # (B, C, T, H, W)
        ef_preds = ef_flat.view(B, T).mean(dim=1).unsqueeze(1) # (B, 1)
        
        loss = 0.0
        comps = {}
        
        # 1. Segmentation
        if 'seg' in self.criterions:
            l_seg = self._calc_seg_loss(seg_logits, batch)
            loss += l_seg
            comps['seg'] = l_seg.item()
            
        # 2. EF Regression
        if 'ef_reg' in self.criterions:
            targets = batch["target"].to(self.device).view(-1, 1)
            l_ef = self.criterions['ef_reg'](ef_preds, targets)
            loss += l_ef
            comps['ef_reg'] = l_ef.item()
            
        # 3. Consistency
        if 'consistency' in self.criterions:
            l_cons = self.criterions['consistency'](seg_logits, ef_preds)
            loss += l_cons
            comps['consistency'] = l_cons.item()
            
        return loss, comps

    def _train_dual(self, batch):
        imgs = batch.get("video", batch.get("image")).to(self.device)
        ef_seq, seg_logits, ef_simpson = self.model(imgs)
        
        loss = 0.0
        comps = {}
        
        if 'seg' in self.criterions:
            l_seg = self._calc_seg_loss(seg_logits, batch)
            loss += l_seg
            comps['seg'] = l_seg.item()
            
        if 'ef' in self.criterions:
            targets = batch["target"].to(self.device).view(-1, 1)
            l_ef = self.criterions['ef'](ef_seq, targets)
            loss += l_ef
            comps['ef'] = l_ef.item()
            
        return loss, comps

    def _train_regression(self, batch):
        imgs = batch.get("video", batch.get("image")).to(self.device)
        preds, seg_logits = self.model(imgs)
        targets = batch["target"].to(self.device).view(-1, 1)
        
        loss = 0.0
        comps = {}
        
        if 'reg' in self.criterions:
            l_reg = self.criterions['reg'](preds, targets)
            loss += l_reg
            comps['reg'] = l_reg.item()
            
        if 'ef' in self.criterions:
            l_ef, _ = self.criterions['ef'](seg_logits, targets)
            loss += l_ef
            comps['ef'] = l_ef.item()
            
        return loss, comps

    def _train_seg_vae(self, batch):
        imgs = batch["image"].to(self.device)
        labs = batch["label"].to(self.device)
        logits, mu, log_var = self.model(imgs)
        
        loss = 0.0
        comps = {}
        
        if 'dice' in self.criterions:
            l_dice = self.criterions['dice'](logits, labs)
            loss += l_dice
            comps['dice'] = l_dice.item()
            
        if 'kl' in self.criterions:
            l_kl = self.criterions['kl'](mu, log_var)
            loss += l_kl
            comps['kl'] = l_kl.item()
            
        return loss, comps

    def _calc_seg_loss(self, seg_logits, batch):
        """Helper for masked 3D/2D segmentation loss"""
        mask_targets = batch.get("label").to(self.device)
        frame_mask = batch.get("frame_mask")
        
        if frame_mask is not None:
            B, C, T, H, W = seg_logits.shape
            frame_mask = frame_mask.to(self.device).view(-1)
            valid_idx = torch.nonzero(frame_mask).squeeze()
            
            if valid_idx.numel() == 0:
                return torch.tensor(0.0, device=self.device, requires_grad=True)
                
            sl_flat = seg_logits.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
            mt_flat = mask_targets.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
            
            return self.criterions['seg'](sl_flat[valid_idx], mt_flat[valid_idx])
        else:
            return self.criterions['seg'](seg_logits, mask_targets)

    # --- Validation ---

    def _validate(self):
        self.model.eval()
        for m in self.metrics.values(): m.reset()
        
        dev_type = self.device.type if hasattr(self.device, 'type') else str(self.device)
        with torch.no_grad(), torch.amp.autocast(device_type=dev_type):
            for batch in tqdm(self.ld_va, desc="Validating", mininterval=2.0, leave=False):
                if self.is_segmentation:
                    self._val_segmentation(batch)
                elif self.is_skeletal:
                    self._val_skeletal(batch)
                elif self.is_unet_2d:
                    self._val_unet_2d(batch)
                elif self.is_regression or self.is_dual_stream:
                    self._val_reg_dual(batch)
                else:
                    self._val_seg(batch)
                    
        return self._aggregate_metrics()

    def _val_segmentation(self, batch):
        imgs = batch["video"].to(self.device)
        lengths = batch.get("lengths")
        if lengths is not None:
            lengths = lengths.to(self.device)

        outputs = self.model(imgs, lengths=lengths)
        mask_logits = outputs['mask_logits']
        pred_ef = outputs['pred_ef']
        pred_edv = outputs['pred_edv']
        pred_esv = outputs['pred_esv']

        # Targets
        ef_target = batch.get("target_ef").to(self.device).view(-1, 1) if "target_ef" in batch else batch.get("target").to(self.device).view(-1, 1)
        edv_target = batch.get("target_edv").to(self.device).view(-1, 1)
        esv_target = batch.get("target_edv").to(self.device).view(-1, 1)

        # 1. EF Metrics
        if 'mae' in self.metrics:
            self.metrics['mae'](pred_ef, ef_target)
            if 'rmse' in self.metrics: self.metrics['rmse'](pred_ef, ef_target)
            if 'r2' in self.metrics: self.metrics['r2'](pred_ef, ef_target)

        # 2. Volume Metrics (EDV/ESV)
        # Filter out invalid targets (-1.0)
        if 'mae_edv' in self.metrics:
            # EDV
            valid_edv = (edv_target >= 0)
            if valid_edv.any():
                self.metrics['mae_edv'](pred_edv[valid_edv], edv_target[valid_edv])
                self.metrics['rmse_edv'](pred_edv[valid_edv], edv_target[valid_edv])
                self.metrics['r2_edv'](pred_edv[valid_edv], edv_target[valid_edv])

            # ESV 
            valid_esv = (esv_target >= 0)
            if valid_esv.any():
                self.metrics['mae_esv'](pred_esv[valid_esv], esv_target[valid_esv])
                self.metrics['rmse_esv'](pred_esv[valid_esv], esv_target[valid_esv])
                self.metrics['r2_esv'](pred_esv[valid_esv], esv_target[valid_esv])

        if 'dice' in self.metrics:
            target_masks = batch.get("label")
            frame_mask = batch.get("frame_mask")
            if target_masks is not None:
                target_masks = target_masks.to(self.device)
                B, C, T, H, W = mask_logits.shape

                pred_probs = torch.sigmoid(mask_logits)
                if pred_probs.shape[-2:] != target_masks.shape[-2:]:
                    pred_probs = F.interpolate(
                        pred_probs.view(B, C * T, H, W),
                        size=target_masks.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    ).view(B, C, T, *target_masks.shape[-2:])

                pred_binary = (pred_probs > 0.5).int()
                target_binary = target_masks.int()

                if frame_mask is not None:
                    frame_mask = frame_mask.to(self.device)
                    mask_flat = frame_mask.view(B * T)
                    valid_idx = torch.nonzero(mask_flat).squeeze(-1)

                    if valid_idx.numel() > 0:
                        pred_flat = pred_binary.permute(0, 2, 1, 3, 4).reshape(B * T, C, *pred_binary.shape[-2:])
                        target_flat = target_binary.permute(0, 2, 1, 3, 4).reshape(B * T, C, *target_binary.shape[-2:])
                        self.metrics['dice'](pred_flat[valid_idx], target_flat[valid_idx])
                else:
                    pred_flat = pred_binary.permute(0, 2, 1, 3, 4).reshape(B * T, C, *pred_binary.shape[-2:])
                    target_flat = target_binary.permute(0, 2, 1, 3, 4).reshape(B * T, C, *target_binary.shape[-2:])
                    self.metrics['dice'](pred_flat, target_flat)

    def _val_skeletal(self, batch):
        imgs = batch["video"].to(self.device)
        lengths = batch.get("lengths")
        if lengths is not None:
            lengths = lengths.to(self.device)

        preds, vol, ef = self.model(imgs, lengths=lengths) if lengths is not None else self.model(imgs)

        if 'mae' in self.metrics:
            targets = batch["target"].to(self.device).view(-1, 1)
            self.metrics['mae'](ef, targets)
            if 'rmse' in self.metrics: self.metrics['rmse'](ef, targets)
            if 'r2' in self.metrics: self.metrics['r2'](ef, targets)

        if 'skeletal' in self.metrics:
            kps_gt = batch.get("keypoints")
            mask = batch.get("frame_mask")
            if kps_gt is not None and mask is not None:
                kps_gt = kps_gt.to(self.device)
                mask_bool = mask.to(self.device).bool()
                if mask_bool.any():
                    self.metrics['skeletal'](preds[mask_bool], kps_gt[mask_bool])

    def _val_unet_2d(self, batch):
        imgs = batch.get("video", batch.get("image")).to(self.device)
        B, C, T, H, W = imgs.shape
        imgs_flat = imgs.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        logits_flat, _, _ = self.model(imgs_flat)
        
        # Binary Classification Logic
        v_pred = (torch.sigmoid(logits_flat) > 0.5).int()
        v_gt = batch["label"].to(self.device).permute(0, 2, 1, 3, 4).reshape(-1, 1, H, W).int()
        
        if 'dice' in self.metrics:
             self.metrics['dice'](v_pred, v_gt)

    def _val_reg_dual(self, batch):
        imgs = batch.get("video", batch.get("image")).to(self.device)
        if self.is_dual_stream:
            preds, _, _ = self.model(imgs)
        else:
            preds, _ = self.model(imgs)
            
        if 'mae' in self.metrics:
            targets = batch["target"].to(self.device).view(-1, 1)
            self.metrics['mae'](preds, targets)
            if 'rmse' in self.metrics: self.metrics['rmse'](preds, targets)
            if 'r2' in self.metrics: self.metrics['r2'](preds, targets)

    def _val_seg(self, batch):
        imgs = batch["image"].to(self.device)
        labs = batch["label"].to(self.device)
        logits, _, _ = self.model(imgs)
        
        pred = (torch.sigmoid(logits) > 0.5)
        if 'dice' in self.metrics:
            self.metrics['dice'](pred, labs)

    def _aggregate_metrics(self):
        mae = float(self.metrics['mae'].aggregate()) if 'mae' in self.metrics else 0.0
        rmse = float(self.metrics['rmse'].aggregate()) if 'rmse' in self.metrics else 0.0
        r2 = float(self.metrics['r2'].aggregate()) if 'r2' in self.metrics else 0.0

        # EDV
        mae_edv = float(self.metrics['mae_edv'].aggregate()) if 'mae_edv' in self.metrics else 0.0
        rmse_edv = float(self.metrics['rmse_edv'].aggregate()) if 'rmse_edv' in self.metrics else 0.0
        r2_edv = float(self.metrics['r2_edv'].aggregate()) if 'r2_edv' in self.metrics else 0.0

        # ESV
        mae_esv = float(self.metrics['mae_esv'].aggregate()) if 'mae_esv' in self.metrics else 0.0
        rmse_esv = float(self.metrics['rmse_esv'].aggregate()) if 'rmse_esv' in self.metrics else 0.0
        r2_esv = float(self.metrics['r2_esv'].aggregate()) if 'r2_esv' in self.metrics else 0.0

        dice = 0.0
        if 'dice' in self.metrics:
            buffer = self.metrics['dice'].get_buffer()
            if buffer is not None and len(buffer) > 0:
                dice = float(self.metrics['dice'].aggregate().cpu())

        if self.is_segmentation:
            return (dice, mae, rmse, r2, mae_edv, rmse_edv, r2_edv, mae_esv, rmse_esv, r2_esv)
        elif self.is_skeletal and 'skeletal' in self.metrics:
            skel = float(self.metrics['skeletal'].aggregate())
            return (-mae, mae, dice, skel, rmse, r2)
        elif self.is_regression or self.is_dual_stream:
            return (-mae, mae, dice, rmse, r2)
        else:
            return dice

    # --- Utils ---
    
    def _save_checkpoint(self, ep, metric, is_best=False):
        rng_state = {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate()
        }
        state = {
            'epoch': ep,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'rng_state': rng_state,
            'best_metric': metric
        }
        if is_best:
            save_checkpoint(self.vault_path, state)
            logger.info(f"Saved Best Checkpoint: {self.vault_path}")
        save_checkpoint(self.ckpt_path, state)

    def _load_checkpoint(self, path):
        return load_full_checkpoint(path, self.model, optimizer=self.opt, scaler=self.scaler, device=self.device, load_rng=True)

    def _log_epoch(self, ep, loss, comps, val_res):
        msg = f"E{ep:03d} loss={loss:.4f} " + " ".join([f"{k}={v:.4f}" for k, v in comps.items()])
        if isinstance(val_res, tuple):
            if self.is_segmentation:
                # (dice, mae, rmse, r2, mae_edv, rmse_edv, r2_edv, mae_esv, rmse_esv, r2_esv)
                msg += f" valDice={val_res[0]:.4f} valMAE={val_res[1]:.4f} valRMSE={val_res[2]:.4f} valR2={val_res[3]:.4f}"
                msg += f" valMAE_EDV={val_res[4]:.4f} valRMSE_EDV={val_res[5]:.4f} valR2_EDV={val_res[6]:.4f}"
                msg += f" valMAE_ESV={val_res[7]:.4f} valRMSE_ESV={val_res[8]:.4f} valR2_ESV={val_res[9]:.4f}"
            elif self.is_skeletal:
                 msg += f" valMAE={val_res[1]:.4f} valDice={val_res[2]:.4f} valSkel={val_res[3]:.4f} valRMSE={val_res[4]:.4f} valR2={val_res[5]:.4f}"
            elif self.is_regression or self.is_dual_stream:
                 msg += f" valMAE={val_res[1]:.4f} valDice={val_res[2]:.4f} valRMSE={val_res[3]:.4f} valR2={val_res[4]:.4f}"
            else:
                msg += f" valMAE={val_res[1]:.4f} valDice={val_res[2]:.4f}"
        else:
            msg += f" valDice={val_res:.4f}"
        logger.info(msg)
        
        if wandb.run is not None:
            log_dict = {"val/loss": loss}
            if isinstance(val_res, tuple):
                if self.is_segmentation:
                    log_dict.update({
                        "val/dice": val_res[0],
                        "val/mae": val_res[1],
                        "val/rmse": val_res[2],
                        "val/r2": val_res[3],
                        "val/mae_edv": val_res[4],
                        "val/rmse_edv": val_res[5],
                        "val/r2_edv": val_res[6],
                        "val/mae_esv": val_res[7],
                        "val/rmse_esv": val_res[8],
                        "val/r2_esv": val_res[9]
                    })
                elif self.is_skeletal:
                    log_dict["val/mae"] = val_res[1]
                    log_dict["val/dice"] = val_res[2]
                    log_dict["val/skeletal"] = val_res[3]
                    log_dict["val/rmse"] = val_res[4]
                    log_dict["val/r2"] = val_res[5]
                elif self.is_regression or self.is_dual_stream:
                    log_dict["val/mae"] = val_res[1]
                    log_dict["val/dice"] = val_res[2]
                    log_dict["val/rmse"] = val_res[3]
                    log_dict["val/r2"] = val_res[4]
                else:
                    log_dict["val/mae"] = val_res[1]
                    log_dict["val/dice"] = val_res[2]
            else:
                log_dict["val/dice"] = val_res
            wandb.log(log_dict)

    def _debug_gradients(self, ep):
        # Optional gradient tracing
        if self.cfg.get('losses', {}).get('debug', {}).get('trace_gradients'):
            pass # Simplified out for now.

    def evaluate_test(self):
        self.model.eval()
        records = []
        with torch.no_grad():
            for batch in tqdm(self.ld_ts, desc="Testing"):
                if self.is_skeletal:
                    self._test_skeletal(batch, records)
        
        if records:
            pd.DataFrame(records).to_csv(self.cfg['training']['test_metrics_csv'])

    def _test_skeletal(self, batch, records):
        imgs = batch["video"].to(self.device)
        targets = batch["target"].to(self.device).view(-1, 1)
        _, _, ef_preds = self.model(imgs)
        mae = torch.abs(ef_preds - targets).cpu().numpy()
        for i, val in enumerate(mae):
            records.append({"case": batch["case"][i], "MAE": val})

    def get_examples(self, num_examples=3):
        # Placeholder for visualization logic
        return []
