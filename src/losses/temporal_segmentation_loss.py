import torch
import torch.nn as nn
from monai.losses import DiceCELoss

from src.registry import register_loss
from src.utils.logging import get_logger
from .cycle_consistency import CycleConsistencyLoss

logger = get_logger()


@register_loss("TemporalWeakSegLoss")
class TemporalWeakSegLoss(nn.Module):
    """
    Extended weakly-supervised segmentation loss with cycle consistency.
    
    Components:
        - Dice+CE: Strong supervision on labeled frames only (ED/ES)
        - EF Regression: Weak supervision backprops through entire volume curve
        - Volume Regression: Direct supervision on EDV/ESV (Conditional on frame presence)
        - Cycle Consistency: Self-supervision via warp agreement on unlabeled frames
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        ef_weight: float = 10.0,
        smooth_weight: float = 0.0,
        cycle_weight: float = 0.5,
        volume_weight: float = 0.1
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.ef_weight = ef_weight
        self.smooth_weight = smooth_weight
        self.cycle_weight = cycle_weight
        self.volume_weight = volume_weight

        self.dice_func = DiceCELoss(sigmoid=True, reduction='mean')
        self.l1 = nn.L1Loss()
        self.cycle_loss = CycleConsistencyLoss()

        logger.info(
            f"TemporalWeakSegLoss initialized: dice={dice_weight}, ef={ef_weight}, "
            f"cycle={cycle_weight}, volume={volume_weight}"
        )

    def forward(
        self,
        pred_logits: torch.Tensor,
        target_masks: torch.Tensor,
        pred_ef: torch.Tensor,
        target_ef: torch.Tensor,
        frame_mask: torch.Tensor,
        frames: torch.Tensor = None,
        target_edv: torch.Tensor = None,
        target_esv: torch.Tensor = None
    ):
        """
        Args:
            pred_logits: (B, 1, T, H, W) - Raw logits
            target_masks: (B, T, 1, H, W) - Ground truth masks
            pred_ef: (B,) or (B, 1) - Predicted EF
            target_ef: (B,) - Target EF
            frame_mask: (B, T) - 1.0 if ES, 2.0 if ED, 0.0 otherwise
            frames: (B, C, T, H, W) - Input video for cycle loss (optional)
            target_edv: (B,) or (B, 1) - Target End-Diastolic Volume
            target_esv: (B,) or (B, 1) - Target End-Systolic Volume

        Returns:
            total_loss: Scalar
            loss_dict: Dictionary of individual loss components
        """
        loss_dice = self._compute_dice_loss(pred_logits, target_masks, frame_mask)

        pred_ef_flat = pred_ef.view(-1) if pred_ef.dim() > 1 else pred_ef
        target_ef_flat = target_ef.view(-1) if target_ef.dim() > 1 else target_ef
        loss_ef = self.l1(pred_ef_flat, target_ef_flat)

        if frames is not None and self.cycle_weight > 0:
            probs = torch.sigmoid(pred_logits)
            probs_for_cycle = probs.permute(0, 2, 1, 3, 4)
            loss_cycle = self.cycle_loss(probs_for_cycle, frames, frame_mask)
        else:
            loss_cycle = torch.tensor(0.0, device=pred_logits.device)

        if self.volume_weight > 0 and target_edv is not None and target_esv is not None:
             loss_volume = self._compute_volume_loss(pred_logits, frame_mask, target_edv, target_esv)
        else:
             loss_volume = torch.tensor(0.0, device=pred_logits.device)

        loss_smooth = self._compute_smoothness_loss(torch.sigmoid(pred_logits))

        total_loss = (
            self.dice_weight * loss_dice +
            self.ef_weight * loss_ef +
            self.smooth_weight * loss_smooth +
            self.cycle_weight * loss_cycle +
            self.volume_weight * loss_volume
        )

        return total_loss, {
            "dice": loss_dice,
            "ef": loss_ef,
            "cycle": loss_cycle,
            "smooth": loss_smooth,
            "volume": loss_volume
        }

    def _compute_dice_loss(
        self,
        pred_logits: torch.Tensor,
        target_masks: torch.Tensor,
        frame_mask: torch.Tensor
    ) -> torch.Tensor:
        """Vectorized Dice+CE on labeled frames only."""
        if pred_logits.shape[1] == 1 and pred_logits.shape[2] > 1:
            pred_logits = pred_logits.permute(0, 2, 1, 3, 4)

        if target_masks.shape[1] == 1 and target_masks.shape[2] > 1:
            target_masks = target_masks.permute(0, 2, 1, 3, 4)

        B, T, C, H, W = pred_logits.shape

        pred_flat = pred_logits.reshape(-1, C, H, W)
        target_flat = target_masks.reshape(-1, C, H, W)
        mask_flat = frame_mask.reshape(-1)

        valid_indices = mask_flat > 0.5

        if valid_indices.sum() == 0:
            return torch.tensor(0.0, device=pred_logits.device)

        return self.dice_func(
            pred_flat[valid_indices],
            target_flat[valid_indices]
        )

    def _compute_volume_loss(
        self,
        pred_logits: torch.Tensor,
        frame_mask: torch.Tensor,
        target_edv: torch.Tensor,
        target_esv: torch.Tensor
    ) -> torch.Tensor:
        """Computes L1 loss between predicted and target volumes at ED/ES frames."""
        # pred_logits: (B, 1, T, H, W)
        pred_probs = torch.sigmoid(pred_logits)
        
        # Sum over spatial dimensions (H, W) to get volume per frame (in pixels)
        # Result: (B, 1, T) -> (B, T)
        pred_vol = pred_probs.sum(dim=(-2, -1)).squeeze(1)

        loss_vol = []
        
        B = pred_vol.shape[0]
        
        # Iterate over batch to find specific ED/ES indices
        for b in range(B):
            # ED Frame (marked as 2.0)
            ed_idx = (frame_mask[b] == 2.0).nonzero()
            # ES Frame (marked as 1.0)
            es_idx = (frame_mask[b] == 1.0).nonzero()
            
            if ed_idx.numel() > 0:
                idx = ed_idx[0].item()
                # target_edv shape (B,) or (B, 1)
                t_edv = target_edv[b].view(-1)
                p_edv = pred_vol[b, idx].view(-1)
                loss_vol.append(torch.abs(p_edv - t_edv))
                
            if es_idx.numel() > 0:
                idx = es_idx[0].item()
                t_esv = target_esv[b].view(-1)
                p_esv = pred_vol[b, idx].view(-1)
                loss_vol.append(torch.abs(p_esv - t_esv))
                
        if len(loss_vol) > 0:
            return torch.mean(torch.cat(loss_vol))
        else:
            return torch.tensor(0.0, device=pred_logits.device)

    def _compute_smoothness_loss(self, pred_logits):
        """Total Variation Loss to force spatial coherence"""
        # Penalize differences between adjacent pixels (noise)
        diff_h = torch.abs(pred_logits[..., 1:, :] - pred_logits[..., :-1, :])
        diff_w = torch.abs(pred_logits[..., :, 1:] - pred_logits[..., :, :-1])
        return diff_h.mean() + diff_w.mean()
