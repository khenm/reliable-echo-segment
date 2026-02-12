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
        volume_weight: float = 0.0,
        cycle_weight: float = 0.5
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.ef_weight = ef_weight
        self.volume_weight = volume_weight
        self.cycle_weight = cycle_weight

        self.dice_func = DiceCELoss(sigmoid=True, reduction='mean')
        self.l1 = nn.L1Loss()
        self.cycle_loss = CycleConsistencyLoss()

        logger.info(
            f"TemporalWeakSegLoss initialized: dice={dice_weight}, ef={ef_weight}, "
            f"volume={volume_weight}, cycle={cycle_weight}"
        )

    def forward(
        self,
        pred_logits: torch.Tensor,
        target_masks: torch.Tensor,
        pred_ef: torch.Tensor,
        target_ef: torch.Tensor,
        frame_mask: torch.Tensor,
        frames: torch.Tensor = None,
        pred_edv: torch.Tensor = None,
        target_edv: torch.Tensor = None,
        pred_esv: torch.Tensor = None,
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
            pred_edv, target_edv, pred_esv, target_esv: Volume regression (optional)

        Returns:
            total_loss: Scalar
            loss_dict: Dictionary of individual loss components
        """
        loss_dice = self._compute_dice_loss(pred_logits, target_masks, frame_mask)

        pred_ef_flat = pred_ef.view(-1) if pred_ef.dim() > 1 else pred_ef
        target_ef_flat = target_ef.view(-1) if target_ef.dim() > 1 else target_ef
        loss_ef = self.l1(pred_ef_flat, target_ef_flat)

        loss_volume = torch.tensor(0.0, device=pred_logits.device)
        
        if self.volume_weight > 0 and pred_edv is not None and target_edv is not None:
             # Check distinct presence of frames based on mask labels
             # 1.0 = ES Frame, 2.0 = ED Frame
             has_es = (frame_mask == 1.0).any(dim=1)
             has_ed = (frame_mask == 2.0).any(dim=1)
             
             target_edv_flat = target_edv.view(-1)
             target_esv_flat = target_esv.view(-1)
             
             # Calculate EDV loss only where ED frame exists
             l_edv = torch.tensor(0.0, device=pred_logits.device)
             if has_ed.any():
                 valid_edv = (target_edv_flat > 0) & has_ed
                 if valid_edv.any():
                     l_edv = self.l1(pred_edv.view(-1)[valid_edv], target_edv_flat[valid_edv])

             # Calculate ESV loss only where ES frame exists
             l_esv = torch.tensor(0.0, device=pred_logits.device)
             if has_es.any():
                 valid_esv = (target_esv_flat > 0) & has_es
                 if valid_esv.any():
                     l_esv = self.l1(pred_esv.view(-1)[valid_esv], target_esv_flat[valid_esv])
                     
             loss_volume = l_edv + l_esv

        if frames is not None and self.cycle_weight > 0:
            probs = torch.sigmoid(pred_logits)
            probs_for_cycle = probs.permute(0, 2, 1, 3, 4)
            loss_cycle = self.cycle_loss(probs_for_cycle, frames, frame_mask)
        else:
            loss_cycle = torch.tensor(0.0, device=pred_logits.device)

        total_loss = (
            self.dice_weight * loss_dice +
            self.ef_weight * loss_ef +
            self.volume_weight * loss_volume +
            self.cycle_weight * loss_cycle
        )

        return total_loss, {
            "dice": loss_dice,
            "ef": loss_ef,
            "volume": loss_volume,
            "cycle": loss_cycle
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
