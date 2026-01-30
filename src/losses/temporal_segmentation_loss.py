import torch
import torch.nn as nn
from monai.losses import DiceCELoss

from src.registry import register_loss
from src.utils.logging import get_logger
from .geometric_smooth import GeometricSmoothLoss
from .cycle_consistency import CycleConsistencyLoss

logger = get_logger()


@register_loss("TemporalWeakSegLoss")
class TemporalWeakSegLoss(nn.Module):
    """
    Extended weakly-supervised segmentation loss with cycle consistency.
    
    Components:
        - Dice+CE: Strong supervision on labeled frames only (ED/ES)
        - EF Regression: Weak supervision backprops through entire volume curve
        - Geometric Smoothness: 2nd-order temporal coherence (centroid + area)
        - Contrast: Encourages motion variance for healthy patients (EF > 40%)
        - Cycle Consistency: Self-supervision via warp agreement on unlabeled frames
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        ef_weight: float = 10.0,
        smooth_weight: float = 1.0,
        contrast_weight: float = 0.1,
        cycle_weight: float = 0.5
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.ef_weight = ef_weight
        self.smooth_weight = smooth_weight
        self.contrast_weight = contrast_weight
        self.cycle_weight = cycle_weight

        self.dice_func = DiceCELoss(sigmoid=True, reduction='mean')
        self.mse = nn.MSELoss()
        self.geo_loss = GeometricSmoothLoss()
        self.cycle_loss = CycleConsistencyLoss()

        logger.info(
            f"TemporalWeakSegLoss initialized: dice={dice_weight}, ef={ef_weight}, "
            f"smooth={smooth_weight}, contrast={contrast_weight}, cycle={cycle_weight}"
        )

    def forward(
        self,
        pred_logits: torch.Tensor,
        target_masks: torch.Tensor,
        pred_ef: torch.Tensor,
        target_ef: torch.Tensor,
        frame_mask: torch.Tensor,
        frames: torch.Tensor = None
    ):
        """
        Args:
            pred_logits: (B, 1, T, H, W) - Raw logits
            target_masks: (B, T, 1, H, W) - Ground truth masks
            pred_ef: (B,) or (B, 1) - Predicted EF
            target_ef: (B,) - Target EF
            frame_mask: (B, T) - 1.0 if labeled (ED/ES), 0.0 otherwise
            frames: (B, C, T, H, W) - Input video for cycle loss (optional)

        Returns:
            total_loss: Scalar
            loss_dict: Dictionary of individual loss components
        """
        loss_dice = self._compute_dice_loss(pred_logits, target_masks, frame_mask)

        pred_ef_flat = pred_ef.view(-1) if pred_ef.dim() > 1 else pred_ef
        target_ef_flat = target_ef.view(-1) if target_ef.dim() > 1 else target_ef
        loss_ef = self.mse(pred_ef_flat, target_ef_flat)

        probs = torch.sigmoid(pred_logits)
        loss_smooth = self.geo_loss(probs)
        loss_contrast = self._compute_contrast_loss(probs, target_ef)

        if frames is not None and self.cycle_weight > 0:
            probs_for_cycle = probs.permute(0, 2, 1, 3, 4)
            loss_cycle = self.cycle_loss(probs_for_cycle, frames, frame_mask)
        else:
            loss_cycle = torch.tensor(0.0, device=pred_logits.device)

        total_loss = (
            self.dice_weight * loss_dice +
            self.ef_weight * loss_ef +
            self.smooth_weight * loss_smooth +
            self.contrast_weight * loss_contrast +
            self.cycle_weight * loss_cycle
        )

        return total_loss, {
            "dice": loss_dice,
            "ef": loss_ef,
            "smooth": loss_smooth,
            "contrast": loss_contrast,
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

    def _compute_contrast_loss(
        self,
        probs: torch.Tensor,
        target_ef: torch.Tensor
    ) -> torch.Tensor:
        """Encourages temporal variance for healthy hearts."""
        if self.contrast_weight <= 0:
            return torch.tensor(0.0, device=probs.device)

        if probs.shape[1] == 1 and probs.shape[2] > 1:
            probs = probs.permute(0, 2, 1, 3, 4)

        T = probs.shape[1]
        if T <= 1:
            return torch.tensor(0.0, device=probs.device)

        var_per_sample = probs.var(dim=1, correction=0).mean(dim=(1, 2, 3))
        should_move = (target_ef > 0.40).float()
        loss_per_sample = -1.0 * var_per_sample * should_move

        return loss_per_sample.mean()
