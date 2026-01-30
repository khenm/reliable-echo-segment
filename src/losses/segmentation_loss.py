import torch
import torch.nn as nn
from monai.losses import DiceCELoss

from src.registry import register_loss
from src.utils.logging import get_logger
from .geometric_smooth import GeometricSmoothLoss

logger = get_logger()


@register_loss("WeakSegLoss")
class WeakSegLoss(nn.Module):
    """
    Unified weakly-supervised segmentation loss.
    
    Components:
        - Dice+CE: Strong supervision on labeled frames only (ED/ES)
        - EF Regression: Weak supervision backprops through entire volume curve
        - Geometric Smoothness: 2nd-order temporal coherence (centroid + area)
        - Contrast: Encourages motion variance for healthy patients (EF > 40%)
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        ef_weight: float = 10.0,
        smooth_weight: float = 1.0,
        contrast_weight: float = 0.1
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.ef_weight = ef_weight
        self.smooth_weight = smooth_weight
        self.contrast_weight = contrast_weight

        self.dice_func = DiceCELoss(sigmoid=True, reduction='mean')
        self.mse = nn.MSELoss()
        self.geo_loss = GeometricSmoothLoss()

    def forward(
        self,
        pred_logits: torch.Tensor,
        target_masks: torch.Tensor,
        pred_ef: torch.Tensor,
        target_ef: torch.Tensor,
        frame_mask: torch.Tensor
    ):
        """
        Args:
            pred_logits: (B, T, 1, H, W) - Raw logits
            target_masks: (B, T, 1, H, W) - Ground truth masks
            pred_ef: (B,) - Predicted EF
            target_ef: (B,) - Target EF
            frame_mask: (B, T) - 1.0 if labeled (ED/ES), 0.0 otherwise

        Returns:
            total_loss: Scalar
            loss_dict: Dictionary of individual loss components
        """
        loss_dice = self._compute_dice_loss(pred_logits, target_masks, frame_mask)
        loss_ef = self.mse(pred_ef, target_ef)

        probs = torch.sigmoid(pred_logits)
        loss_smooth = self.geo_loss(probs)
        loss_contrast = self._compute_contrast_loss(probs, target_ef)

        total_loss = (
            self.dice_weight * loss_dice +
            self.ef_weight * loss_ef +
            self.smooth_weight * loss_smooth +
            self.contrast_weight * loss_contrast
        )

        return total_loss, {
            "dice": loss_dice,
            "ef": loss_ef,
            "smooth": loss_smooth,
            "contrast": loss_contrast
        }

    def _compute_dice_loss(
        self,
        pred_logits: torch.Tensor,
        target_masks: torch.Tensor,
        frame_mask: torch.Tensor
    ) -> torch.Tensor:
        B, T = pred_logits.shape[:2]
        total_loss = 0.0
        num_labeled = 0

        for b in range(B):
            for t in range(T):
                if frame_mask[b, t] > 0.5:
                    pred_frame = pred_logits[b, t:t+1]
                    target_frame = target_masks[b, t:t+1]
                    total_loss = total_loss + self.dice_func(pred_frame, target_frame)
                    num_labeled += 1

        if num_labeled == 0:
            return torch.tensor(0.0, device=pred_logits.device)

        return total_loss / num_labeled

    def _compute_contrast_loss(
        self,
        probs: torch.Tensor,
        target_ef: torch.Tensor
    ) -> torch.Tensor:
        if self.contrast_weight <= 0:
            return torch.tensor(0.0, device=probs.device)

        T = probs.shape[1]
        if T <= 1:
            return torch.tensor(0.0, device=probs.device)

        mask_variance = probs.var(dim=1, correction=0).mean()
        should_move = (target_ef > 0.40).float()
        return -1.0 * mask_variance * should_move.mean()
