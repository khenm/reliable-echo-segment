import torch
import torch.nn as nn
from monai.losses import DiceCELoss
from src.registry import register_loss
from src.utils.logging import get_logger

logger = get_logger()


@register_loss("SegmentationLoss")
class SegmentationLoss(nn.Module):
    """
    Phased Loss for Temporal Segmentation.

    Phase 1 (Geometry, epochs 1-50): High dice_weight, zero smooth_weight
        - Dice+CE forces mask overlap with GT, prevents sphere collapse
    Phase 2 (Dynamics, epochs 51-100): Add smooth_weight
        - Temporal smoothness penalizes mask flicker between frames
    """
    def __init__(self, dice_weight=1.0, smooth_weight=0.0, phase_switch_epoch=50):
        super().__init__()
        self.dice_ce = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        self.dice_weight = dice_weight
        self.smooth_weight = smooth_weight
        self.phase_switch_epoch = phase_switch_epoch
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def forward(self, pred_logits, target_masks, frame_mask=None):
        """
        Args:
            pred_logits: (B, 1, T, H, W) - Raw logits from decoder
            target_masks: (B, 1, T, H, W) - Ground truth binary masks
            frame_mask: (B, T) - Optional mask for valid frames
        Returns:
            total_loss: Scalar
            components: Dict of loss components
        """
        B, C, T, H, W = pred_logits.shape

        if target_masks.shape[-2:] != pred_logits.shape[-2:]:
            pred_logits = nn.functional.interpolate(
                pred_logits.view(B, C * T, H, W),
                size=target_masks.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).view(B, C, T, *target_masks.shape[-2:])

        if frame_mask is not None:
            frame_mask_expanded = frame_mask.view(B, 1, T, 1, 1).expand_as(pred_logits)

            valid_count = frame_mask.sum()
            if valid_count == 0:
                loss_dice = torch.tensor(0.0, device=pred_logits.device, requires_grad=True)
            else:
                pred_flat = pred_logits.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
                target_flat = target_masks.permute(0, 2, 1, 3, 4).reshape(B * T, C, *target_masks.shape[-2:])
                mask_flat = frame_mask.view(B * T)

                valid_idx = torch.nonzero(mask_flat).squeeze(-1)

                if valid_idx.numel() > 0:
                    loss_dice = self.dice_ce(pred_flat[valid_idx], target_flat[valid_idx])
                else:
                    loss_dice = torch.tensor(0.0, device=pred_logits.device, requires_grad=True)
        else:
            loss_dice = self.dice_ce(pred_logits, target_masks)

        effective_smooth_weight = self.smooth_weight
        if self.current_epoch < self.phase_switch_epoch:
            effective_smooth_weight = 0.0

        loss_smooth = torch.tensor(0.0, device=pred_logits.device)
        if effective_smooth_weight > 0 and T > 1:
            pred_probs = torch.sigmoid(pred_logits)
            diff = pred_probs[:, :, 1:] - pred_probs[:, :, :-1]

            if frame_mask is not None:
                pair_mask = frame_mask[:, :-1] * frame_mask[:, 1:]
                pair_mask = pair_mask.view(B, 1, T - 1, 1, 1).expand_as(diff)
                loss_smooth = (diff.abs() * pair_mask).sum() / (pair_mask.sum() + 1e-6)
            else:
                loss_smooth = diff.abs().mean()

        total_loss = (self.dice_weight * loss_dice) + (effective_smooth_weight * loss_smooth)

        return total_loss, {
            "loss_dice": loss_dice,
            "loss_smooth": loss_smooth,
        }
