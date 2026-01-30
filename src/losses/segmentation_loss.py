import torch
import torch.nn as nn
from monai.losses import DiceCELoss
from src.registry import register_loss
from src.utils.logging import get_logger

logger = get_logger()


@register_loss("WeaklySupervisedSegLoss")
class WeaklySupervisedSegLoss(nn.Module):
    """
    Weakly Supervised Loss for Temporal Segmentation.

    Components:
        1. Dice+CE (Strong): Applied to labeled frames only (via frame_mask)
        2. EF Regression (Weak): Backprops through entire volume curve
        3. Temporal Smoothness (Unsupervised): Prevents frame jitter

    The EF loss enables supervision on ALL frames through the differentiable
    volume calculation path, even when only ED/ES frames have ground-truth masks.
    """
    def __init__(self, dice_weight=1.0, ef_weight=1.0, smooth_weight=0.5, phase_switch_epoch=50):
        super().__init__()
        self.dice_ce = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        self.mse = nn.MSELoss()

        self.dice_weight = dice_weight
        self.ef_weight = ef_weight
        self.smooth_weight = smooth_weight
        self.phase_switch_epoch = phase_switch_epoch
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def forward(self, pred_logits, target_masks, pred_ef, target_ef, frame_mask=None):
        """
        Args:
            pred_logits: (B, 1, T, H, W) - Raw logits from decoder
            target_masks: (B, 1, T, H, W) - Ground truth binary masks
            pred_ef: (B, 1) - EF calculated from predicted masks
            target_ef: (B, 1) - Clinical EF ground truth
            frame_mask: (B, T) - Optional mask for valid frames (1=labeled, 0=unlabeled)
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

        # 1. Strong Supervision (Dice+CE) - ONLY on labeled frames
        if frame_mask is not None:
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

        # 2. Weak Supervision (EF Regression) - Backprops to ALL frames
        loss_ef = self.mse(pred_ef, target_ef)

        # 3. Unsupervised (Temporal Smoothness) - Phased activation
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

        total_loss = (
            self.dice_weight * loss_dice +
            self.ef_weight * loss_ef +
            effective_smooth_weight * loss_smooth
        )

        return total_loss, {
            "loss_dice": loss_dice,
            "loss_ef": loss_ef,
            "loss_smooth": loss_smooth,
        }

