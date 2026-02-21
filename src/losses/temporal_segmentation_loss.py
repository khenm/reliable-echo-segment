import torch
import torch.nn as nn
from monai.losses import DiceCELoss
import torch.nn.functional as F

from src.registry import register_loss
from src.utils.logging import get_logger

logger = get_logger()


@register_loss("TemporalWeakSegLoss")
class TemporalWeakSegLoss(nn.Module):
    def __init__(
        self,
        dice_weight: float = 1.0,
        volume_weight: float = 1.0,      # Increased importance
        phase_geom_weight: float = 5.0,  # High weight to tether the system
        multi_beat_weight: float = 0.1
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.volume_weight = volume_weight
        self.phase_geom_weight = phase_geom_weight
        self.multi_beat_weight = multi_beat_weight

        self.dice_func = DiceCELoss(sigmoid=True, reduction='mean')
        self.mse = nn.MSELoss()

    def forward(
        self,
        pred_logits: torch.Tensor,    # (B, 1, T, H, W)
        target_masks: torch.Tensor,   # (B, T, 1, H, W)
        frame_mask: torch.Tensor,    # (B, T)
        target_edv: torch.Tensor,
        target_esv: torch.Tensor,
        pred_edv: torch.Tensor,
        pred_esv: torch.Tensor,
        pred_raw_area: torch.Tensor,  # From detached mask_probs
        pred_vol_curve: torch.Tensor, # From Fourier Head
        pred_phase_vel: torch.Tensor  # For rhythm stability
    ):
        # 1. Spatial Anchor (Dice)
        loss_dice = self._compute_dice_loss(pred_logits, target_masks, frame_mask)

        # 2. Peak Volume Supervision (Supervise the Fourier Extrema)
        loss_vol = self._compute_volume_loss(pred_edv, pred_esv, target_edv, target_esv)

        # 3. Phase-Geometry Alignment (Min-Max Normalized)
        # Ensures 'Max Area' == 'Max Volume'
        loss_phase_geom = self._compute_alignment_loss(pred_raw_area, pred_vol_curve)

        # 4. Rhythm Stability (Minimize Phase Velocity Variance)
        # Prevents the 'clock' from skipping beats
        loss_rhythm = pred_phase_vel.var(dim=1).mean()

        total_loss = (
            self.dice_weight * loss_dice +
            self.volume_weight * loss_vol +
            self.phase_geom_weight * loss_phase_geom +
            self.multi_beat_weight * loss_rhythm
        )

        return total_loss, {
            "dice": loss_dice,
            "volume": loss_vol,
            "phase_geom": loss_phase_geom,
            "multi_beat": loss_rhythm
        }

    def _compute_alignment_loss(self, area, vol):
        """Standardized Min-Max Alignment."""
        area = area.squeeze()
        vol = vol.squeeze()
        
        # Min-Max Normalize both to [0, 1] to compare morphology
        a_norm = (area - area.min(1, keepdim=True)[0]) / (area.max(1, keepdim=True)[0] - area.min(1, keepdim=True)[0] + 1e-6)
        v_norm = (vol - vol.min(1, keepdim=True)[0]) / (vol.max(1, keepdim=True)[0] - vol.min(1, keepdim=True)[0] + 1e-6)
        
        return self.mse(a_norm, v_norm)

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
        pred_edv: torch.Tensor,
        pred_esv: torch.Tensor,
        target_edv: torch.Tensor,
        target_esv: torch.Tensor
    ) -> torch.Tensor:
        """Computes L1 loss between predicted and target volumes."""
        p_edv = pred_edv.view(-1)
        t_edv = target_edv.view(-1)
        
        p_esv = pred_esv.view(-1)
        t_esv = target_esv.view(-1)
        
        # Filter out invalid targets (-1 or whatever marker for missing data)
        valid_edv = t_edv >= 0
        valid_esv = t_esv >= 0
        
        loss_vol = []
        
        if valid_edv.any():
            loss_vol.append(F.l1_loss(p_edv[valid_edv], t_edv[valid_edv]))
            
        if valid_esv.any():
            loss_vol.append(F.l1_loss(p_esv[valid_esv], t_esv[valid_esv]))
            
        if len(loss_vol) > 0:
            return torch.stack(loss_vol).mean()
        else:
            return torch.tensor(0.0, device=pred_edv.device)
