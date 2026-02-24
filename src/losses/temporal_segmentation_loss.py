import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss

from src.registry import register_loss
from src.utils.logging import get_logger

logger = get_logger()


@register_loss("TemporalWeakSegLoss")
class TemporalWeakSegLoss(nn.Module):
    """
    Computes spatial and volumetric regression losses for echocardiography video segments.
    Focuses strictly on mask-guided spatial grounding and direct volume regression.
    """
    def __init__(
        self,
        dice_weight: float = 1.0,
        volume_weight: float = 1.0,
        sv_weight: float = 1.0,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.volume_weight = volume_weight
        self.sv_weight = sv_weight
        
        self.dice_func = DiceCELoss(sigmoid=True, reduction='mean')

    def forward(
        self,
        pred_logits: torch.Tensor,
        target_masks: torch.Tensor,
        frame_mask: torch.Tensor,
        target_edv: torch.Tensor,
        target_esv: torch.Tensor,
        pred_edv: torch.Tensor,
        pred_esv: torch.Tensor,
        **kwargs
    ):
        """
        Accepts **kwargs to gracefully handle legacy pipeline arguments (e.g., pred_vol_curve)
        during the architectural transition without breaking the forward pass.
        """
        loss_dice = self._compute_dice_loss(pred_logits, target_masks, frame_mask)
        loss_vol, loss_edv, loss_esv, loss_sv = self._compute_volume_loss(
            pred_edv, pred_esv, target_edv, target_esv
        )

        total_loss = (self.dice_weight * loss_dice) + (self.volume_weight * loss_vol)

        loss_dict = {
            "dice_loss": loss_dice,
            "volume_loss": loss_vol,
            "edv_loss": loss_edv,
            "esv_loss": loss_esv,
            "sv_loss": loss_sv,
        }

        return total_loss, loss_dict

    def _compute_dice_loss(
        self,
        pred_logits: torch.Tensor,
        target_masks: torch.Tensor,
        frame_mask: torch.Tensor
    ) -> torch.Tensor:
        """Vectorized Dice+CE on valid labeled frames."""
        if pred_logits.shape[1] == 1 and pred_logits.shape[2] > 1:
            pred_logits = pred_logits.permute(0, 2, 1, 3, 4)

        if target_masks.shape[1] == 1 and target_masks.shape[2] > 1:
            target_masks = target_masks.permute(0, 2, 1, 3, 4)

        batch_size, seq_len, channels, height, width = pred_logits.shape

        pred_flat = pred_logits.reshape(-1, channels, height, width)
        target_flat = target_masks.reshape(-1, channels, height, width)
        mask_flat = frame_mask.reshape(-1)

        valid_indices = mask_flat > 0.5

        if valid_indices.sum() == 0:
            return 0.0 * pred_logits.sum()

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
    ):
        """Computes L1 loss with Stroke Volume constraint."""
        p_edv = pred_edv.view(-1)
        t_edv = target_edv.view(-1)
        p_esv = pred_esv.view(-1)
        t_esv = target_esv.view(-1)
        
        valid_edv = t_edv >= 0
        valid_esv = t_esv >= 0
        
        loss_edv = torch.tensor(0.0, device=pred_edv.device)
        loss_esv = torch.tensor(0.0, device=pred_esv.device)
        loss_sv = torch.tensor(0.0, device=pred_edv.device)
        has_loss = False
        
        if valid_edv.any():
            loss_edv = F.l1_loss(p_edv[valid_edv], t_edv[valid_edv])
            has_loss = True
            
        if valid_esv.any():
            loss_esv = F.l1_loss(p_esv[valid_esv], t_esv[valid_esv])
            has_loss = True
            
        valid_both = valid_edv & valid_esv
        if valid_both.any():
            pred_sv = p_edv[valid_both] - p_esv[valid_both]
            true_sv = t_edv[valid_both] - t_esv[valid_both]
            loss_sv = F.l1_loss(pred_sv, true_sv)
            
        if not has_loss:
            total_loss = 0.0 * pred_edv.sum() + 0.0 * pred_esv.sum()
        else:
            total_loss = loss_edv + loss_esv + (self.sv_weight * loss_sv)
            
        return total_loss, loss_edv, loss_esv, loss_sv