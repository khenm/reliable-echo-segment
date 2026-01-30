import math
from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor
from src.registry import register_loss

@register_loss("DifferentiableEFLoss")
class DifferentiableEFLoss(nn.Module):
    """
    Differentiable Ejection Fraction Loss using Simpson's Method of Disks.
    Calculates LV volume from segmentation masks and derives EF for training.
    """
    def __init__(
        self, 
        pixel_spacing: float = 1.0, 
        weight: float = 1.0, 
        foreground_class: int = 1,
        eps: float = 1e-6
    ) -> None:
        super().__init__()
        self.pixel_spacing = pixel_spacing
        self.weight = weight
        self.foreground_class = foreground_class
        self.eps = eps
        self.mse = nn.MSELoss()

    def get_volume(self, mask: Tensor) -> Tensor:
        """
        Calculates LV Volume using Simpson's Rule (Method of Disks).
        Returns: Volume per frame, shape [B, T].
        """
        if mask.ndim == 4:
            # (B, C, H, W) -> Add T=1
            mask = mask.unsqueeze(2)

        if mask.shape[1] == 1:
            lv_prob = mask[:, 0]
        else:
            lv_prob = mask[:, self.foreground_class]
        
        # lv_prob shape: (B, T, H, W)
        diameter = torch.sum(lv_prob, dim=3) * self.pixel_spacing  # [B, T, H]
        disk_areas = (math.pi / 4.0) * (diameter ** 2)
        volume = torch.sum(disk_areas, dim=2) * self.pixel_spacing  # [B, T]
        
        return volume

    def forward(self, pred_masks: Tensor, target_ef: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Computes the differentiable EF loss.
        """
        pred_probs = torch.sigmoid(pred_masks)
        volumes = self.get_volume(pred_probs)
        
        ed_vol, _ = torch.max(volumes, dim=1)
        es_vol, _ = torch.min(volumes, dim=1)
        
        pred_ef = (ed_vol - es_vol) / (ed_vol + self.eps)
        
        target_ef_norm = target_ef / 100.0 if target_ef.max() > 1.0 else target_ef
        
        loss = self.mse(pred_ef, target_ef_norm)
        
        return self.weight * loss, pred_ef
