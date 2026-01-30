import torch
import torch.nn as nn
import torch.nn.functional as F
from src.registry import register_loss

def differentiable_volume(probs: torch.Tensor, pixel_spacing: float = 1.0, step_size: float = 1.0) -> torch.Tensor:
    """
    Calculates LV volume using Monoplane Simpson's rule in a differentiable manner.
    Formula: Volume = sum(diameter^2) * step_size * (pi / 4)
    """
    if probs.dim() == 4:
         probs = probs.squeeze(1) # (B, H, W)
         
    # Soft diameter = sum of probabilities in a row (dim 2 for width)
    diameters_pixels = torch.sum(probs, dim=2) # (B, H)
    
    # Convert to physical units
    diameters_mm = diameters_pixels * pixel_spacing
    
    # Monoplane Simpson's Formula
    area_slices = (diameters_mm ** 2) * (3.14159265359 / 4.0)
    volume = torch.sum(area_slices, dim=1) * step_size
    
    return volume

@register_loss("ConsistencyLoss")
class ConsistencyLoss(nn.Module):
    """
    Consistency Loss to enforce agreement between Geometric EF and Predicted EF.
    L_consistency = |EF_geometric - EF_predicted|^2
    """
    def __init__(self, pixel_spacing: float = 0.3, step_size: float = 0.3, detach_gradients: bool = True):
        super().__init__()
        self.pixel_spacing = pixel_spacing
        self.step_size = step_size
        self.detach_gradients = detach_gradients
        self.mse = nn.MSELoss()

    def forward(self, mask_logits: torch.Tensor, ef_pred: torch.Tensor) -> torch.Tensor:
        # If input is (B=1, T, C, H, W)
        if mask_logits.dim() == 5:
            b, t, _, h, w = mask_logits.shape
            # LV channel (usually class 1)
            lv_probs = F.softmax(mask_logits, dim=2)[:, :, 1, :, :] # (B, T, H, W)
            
            if self.detach_gradients:
                lv_probs_detached = lv_probs.detach()
            else:
                lv_probs_detached = lv_probs

            # Calculate Geometric EF
            lv_probs_flat = lv_probs_detached.view(-1, h, w)
            volumes = differentiable_volume(lv_probs_flat, self.pixel_spacing, self.step_size)
            volumes = volumes.view(b, t)
            
            ed_vol, _ = torch.max(volumes, dim=1)
            es_vol, _ = torch.min(volumes, dim=1)
            
            ef_geo = (ed_vol - es_vol) / (ed_vol + 1e-6)
            
            # Handle ef_pred shapes
            if ef_pred.dim() == 2 and ef_pred.shape[1] == 1:
                ef_pred_val = ef_pred.squeeze(1)
            elif ef_pred.dim() == 1:
                ef_pred_val = ef_pred
            else:
                 ef_pred_val = ef_pred.view(b, -1).mean(dim=1)

            return self.mse(ef_geo, ef_pred_val)
            
        else:
            return torch.tensor(0.0, device=mask_logits.device, requires_grad=True)