
import torch
import torch.nn as nn
import torch.nn.functional as F

def differentiable_volume(probs: torch.Tensor, pixel_spacing: float = 1.0, step_size: float = 1.0) -> torch.Tensor:
    """
    Calculates LV volume using Monoplane Simpson's rule in a differentiable manner.
    
    Formula: Volume = sum(diameter^2) * step_size * (pi / 4)
    
    Args:
        probs (torch.Tensor): Soft segmentation probabilities for the LV class. Shape (B, 1, H, W) or (B, H, W).
        pixel_spacing (float): Mm per pixel.
        step_size (float): Thickness of each slice (usually equal to pixel_spacing for 2D analysis).
    
    Returns:
        torch.Tensor: Calculated volume for each item in the batch. Shape (B,).
    """
    if probs.dim() == 4:
         probs = probs.squeeze(1) # (B, H, W)
         
    # 1. Calculate Diameters (Row sums)
    # We assume the image is oriented such that slices are rows.
    # If not, we might need to transpose. Standard Echo apical view usually has apex at top/bottom.
    # Simpson's method slices perpendicular to the long axis. 
    # Assuming vertical long axis, we sum across width (dim 2).
    
    # Soft diameter = sum of probabilities in a row
    diameters_pixels = torch.sum(probs, dim=2) # (B, H)
    
    # Convert to physical units
    diameters_mm = diameters_pixels * pixel_spacing
    
    # 2. Apply Monoplane Simpson's Formula
    # Volume = sum(d^2) * step * pi / 4
    
    area_slices = (diameters_mm ** 2) * (3.14159265359 / 4.0)
    volume = torch.sum(area_slices, dim=1) * step_size
    
    return volume

class ConsistencyLoss(nn.Module):
    """
    Consistency Loss to enforce agreement between Geometric EF and Predicted EF.
    
    L_consistency = |EF_geometric - EF_predicted|^2
    
    Where EF_geometric is calculated from the predicted masks using Simpson's rule.
    """
    def __init__(self, pixel_spacing: float = 0.3, step_size: float = 0.3, detach_gradients: bool = True): # Defaults, should be adjustable
        super().__init__()
        self.pixel_spacing = pixel_spacing
        self.step_size = step_size
        self.detach_gradients = detach_gradients
        
        # Loss function (MSE)
        self.mse = nn.MSELoss()

    def forward(self, mask_logits: torch.Tensor, ef_pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mask_logits (torch.Tensor): Predicted segmentation logits (B, C, H, W) or (B, T, C, H, W).
                                        We need to identify ED and ES frames.
                                        If input is a batch of independent frames (B, ...), we can't easily calc EF 
                                        unless the batch *is* the sequence or we have metadata.
                                        
            ef_pred (torch.Tensor): Predicted EF from regression head (B, 1).
            
        Returns:
            torch.Tensor: Scalar loss.
        """
        # Note: This logic assumes we can calculate EF from the input text.
        # If the input is just random frames, EF Calculation is impossible.
        # We need a strategy to handle this.
        # Options:
        # 1. Input is (B, T, C, H, W) -> Video clip. Ideal.
        # 2. Input is (B, C, H, W) but we know which ones are ED/ES.
        
        # Implementation assumes input includes enough info to find ED/ES.
        # Let's assume input is (B, T, C, H, W) for now, or T is merged into B but we restructure.
        
        # If input is (B=1, T, C, H, W)
        if mask_logits.dim() == 5:
            b, t, _, h, w = mask_logits.shape
            # Reshape for volume calc: (B*T, C, H, W)
            # Take LV channel (usually class 1, but depends on config)
            lv_probs = F.softmax(mask_logits, dim=2)[:, :, 1, :, :] # (B, T, H, W)
            
            # Calculate volume for all frames
            # Flatten to (B*T, H, W)
            # 1. Conditionally DETACH the segmentation probabilities
            if self.detach_gradients:
                lv_probs_detached = lv_probs.detach()
            else:
                lv_probs_detached = lv_probs

            # 2. Calculate Geometric EF using the detached probabilities
            lv_probs_flat = lv_probs_detached.view(-1, h, w)
            volumes = differentiable_volume(lv_probs_flat, self.pixel_spacing, self.step_size)
            volumes = volumes.view(b, t)
            
            # Identify ED and ES
            ed_vol, _ = torch.max(volumes, dim=1)
            es_vol, _ = torch.min(volumes, dim=1)
            
            # Calculate Geometric EF
            # Handle potential div by zero or negative
            ef_geo = (ed_vol - es_vol) / (ed_vol + 1e-6)
            
            # ef_pred might be (B, 1) or (B, T). If (B, T) we usually take average/max or it's a video-level pred.
            if ef_pred.dim() == 2 and ef_pred.shape[1] == 1:
                ef_pred_val = ef_pred.squeeze(1)
            elif ef_pred.dim() == 1:
                ef_pred_val = ef_pred
            else:
                 # If ef_pred is per frame? Usually EF is video-level.
                 # Let's assume B, 1
                 ef_pred_val = ef_pred.view(b, -1).mean(dim=1)

            loss = self.mse(ef_geo, ef_pred_val)
            return loss
            
        else:
            # Fallback for single frames / random batches?
            # Cannot calc EF from single frame. Return 0.
            return torch.tensor(0.0, device=mask_logits.device, requires_grad=True)

