import torch
import torch.nn as nn

class DifferentiableSimpson(nn.Module):
    """
    Computes Left Ventricle (LV) Volume using Single-Plane Simpson's Method.
    Formula: V = (8 * A^2) / (3 * pi * L)
    
    A: Area of LV mask
    L: Length of LV (Long axis)
    """
    def __init__(self):
        super().__init__()

    def forward(self, mask):
        """
        Args:
            mask: (B, T, H, W) or (B, 1, T, H, W). Soft prob mask (0-1).
        Returns:
            volume: (B, T)
        """
        if mask.dim() == 5:
            mask = mask.squeeze(1) # Remove channel if 1
            
        # 1. Calculate Area (A)
        # Sum of pixels per frame
        # mask is (B, T, H, W)
        A = mask.sum(dim=(2, 3)) # (B, T)
        
        # 2. Calculate Length (L)
        # We approximate L by the bounding box height over the y-axis.
        # Project mask to Y-axis
        proj_y = mask.sum(dim=3) # (B, T, H)
        existence_y = torch.sigmoid((proj_y - 0.1) * 20.0) # Sharp sigmoid
        L = existence_y.sum(dim=2) # (B, T)
        
        # Avoid division by zero
        L = torch.clamp(L, min=1.0)
        
        # 3. Calculate Volume
        # V = (8 * A^2) / (3 * pi * L)
        factor = 8.0 / (3.0 * 3.14159)
        V = factor * (A ** 2) / L
        
        return V

    @staticmethod
    def calculate_ef(volumes):
        """
        Args:
            volumes: (B, T)
        Returns:
            ef: (B, 1)
        """
        # EDV = max volume
        # ESV = min volume
        
        edv, _ = volumes.max(dim=1)
        esv, _ = volumes.min(dim=1)
        
        ef = (edv - esv) / (edv + 1e-6)
        return ef.unsqueeze(1)
