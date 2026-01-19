import torch
import torch.nn.functional as F
import math

class VideoAugmentor:
    """
    Provides consistent augmentations for video tensors of shape (B, C, T, H, W).
    Ensures that spatial transformations are applied consistently across ALL frames (T).
    """
    def __init__(self, hflip=True, shift_limit=0.1):
        self.hflip = hflip
        self.shift_limit = shift_limit

    def __call__(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T, H, W).
        Returns:
            torch.Tensor: Augmented tensor of same shape.
        """
        return self.augment(x)

    def augment(self, x):
        B, C, T, H, W = x.shape
        
        # 1. Random Horizontal Flip
        if self.hflip and torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[-1]) # Flip W dimension

        # 2. Random Translation (Shift)
        # We apply the SAME shift to all frames in the video to maintain temporal consistency
        if self.shift_limit > 0:
            dx = (torch.rand(1).item() * 2 - 1) * self.shift_limit * W
            dy = (torch.rand(1).item() * 2 - 1) * self.shift_limit * H
            
            # Create affine grid
            # Theta: (B, 2, 3)
            # Translation in grid_sample uses normalized coordinates [-1, 1]
            # tx refers to horizontal shift, ty to vertical
            tx = -dx / (W / 2.0)
            ty = -dy / (H / 2.0)
            
            theta = torch.tensor([[1, 0, tx], [0, 1, ty]], dtype=x.dtype, device=x.device)
            theta = theta.repeat(B * T, 1, 1) # Apply to each frame efficiently
            
            grid = F.affine_grid(theta, torch.Size((B * T, C, H, W)), align_corners=False)
            
            # Reshape x to (B*T, C, H, W) for grid_sample
            x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            x_shifted = F.grid_sample(x_reshaped, grid, mode='bilinear', padding_mode='border', align_corners=False)
            
            # Reshape back to (B, C, T, H, W) -> (B, T, C, H, W) -> permute
            x = x_shifted.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)

        return x
