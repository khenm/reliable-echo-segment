import torch
import torch.nn as nn
import torch.nn.functional as F

class LogVolumeLoss(nn.Module):
    def __init__(self, lambda_vol=0.1, sigmoid=True):
        super().__init__()
        self.lambda_vol = lambda_vol
        self.sigmoid = sigmoid
        # SmoothL1 acts like Log-Cosh: Square error for small diffs, Linear for large diffs
        self.regression_loss = nn.SmoothL1Loss(beta=1.0) 

    def get_volume(self, mask_probs):
        """
        Differentiable Volume Calculation
        Args:
            mask_probs: (Batch, 1, H, W) - Soft probabilities (0..1)
        Returns:
            volume: (Batch, ) - Sum of pixels (Area/Volume proxy)
        """
        # Sum probabilities to get soft area/volume
        # If you have pixel spacing (cm/pixel), multiply here: sum * spacing_x * spacing_y
        vol = torch.sum(mask_probs, dim=(1, 2, 3)) 
        return vol

    def forward(self, pred_masks, target_masks):
        """
        Args:
            pred_masks: Logits or Probs
            target_masks: Binary Ground Truth (0 or 1)
        """
        # 1. Ensure predictions are probabilities (0..1)
        if self.sigmoid:
            pred_probs = torch.sigmoid(pred_masks)
        else:
            pred_probs = pred_masks
        
        # 2. Compute Volumes (Differentiable)
        pred_vol = self.get_volume(pred_probs)
        target_vol = self.get_volume(target_masks.float())

        # 3. Log-Space Transformation (The "Equalizer" for ESV vs EDV)
        # Add epsilon for numerical stability
        log_pred = torch.log(pred_vol + 1e-6)
        log_target = torch.log(target_vol + 1e-6)

        # 4. Compute Loss (Behaves like Log-Cosh)
        loss = self.regression_loss(log_pred, log_target)

        return self.lambda_vol * loss
