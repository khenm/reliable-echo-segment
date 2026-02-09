import torch
import torch.nn as nn

class VolumeLoss(nn.Module):
    def __init__(self, lambda_vol=0.1, sigmoid=True):
        super().__init__()
        self.lambda_vol = lambda_vol
        self.sigmoid = sigmoid
        self.loss_fn = nn.L1Loss()

    def get_volume(self, mask_probs):
        """
        Differentiable Volume Calculation
        Args:
            mask_probs: (Batch, C, T, H, W) or (Batch, C, H, W)
        Returns:
            volume: (Batch, ) - Sum of pixels (Area/Volume proxy)
        """
        dims = tuple(range(2, mask_probs.ndim))
        vol = torch.sum(mask_probs, dim=dims) 
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

        # 3. Compute L1 Loss on Volume
        loss = self.loss_fn(pred_vol, target_vol)

        return self.lambda_vol * loss
