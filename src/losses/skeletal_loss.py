import torch
import torch.nn as nn
from .topology_loss import TopologyLoss

class SkeletalLoss(nn.Module):
    """
    Multi-Task Loss for Spatio-Temporal Skeletal Model.
    
    Components:
    1. Supervised: MSE between predicted and true points (only on labeled frames).
    2. Unsupervised: Smoothness constraint (MSE between P_t and P_{t-1}).
    3. Unsupervised: Topology constraint (Spring & Collapse prevention).
    """
    def __init__(self, smooth_weight=0.1, topology_weight=1.0):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.smooth_weight = smooth_weight
        self.topology_loss = TopologyLoss(weight=topology_weight)

    def forward(self, preds, targets, frame_mask=None):
        """
        Args:
            preds: (B, T, 42, 2)
            targets: (B, T, 42, 2)
            frame_mask: (B, T) - 1.0 for labeled, 0.0 for unlabeled
        """
        # 1. Supervised Loss (L_point)
        # Calculate MSE per frame
        loss_mse = self.mse(preds, targets) # (B, T, 42, 2)
        loss_mse = loss_mse.mean(dim=(2, 3)) # (B, T) - Mean over points and coords
        
        if frame_mask is not None:
             # Apply mask
             loss_sup = (loss_mse * frame_mask).sum()
             
             # Normalize by number of labeled frames (plus eps)
             num_labeled = frame_mask.sum() + 1e-6
             loss_sup = loss_sup / num_labeled
        else:
             loss_sup = loss_mse.mean()
             
        # 2. Smoothness Loss (L_smooth)
        # Minimize velocity: || P_t - P_{t-1} ||^2
        if preds.shape[1] > 1:
            diffs = preds[:, 1:] - preds[:, :-1] # (B, T-1, 42, 2)
            # We want to minimize the magnitude of jumps
            loss_smooth = (diffs ** 2).mean() 
        else:
            loss_smooth = torch.tensor(0.0, device=preds.device)
            
        # 3. Topology Loss (Structure)
        loss_topo = self.topology_loss(preds)
        
        total_loss = loss_sup + self.smooth_weight * loss_smooth + loss_topo
        
        return total_loss, {
            "loss_sup": loss_sup,
            "loss_smooth": loss_smooth,
            "loss_topo": loss_topo
        }
