import torch
import torch.nn as nn
from .topology_loss import TopologyLoss
from src.registry import register_loss

@register_loss("SkeletalLoss")
class SkeletalLoss(nn.Module):
    """
    Configurable Loss for Skeletal Tracking.

    Phased Training Strategy:
    1. Phase 1 (Geometry): High skeletal_weight, High topo_weight, Zero smooth_weight.
       -> Fixes the collapsed points problem.
    2. Phase 2 (Dynamics): Add smooth_weight.
    """
    def __init__(self, smooth_weight=0.0, topology_weight=1.0, skeletal_weight=1.0):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.smooth_weight = smooth_weight
        self.skeletal_weight = skeletal_weight
        self.topology_loss = TopologyLoss(weight=topology_weight)

    def forward(self, preds, targets, frame_mask=None):
        loss_sup = self._calc_supervised_loss(preds, targets, frame_mask)
        loss_smooth = self._calc_smoothness_loss(preds)
        loss_topo = self.topology_loss(preds)
        
        total_loss = (self.skeletal_weight * loss_sup) + \
                     (self.smooth_weight * loss_smooth) + \
                     loss_topo
        
        return total_loss, {
            "loss_sup": loss_sup,
            "loss_smooth": loss_smooth,
            "loss_topo": loss_topo
        }

    def _calc_supervised_loss(self, preds, targets, frame_mask=None):
        if self.skeletal_weight <= 0:
            return torch.tensor(0.0, device=preds.device)
            
        loss_mse = self.mse(preds, targets).mean(dim=(2, 3))
        
        if frame_mask is not None:
             loss_sup = (loss_mse * frame_mask).sum()
             num_labeled = frame_mask.sum() + 1e-6
             return loss_sup / num_labeled
        
        return loss_mse.mean()

    def _calc_smoothness_loss(self, preds):
        if self.smooth_weight <= 0 or preds.shape[1] <= 1:
            return torch.tensor(0.0, device=preds.device)
            
        diffs = preds[:, 1:] - preds[:, :-1]
        return (diffs ** 2).mean()
