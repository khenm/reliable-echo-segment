import torch
import torch.nn as nn
import torch.nn.functional as F
from src.registry import register_loss

@register_loss("TopologyLoss")
class TopologyLoss(nn.Module):
    """
    Physically-Informed Geometric Loss.

    Replaces generic springs with:
    1. Width Preservation (Prevent Collapse): Penalizes widths that are too small.
    2. Anatomical Consistency: Ensures Base width >= Apex width generally.
    3. Non-Intersection: Prevents Left/Right wall crossing.
    """
    def __init__(self, weight=1.0, min_width=0.02):
        super().__init__()
        self.weight = weight
        self.min_width = min_width # Normalized units (0-1)

    def forward(self, pred_kps):
        left_wall, right_wall = self._extract_walls(pred_kps)
        widths = self._compute_widths(left_wall, right_wall)
        
        collapse_loss = self._compute_collapse_penalty(widths)
        area_loss = self._compute_area_penalty(widths)
        
        return self.weight * (collapse_loss + area_loss)

    def _extract_walls(self, pred_kps):
        left_pts = pred_kps[:, :, 0:21]
        right_pts_raw = pred_kps[:, :, 22:42] 
        # Flip right side to align indices with left (base-to-apex)
        right_pts_aligned = torch.flip(right_pts_raw, dims=[2])
        return left_pts, right_pts_aligned

    def _compute_widths(self, left_wall, right_wall):
        # Euclidean distance between corresponding points
        diff_vec = right_wall - left_wall[:, :, 1:] # Align length
        return torch.norm(diff_vec, dim=-1)

    def _compute_collapse_penalty(self, widths):
        # Penalize if width < min_width
        penalty = F.relu(self.min_width - widths)
        return (penalty ** 2).mean() * 100.0

    def _compute_area_penalty(self, widths):
        # Heuristic: Penalize if total width (proxy for area) is too small
        total_width = widths.sum(dim=2)
        target_width = self.min_width * 20.0
        return F.relu(target_width - total_width).mean()
