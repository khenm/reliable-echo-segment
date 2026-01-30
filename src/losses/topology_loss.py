import torch
import torch.nn as nn
import torch.nn.functional as F
from src.registry import register_loss


@register_loss("TopologyLoss")
class TopologyLoss(nn.Module):
    """
    Physically-Informed Geometric Loss for APEX-FIRST keypoint data.

    Prevents wall collapse and zero-volume hearts by penalizing
    insufficient widths between left/right wall pairs.

    Data Layout (42 keypoints):
        - Index 0: Apex Tip
        - Index 1-20: Left Wall (Apex -> Base)
        - Index 21-41: Right Wall (Base -> Apex)
    """

    WALL_SIZE = 20
    COLLAPSE_SCALE = 100.0

    def __init__(self, weight: float = 1.0, min_width: float = 0.02):
        super().__init__()
        self.weight = weight
        self.min_width = min_width

    def forward(self, pred_kps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_kps: Predicted keypoints (B, T, 42, 2).

        Returns:
            Weighted sum of collapse and area penalties.
        """
        all_widths = self._compute_widths(pred_kps)
        collapse_penalty = self._collapse_penalty(all_widths)
        area_penalty = self._area_penalty(all_widths)
        return self.weight * (collapse_penalty + area_penalty)

    def _compute_widths(self, pred_kps: torch.Tensor) -> torch.Tensor:
        left_wall = pred_kps[:, :, 1:21]
        right_wall_aligned = torch.flip(pred_kps[:, :, 22:42], dims=[2])

        wall_widths = torch.norm(left_wall - right_wall_aligned, dim=-1)

        base_l = pred_kps[:, :, 20:21]
        base_r = pred_kps[:, :, 21:22]
        base_width = torch.norm(base_l - base_r, dim=-1)

        return torch.cat([wall_widths, base_width], dim=2)

    def _collapse_penalty(self, widths: torch.Tensor) -> torch.Tensor:
        collapse_loss = F.relu(self.min_width - widths)
        return (collapse_loss ** 2).mean() * self.COLLAPSE_SCALE

    def _area_penalty(self, widths: torch.Tensor) -> torch.Tensor:
        total_width = widths.sum(dim=2)
        target_width = self.min_width * (self.WALL_SIZE + 1)
        return F.relu(target_width - total_width).mean()
