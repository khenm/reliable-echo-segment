import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class GeometricSmoothLoss(nn.Module):
    """
    Enforces 2nd-Order Smoothness (Minimizes Acceleration).
    Allows linear motion (expansion/contraction) but punishes jitter.
    """

    def __init__(self, centroid_weight: float = 1.0, area_weight: float = 1.0):
        super().__init__()
        self.centroid_weight = centroid_weight
        self.area_weight = area_weight

    def forward(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            masks: (B, T, 1, H, W) - Probability masks
        Returns:
            Scalar loss value
        """
        B, T, C, H, W = masks.shape
        masks = masks.view(B, T, H, W)

        y_coords = torch.linspace(0, 1, H, device=masks.device).view(1, 1, H, 1)
        x_coords = torch.linspace(0, 1, W, device=masks.device).view(1, 1, 1, W)

        area = masks.sum(dim=(2, 3)) / (H * W)
        safe_area = area + 1e-6

        y_cent = (masks * y_coords).sum(dim=(2, 3)) / safe_area
        x_cent = (masks * x_coords).sum(dim=(2, 3)) / safe_area
        centroid = torch.stack([x_cent, y_cent], dim=-1)

        cent_t = centroid[:, 1:-1]
        cent_prev = centroid[:, :-2]
        cent_next = centroid[:, 2:]
        acc_centroid = cent_next - (2 * cent_t) + cent_prev
        loss_centroid = torch.mean(acc_centroid ** 2)

        area_t = area[:, 1:-1]
        area_prev = area[:, :-2]
        area_next = area[:, 2:]
        acc_area = area_next - (2 * area_t) + area_prev
        loss_area = torch.mean(acc_area ** 2)

        return (self.centroid_weight * loss_centroid) + (self.area_weight * loss_area)
