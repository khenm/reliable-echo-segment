import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logging import get_logger

logger = get_logger()


class CycleConsistencyLoss(nn.Module):
    """
    Cycle Consistency Loss for weak supervision on unlabeled frames.
    
    Warps predicted mask from t to t+1 using estimated motion,
    then compares to actual prediction at t+1.
    
    This turns unlabeled intermediate frames into consistency training data.
    """

    def __init__(self, lambda_cycle: float = 1.0):
        super().__init__()
        self.lambda_cycle = lambda_cycle

    def _compute_flow_from_diff(
        self,
        frame_t: torch.Tensor,
        frame_t1: torch.Tensor
    ) -> torch.Tensor:
        """
        Lightweight optical flow estimation via gradient-based method.
        Uses spatial gradients + temporal difference to estimate motion.
        
        Args:
            frame_t: (B, C, H, W)
            frame_t1: (B, C, H, W)
        Returns:
            flow: (B, 2, H, W) - estimated (dx, dy) motion field
        """
        if frame_t.shape[1] > 1:
            gray_t = frame_t.mean(dim=1, keepdim=True)
            gray_t1 = frame_t1.mean(dim=1, keepdim=True)
        else:
            gray_t = frame_t
            gray_t1 = frame_t1

        It = gray_t1 - gray_t

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               device=frame_t.device, dtype=frame_t.dtype).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               device=frame_t.device, dtype=frame_t.dtype).view(1, 1, 3, 3)

        Ix = F.conv2d(gray_t, sobel_x, padding=1)
        Iy = F.conv2d(gray_t, sobel_y, padding=1)

        denom = Ix**2 + Iy**2 + 1e-6
        u = -It * Ix / denom
        v = -It * Iy / denom

        u = torch.clamp(u, -5, 5)
        v = torch.clamp(v, -5, 5)

        flow = torch.cat([u, v], dim=1)
        return flow

    def _warp_mask(
        self,
        mask: torch.Tensor,
        flow: torch.Tensor
    ) -> torch.Tensor:
        """
        Warp mask using optical flow via grid_sample.
        
        Args:
            mask: (B, 1, H, W)
            flow: (B, 2, H, W)
        Returns:
            warped: (B, 1, H, W)
        """
        B, _, H, W = mask.shape

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=mask.device),
            torch.linspace(-1, 1, W, device=mask.device),
            indexing='ij'
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

        flow_normalized = flow.permute(0, 2, 3, 1)
        flow_normalized = flow_normalized * 2.0 / torch.tensor([W, H], device=mask.device).float()

        sample_grid = base_grid + flow_normalized
        sample_grid = sample_grid.clamp(-1, 1)

        warped = F.grid_sample(mask, sample_grid, mode='bilinear', padding_mode='border', align_corners=True)
        return warped

    def forward(
        self,
        pred_masks: torch.Tensor,
        frames: torch.Tensor,
        frame_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            pred_masks: (B, T, 1, H, W) - sigmoid probabilities
            frames: (B, C, T, H, W) or (B, T, C, H, W) - input video
            frame_mask: (B, T) - optional, 1.0 for labeled frames (exclude from cycle loss)
        Returns:
            loss: scalar
        """
        if frames.dim() == 5 and frames.shape[1] != pred_masks.shape[1]:
            frames = frames.permute(0, 2, 1, 3, 4)

        B, T, C_mask, H, W = pred_masks.shape

        if T < 2:
            return torch.tensor(0.0, device=pred_masks.device)

        total_loss = 0.0
        count = 0

        for t in range(T - 1):
            mask_t = pred_masks[:, t]
            mask_t1 = pred_masks[:, t + 1]

            frame_t = frames[:, t]
            frame_t1 = frames[:, t + 1]

            flow = self._compute_flow_from_diff(frame_t, frame_t1)
            warped_mask = self._warp_mask(mask_t, flow)

            diff = (warped_mask - mask_t1) ** 2

            if frame_mask is not None:
                unlabeled_t = (frame_mask[:, t] < 0.5) & (frame_mask[:, t + 1] < 0.5)
                unlabeled_t = unlabeled_t.view(B, 1, 1, 1).float()
                diff = diff * unlabeled_t

            total_loss = total_loss + diff.mean()
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=pred_masks.device)

        return self.lambda_cycle * (total_loss / count)
