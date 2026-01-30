import torch
import torch.nn as nn

class TopologyLoss(nn.Module):
    """
    Enforces geometric constraints on the skeletal structure (Spring & Collapse prevention).
    """
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, pred_kps):
        """
        pred_kps: (B, T, 42, 2)
        """
        # Construct full left chain tensor
        chain_left = pred_kps[:, :, 0:22] # 0..21
        diffs_left = chain_left[:, :, 1:] - chain_left[:, :, :-1]
        dists_left = torch.norm(diffs_left, dim=-1)
        
        # Variance of distances (Spring Consitency)
        mean_dist_left = dists_left.mean(dim=2, keepdim=True)
        spring_loss_left = ((dists_left - mean_dist_left) ** 2).mean()

        # Let's grab 22..41 and flip it to be Base->Apex order: 41, 40... 22
        right_wall_base_to_apex = torch.flip(pred_kps[:, :, 22:42], dims=[2])
        
        # Full chain: Base(0) -> RightWall(Base->Apex) -> Apex(21)
        chain_right = torch.cat([
            pred_kps[:, :, 0:1],          # Base
            right_wall_base_to_apex,      # 41..22
            pred_kps[:, :, 21:22]         # Apex
        ], dim=2)
        
        diffs_right = chain_right[:, :, 1:] - chain_right[:, :, :-1]
        dists_right = torch.norm(diffs_right, dim=-1)
        
        mean_dist_right = dists_right.mean(dim=2, keepdim=True)
        spring_loss_right = ((dists_right - mean_dist_right) ** 2).mean()
        
        left_pts = pred_kps[:, :, 1:21]   # 1..20
        right_pts = pred_kps[:, :, 22:42] # 22..41 (Apex->Base). Wait.
        
        right_pts_matched = torch.flip(right_pts, dims=[2]) # 41..22
        
        widths = torch.norm(left_pts - right_pts_matched, dim=-1)
        
        # Penalize if width < Threshold (e.g. 0.05 normalized units)
        # Coordinates are in [-1, 1]. 0.05 is 2.5% of range.
        collapse_loss = torch.nn.functional.relu(0.05 - widths).mean()

        return self.weight * (spring_loss_left + spring_loss_right + collapse_loss * 10.0)
