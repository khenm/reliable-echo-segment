import torch
import torch.nn as nn
import timm
from src.registry import register_model

class PointSimpsonCalculator(nn.Module):
    """
    Computes Left Ventricle (LV) Volume using Method of Disks (Simpson's Rule) on coordinate points.
    Assumes points are normalized [0, 1].
    """
    def __init__(self, num_slices=20):
        super().__init__()
        self.num_slices = num_slices

    def forward(self, points):
        """
        Args:
            points: (B, T, 42, 2)
        Returns:
            volume: (B, T)
        """
        B, T, N, _ = points.shape
        points = points.view(B * T, N, 2) # (BT, 42, 2)
        
        # Apex is average of 20 and 21
        apex = (points[:, 20, :] + points[:, 21, :]) / 2.0 # (BT, 2)
        
        # Base is average of 0 and 41
        base = (points[:, 0, :] + points[:, 41, :]) / 2.0 # (BT, 2)
        
        # Long Axis Vector (Base to Apex)
        long_axis = apex - base # (BT, 2)
        length = torch.norm(long_axis, dim=1) # (BT,)
        
        
        # Avoid div by zero
        safe_len = torch.clamp(length, min=1e-6)
        u_x = long_axis[:, 0] / safe_len
        u_y = long_axis[:, 1] / safe_len
        
        # Rotate points to frame where Base is (0,0) and Apex is (0, L)
        # Shift to base
        p_centered = points - base.unsqueeze(1) # (BT, 42, 2)
        
        # Rotation matrix components
        # cos_t = u_y, sin_t = -u_x (Implicitly used below)
        pos_axis = p_centered[:, :, 0] * u_x.unsqueeze(1) + p_centered[:, :, 1] * u_y.unsqueeze(1) # (BT, 42)
        dist_axis = -p_centered[:, :, 0] * u_y.unsqueeze(1) + p_centered[:, :, 1] * u_x.unsqueeze(1) # (BT, 42)
        
        # Now we have gathered (y, x) coords in the rectified system.
        # Left side (0-20): dist_axis should be negative?
        # Right side (21-41): dist_axis should be positive?
        # We take abs(dist_axis) as radius.
        
        radii = torch.abs(dist_axis)
        
        # Slice heights logic replaced by target_h
        # slice_heights = ... (Unused)
        # volumes = ... (Unused)
        
        
        # We will use a fixed number of slices relative to L (0 to 1 scale).
        target_h = torch.linspace(0.05, 0.95, self.num_slices, device=points.device).unsqueeze(0).repeat(B*T, 1) # (BT, 20)
        
        # Separate sides
        n_side = 21
        side1_h = pos_axis[:, :n_side] / safe_len.unsqueeze(1) # (BT, 21) normalized 0-1
        side1_r = radii[:, :n_side]
        
        # side2_h = pos_axis[:, 21:][:, ::-1] / safe_len.unsqueeze(1) 
        # use torch.flip for safety
        side2_h = torch.flip(pos_axis[:, 21:], dims=[1]) / safe_len.unsqueeze(1)
        side2_r = torch.flip(radii[:, 21:], dims=[1])
        
        # vector_interp function
        r1 = self.batch_interp(side1_h, side1_r, target_h) # (BT, 20)
        r2 = self.batch_interp(side2_h, side2_r, target_h) # (BT, 20)
        
        diameter = r1 + r2
        disk_areas = (3.14159 / 4.0) * (diameter ** 2)
        
        # Sum areas * slice_thickness
        # slice_thickness = Length / num_slices
        total_vol = torch.sum(disk_areas, dim=1) * (length / self.num_slices)
        
        return total_vol.view(B, T)

    def batch_interp(self, x, y, x_target):
        """
        Differentiable linear interpolation for batches.
        x: (B, N) sorted ascending (mostly)
        y: (B, N) values
        x_target: (B, M) query points
        """
        _, N = x.shape
        # M = x_target.shape[1]
        x, sort_idx = torch.sort(x, dim=1)
        # gather y
        y = torch.gather(y, 1, sort_idx)
        
        # Find indices
        # torch.searchsorted needs flat input or same specific shape? 
        # It supports batched since PyTorch 1.7
        idx = torch.searchsorted(x, x_target) # (B, M)
        
        # Clamp
        idx_low = torch.clamp(idx - 1, min=0, max=N-2)
        idx_high = idx_low + 1
        
        # Gather x and y
        x_low = torch.gather(x, 1, idx_low)
        x_high = torch.gather(x, 1, idx_high)
        y_low = torch.gather(y, 1, idx_low)
        y_high = torch.gather(y, 1, idx_high)
        
        # Interpolate
        w = (x_target - x_low) / (x_high - x_low + 1e-6)
        w = torch.clamp(w, 0.0, 1.0)
        
        y_interp = y_low + w * (y_high - y_low)
        return y_interp

@register_model("skeletal_tracker")
class SkeletalTracker(nn.Module):
    def __init__(self, backbone='convnext_tiny', num_points=42, hidden_dim=256, pretrained=True):
        super().__init__()
        
        # 1. Spatial Encoder
        if 'convnext' in backbone or 'resnet' in backbone:
            self.encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
            self.feature_dim = self.encoder.num_features
        else:
            raise ValueError(f"Backbone {backbone} not supported.")
            
        # 2. Temporal Tracker
        self.temporal = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=False # Online tracking usually causal? Re-read: "Temporal dynamics". "Martingales in real-time" -> Causal (unidirectional).
        )
        
        # 3. Heads
        self.keypoint_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_points * 2),
            nn.Sigmoid() # Normalize to 0-1
        )
        
        self.simpson = PointSimpsonCalculator()
        
    @classmethod
    def from_config(cls, cfg):
        return cls(
            backbone=cfg['model'].get('backbone', 'convnext_tiny'),
            num_points=cfg['model'].get('num_points', 42),
            hidden_dim=cfg['model'].get('hidden_dim', 256),
            pretrained=cfg['model'].get('pretrained', True)
        )

    def forward(self, x):
        """
        x: (B, C, T, H, W)
        """
        if x.ndim == 5:
            B, C, T, H, W = x.shape
            x = x.transpose(1, 2).reshape(B * T, C, H, W)
        else:
            # Handle 4D input (B, C, H, W) treated as T=1
            B, C, H, W = x.shape
            T = 1
        
        # Encoder
        features = self.encoder(x) # (BT, D)
        features = features.view(B, T, -1) # (B, T, D)
        
        # Temporal
        # hidden state initialization defaults to 0
        hidden_states, _ = self.temporal(features) # (B, T, H)
        
        # Heads
        kps_flat = self.keypoint_head(hidden_states) # (B, T, 42*2)
        kps = kps_flat.view(B, T, 42, 2)
        
        # Volume
        volume = self.simpson(kps) # (B, T)
        
        # EF Calculation (approximate from batch max/min)
        # Note: This is per-sample EF
        edv, _ = volume.max(dim=1, keepdim=True)
        esv, _ = volume.min(dim=1, keepdim=True)
        ef = (edv - esv) / (edv + 1e-6) # (B, 1)
        
        return kps, volume, ef
