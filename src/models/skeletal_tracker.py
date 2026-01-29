import torch
import torch.nn as nn
import timm
from src.registry import register_model

class AreaLengthVolumeCalculator(nn.Module):
    """
    Computes Volume using the Area-Length method.
    V = (8 * Area^2) / (3 * pi * Length)
    This is less sensitive to pixel calibration noise than summation.
    """
    def __init__(self):
        super().__init__()

    def forward(self, lengths, widths):
        """
        Args:
            lengths: (B, T) - Length of the Long Axis
            widths: (B, T, 20) - Widths of the ventricle at 20 slices
        """
        # 1. Calculate Area (Riemann Sum of widths)
        # Assuming slices are equally spaced along the length
        # slice_height = Length / 20
        slice_height = lengths / widths.shape[2] 
        
        # Area = Sum(width_i * slice_height)
        total_width = torch.sum(widths, dim=2) # (B, T)
        area = total_width * slice_height # (B, T)
        
        # 2. Area-Length Formula
        # V = (8 * A^2) / (3 * pi * L)
        # Add epsilon to L to prevent div by zero
        volume = (8.0 * (area ** 2)) / (3.0 * 3.14159 * (lengths + 1e-6))
        
        return volume

@register_model("skeletal_tracker")
class SkeletalTracker(nn.Module):
    @classmethod
    def from_config(cls, cfg):
        model_cfg = cfg.get('model', {})
        return cls(
            backbone=model_cfg.get('backbone', 'convnext_tiny'),
            num_points=model_cfg.get('num_points', 42),
            hidden_dim=model_cfg.get('hidden_dim', 256),
            pretrained=model_cfg.get('pretrained', True)
        )

    def __init__(self, backbone='convnext_tiny', num_points=42, hidden_dim=256, pretrained=True):
        super().__init__()
        
        # 1. Spatial Encoder
        self.encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.feature_dim = self.encoder.num_features
        
        # 2. Temporal Tracker
        self.temporal = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        # 3. Heads (The Tweak: Topology-Preserving)
        # Head A: The Axis (Base Center (x,y), Apex (x,y)) -> 4 outputs
        self.head_axis = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4), 
            nn.Sigmoid() # Normalized 0-1
        )
        
        # Head B: The Widths (10 slices * 2 sides = 20 widths)
        # We predict the *radius* from the axis to the wall
        self.head_widths = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 20),
            nn.Sigmoid() 
        )
        
        self.calculator = AreaLengthVolumeCalculator()
        
    def forward(self, x, lengths=None):
        B, C, T, H, W = x.shape
        
        # 1. Encode
        x_flat = x.transpose(1, 2).reshape(B * T, C, H, W)
        features = self.encoder(x_flat) 
        features = features.view(B, T, -1) 
        
        # 2. Track
        if lengths is not None:
            lengths_cpu = lengths.to('cpu').int()
            packed_feat = nn.utils.rnn.pack_padded_sequence(
                features, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.temporal(packed_feat)
            hidden_states, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=T
            )
        else:
            hidden_states, _ = self.temporal(features)
            
        # 3. Predict Geometry
        # Axis: (B, T, 4) -> [BaseX, BaseY, ApexX, ApexY]
        axis = self.head_axis(hidden_states)
        base = axis[:, :, :2]
        apex = axis[:, :, 2:]
        
        # Widths: (B, T, 20)
        # These are perpendicular distances from axis to wall
        radii = self.head_widths(hidden_states) 
        
        # 4. Calculate Physics
        # Long Axis Length
        long_axis_vec = apex - base
        L_t = torch.norm(long_axis_vec, dim=2) # (B, T)
        
        # Volume (Area-Length Method)
        # Note: We combine left/right radii into total slice widths
        # Assuming radii output is [r_l1, ... r_l10, r_r1, ... r_r10]
        # We pair them: w_1 = r_l1 + r_r1
        left_radii = radii[:, :, :10]
        right_radii = radii[:, :, 10:]
        slice_widths = left_radii + right_radii # (B, T, 10)
        
        volume = self.calculator(L_t, slice_widths) # (B, T)
        
        # 5. Reconstruct 42 Points (For Visualization & Loss)
        # We mathematically plot the points so we can still use your SkeletalLoss!
        kps = self._reconstruct_points(base, apex, left_radii, right_radii)
        
        # 6. EF Calculation (Masked)
        if lengths is not None:
            range_tensor = torch.arange(T, device=x.device).unsqueeze(0)
            mask = range_tensor < lengths.unsqueeze(1)
            
            vol_max = volume.clone()
            vol_min = volume.clone()
            vol_max[~mask] = -1e9
            vol_min[~mask] = 1e9
            
            edv, _ = vol_max.max(dim=1, keepdim=True)
            esv, _ = vol_min.min(dim=1, keepdim=True)
        else:
            edv, _ = volume.max(dim=1, keepdim=True)
            esv, _ = volume.min(dim=1, keepdim=True)

        ef = (edv - esv) / (edv + 1e-6)
        
        return kps, volume, ef

    def _reconstruct_points(self, base, apex, left_radii, right_radii):
        """
        Reconstructs the 42 (x,y) coordinates from Axis + Radii.
        This allows us to still supervise with the ground truth points.
        """
        B, T, _ = base.shape
        
        # Direction Vector
        v = apex - base # (B, T, 2)
        length = torch.norm(v, dim=2, keepdim=True) + 1e-6
        u = v / length # Unit vector pointing up
        
        # Perpendicular Vector (Rotate -90 deg)
        # (x, y) -> (y, -x)
        u_perp = torch.stack([u[:, :, 1], -u[:, :, 0]], dim=2) # (B, T, 2)
        
        # Generate 10 slice positions along the axis (0.05 to 0.95)
        # We have 10 radii per side
        num_slices = 10
        t_vals = torch.linspace(0.05, 0.95, num_slices, device=base.device)
        
        left_radii_20 = torch.nn.functional.interpolate(left_radii.view(B*T, 1, 10), size=20, mode='linear', align_corners=True).view(B, T, 20)
        right_radii_20 = torch.nn.functional.interpolate(right_radii.view(B*T, 1, 10), size=20, mode='linear', align_corners=True).view(B, T, 20)
        
        t_vals_20 = torch.linspace(0.05, 0.95, 20, device=base.device).view(1, 1, 20, 1)
        
        # Calculate centers of slices
        # Center_i = Base + t_i * (Apex - Base)
        # (B, T, 1, 2) + (1, 1, 20, 1) * (B, T, 1, 2) -> (B, T, 20, 2)
        centers = base.unsqueeze(2) + t_vals_20 * v.unsqueeze(2)
        
        # Calculate Left Points: Center + Radius * Perp
        # (B, T, 20, 2) + (B, T, 20, 1) * (B, T, 1, 2)
        p_left = centers + left_radii_20.unsqueeze(3) * u_perp.unsqueeze(2)
        
        # Calculate Right Points: Center - Radius * Perp
        p_right = centers - right_radii_20.unsqueeze(3) * u_perp.unsqueeze(2)
        
        # Apex (Index 20, 21): Just use the Apex coordinate
        p_apex = apex.unsqueeze(2) # (B, T, 1, 2)
        
        # Base (Index 0, 41): Just use Base coordinate? 
        # Ideally, we should predict base width. But using Base Center is fine for now.
        p_base = base.unsqueeze(2)
        
        
        # Let's construct:
        kps_list = [p_base] # 0
        kps_list.append(p_left) # 1..20
        kps_list.append(p_apex) # 21 (Use apex for both 20/21 junction?)
        
        # This reconstruction effectively "projects" the learned shape onto the 42-point skeleton.
        pts_left = torch.cat([p_left, p_apex], dim=2) # 21 points
        pts_right = torch.cat([p_apex, torch.flip(p_right, [2])], dim=2) # 21 points
        
        # This gives 42 points.
        kps = torch.cat([pts_left, pts_right], dim=2)
        
        return kps
