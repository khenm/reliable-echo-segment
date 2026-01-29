import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from src.registry import register_model

class AreaLengthVolumeCalculator(nn.Module):
    """
    Computes Volume using the Area-Length method.
    V = (8 * Area^2) / (3 * pi * Length)
    """
    def __init__(self):
        super().__init__()

    def forward(self, length, widths):
        # widths: (B, T, 20)
        # length: (B, T)
        
        # 1. Calculate Area (Mean width * Length)
        # We sum the widths and multiply by slice height (L / 20)
        # This is Riemann Sum integration.
        total_width = torch.sum(widths, dim=2) # (B, T)
        slice_height = length / widths.shape[2]
        area = total_width * slice_height # (B, T)
        
        # 2. Bullet Formula
        volume = (8.0 * (area ** 2)) / (3.0 * 3.14159 * (length + 1e-6))
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
        
        # 1. Spatial Encoder (2D)
        self.encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.feature_dim = self.encoder.num_features
        
        # 2. Temporal Injection (The Fix!)
        # A 1D Conv that mixes (t-1, t, t+1) features to smooth jitter
        self.temporal_injector = nn.Sequential(
            nn.Conv1d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.feature_dim),
            nn.ReLU(),
            # Optional: Add a second layer for stronger mixing
            nn.Conv1d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.feature_dim),
            nn.ReLU()
        )
        
        # 3. Global Tracker (GRU)
        self.temporal = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        # 4. Heads (Axis + Widths)
        self.head_axis = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 4), nn.Sigmoid() 
        )
        self.head_widths = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 20), nn.Sigmoid() 
        )
        
        self.calculator = AreaLengthVolumeCalculator()
        
    def forward(self, x, lengths=None):
        B, C, T, H, W = x.shape
        
        # 1. Encode Spatial
        x_flat = x.transpose(1, 2).reshape(B * T, C, H, W)
        features = self.encoder(x_flat) 
        features = features.view(B, T, -1) # (B, T, D)
        
        # 2. Inject Temporal Consistency
        # Conv1d expects (Batch, Channels, Time)
        features_t = features.permute(0, 2, 1) # (B, D, T)
        features_t = self.temporal_injector(features_t)
        features = features_t.permute(0, 2, 1) # (B, T, D)
        
        # 3. Track (GRU)
        if lengths is not None:
            lengths_cpu = lengths.to('cpu').int()
            packed_feat = nn.utils.rnn.pack_padded_sequence(features, lengths_cpu, batch_first=True, enforce_sorted=False)
            packed_out, _ = self.temporal(packed_feat)
            hidden_states, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=T)
        else:
            hidden_states, _ = self.temporal(features)
            
        # 4. Predict Geometry
        axis = self.head_axis(hidden_states)
        base, apex = axis[:, :, :2], axis[:, :, 2:]
        radii = self.head_widths(hidden_states)
        
        # 5. Volume & EF
        long_axis = torch.norm(apex - base, dim=2)
        slice_widths = radii[:, :, :10] + radii[:, :, 10:]
        # Upsample widths to 20 slices for calculation precision
        slice_widths = F.interpolate(slice_widths.view(B*T, 1, 10), size=20, mode='linear', align_corners=True).view(B, T, 20)
        
        volume = self.calculator(long_axis, slice_widths)
        
        # Masked Min/Max for EF
        if lengths is not None:
            range_tensor = torch.arange(T, device=x.device).unsqueeze(0)
            mask = range_tensor < lengths.unsqueeze(1)
            vol_max, vol_min = volume.clone(), volume.clone()
            vol_max[~mask] = -1e9
            vol_min[~mask] = 1e9
            edv, _ = vol_max.max(dim=1, keepdim=True)
            esv, _ = vol_min.min(dim=1, keepdim=True)
        else:
            edv, _ = volume.max(dim=1, keepdim=True)
            esv, _ = volume.min(dim=1, keepdim=True)

        ef = (edv - esv) / (edv + 1e-6)
        
        # Reconstruct points for visualization/supervision
        # (Assuming _reconstruct_points logic exists from previous snippet)
        kps = self._reconstruct_points(base, apex, radii[:,:,:10], radii[:,:,10:])
        
        return kps, volume, ef

    def _reconstruct_points(self, base, apex, left, right):
        # ... (Identical to previous implementation) ...
        # (Re-paste the reconstruction logic here if needed)
        # Simplified for brevity:
        B, T, _ = base.shape
        v = apex - base
        length = torch.norm(v, dim=2, keepdim=True) + 1e-6
        u = v / length
        u_perp = torch.stack([u[:,:,1], -u[:,:,0]], dim=2)
        
        # Upsample 10 -> 20
        l_20 = F.interpolate(left.view(B*T, 1, 10), size=20, mode='linear', align_corners=True).view(B, T, 20, 1)
        r_20 = F.interpolate(right.view(B*T, 1, 10), size=20, mode='linear', align_corners=True).view(B, T, 20, 1)
        
        t_vals = torch.linspace(0.05, 0.95, 20, device=base.device).view(1, 1, 20, 1)
        centers = base.unsqueeze(2) + t_vals * v.unsqueeze(2)
        
        p_left = centers + l_20 * u_perp.unsqueeze(2)
        p_right = centers - r_20 * u_perp.unsqueeze(2)
        
        kps = torch.cat([base.unsqueeze(2), p_left, apex.unsqueeze(2), torch.flip(p_right, [2])], dim=2)
        return kps
