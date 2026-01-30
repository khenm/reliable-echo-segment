import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from src.registry import register_model

class AreaLengthVolumeCalculator(nn.Module):
    """
    Computes Volume using the Area-Length method:
    V = (8 * Area^2) / (3 * pi * Length)
    """
    def __init__(self):
        super().__init__()

    def forward(self, length, widths):
        """
        Args:
            length: (B, T)
            widths: (B, T, 20)
        Returns:
            volume: (B, T)
        """
        # Calculate Area via Riemann Sum (Mean width * Length)
        total_width = torch.sum(widths, dim=2)
        slice_height = length / widths.shape[2]
        area = total_width * slice_height
        volume = (8.0 * (area ** 2)) / (3.0 * 3.14159 * (length + 1e-6))
        return volume

class SoftArgmax2D(nn.Module):
    """
    Differentiable Soft-Argmax to convert Heatmaps -> Coordinates.
    """
    def __init__(self, height=56, width=56, normalized_coordinates=True):
        super().__init__()
        self.height = height
        self.width = width
        self.normalized_coordinates = normalized_coordinates
        
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(0, 1, height),
            torch.linspace(0, 1, width),
            indexing='ij'
        )
        self.register_buffer('pos_x', pos_x.unsqueeze(0).unsqueeze(0).clone()) 
        self.register_buffer('pos_y', pos_y.unsqueeze(0).unsqueeze(0).clone())

    def forward(self, heatmaps):
        """
        Args:
            heatmaps: (B, T, K, H, W)
        Returns:
            coords: (B, T, K, 2) in (x, y) order
        """
        B, T, K, H, W = heatmaps.shape
        heatmaps = heatmaps.view(B, T, K, -1)
        heatmaps = F.softmax(heatmaps, dim=-1)
        heatmaps = heatmaps.view(B, T, K, H, W)
        
        expected_y = torch.sum(heatmaps * self.pos_x, dim=[3, 4])
        expected_x = torch.sum(heatmaps * self.pos_y, dim=[3, 4])
        
        return torch.stack([expected_x, expected_y], dim=-1)

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
        
        self.encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.feature_dim = self.encoder.num_features
        
        # Temporal Injection: Mix (t-1, t, t+1) features
        self.temporal_injector = nn.Sequential(
            nn.Conv1d(self.feature_dim, self.feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.feature_dim),
            nn.ReLU(),
            nn.Conv1d(self.feature_dim, self.feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.feature_dim),
            nn.ReLU()
        )
        
        self.temporal = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        self.pre_deconv = nn.Linear(hidden_dim, 64 * 7 * 7)
        
        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1), # 14x14
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # 28x28
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, num_points, kernel_size=4, stride=2, padding=1), # 56x56
        )
        
        self.soft_argmax = SoftArgmax2D(height=56, width=56)
        self.calculator = AreaLengthVolumeCalculator()
        
    def forward(self, x, lengths=None):
        B, C, T, H, W = x.shape
        
        # Encode Spatial
        x_flat = x.transpose(1, 2).reshape(B * T, C, H, W)
        features = self.encoder(x_flat) 
        features = features.view(B, T, -1)
        
        # Inject Temporal Consistency
        features_t = features.permute(0, 2, 1)
        features_t = self.temporal_injector(features_t)
        features = features_t.permute(0, 2, 1)
        
        # Track (GRU)
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
            
        # Predict Heatmaps -> Coordinates
        spatial_seed = self.pre_deconv(hidden_states.reshape(B*T, -1))
        spatial_seed = spatial_seed.view(B*T, 64, 7, 7)
        
        heatmaps = self.heatmap_head(spatial_seed)
        heatmaps = heatmaps.view(B, T, -1, 56, 56)
        
        # Get Coordinates [-1, 1]
        kps = self.soft_argmax(heatmaps)
        
        long_axis, slice_widths = self._derive_volume_metrics(kps)
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
        
        return kps, volume, ef

    def _derive_volume_metrics(self, kps):
        """
        Derives Long Axis Length and Slice Widths from the 42 keypoints.
        """
        base = kps[:, :, 0]
        apex = kps[:, :, 21]
        
        long_axis = torch.norm(apex - base, dim=2)
        
        left_pts = kps[:, :, 1:21]
        right_pts = kps[:, :, 22:42] 
        
        # Flip right points to match left (so 41 aligns with 1)
        right_pts_flipped = torch.flip(right_pts, dims=[2])
        
        slice_widths = torch.norm(left_pts - right_pts_flipped, dim=-1)
        
        return long_axis, slice_widths