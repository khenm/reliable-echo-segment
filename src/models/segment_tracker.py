import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from src.registry import register_model
from src.utils.logging import get_logger

logger = get_logger()


class MaskSlicer(nn.Module):
    """
    Differentiable Method of Disks for extracting volume metrics from masks.
    Takes soft binary masks and calculates slice widths for Simpson's Rule.
    """
    def __init__(self, num_slices=20, threshold=0.5):
        super().__init__()
        self.num_slices = num_slices
        self.threshold = threshold

    def forward(self, masks):
        """
        Args:
            masks: (B, T, 1, H, W) - Sigmoid probabilities
        Returns:
            length: (B, T) - Long axis length in pixels
            widths: (B, T, num_slices) - Widths at slice points
        """
        B, T, C, H, W = masks.shape
        masks = masks.squeeze(2)

        y_coords = torch.linspace(0, H - 1, H, device=masks.device)
        x_coords = torch.linspace(0, W - 1, W, device=masks.device)

        mask_y_proj = masks.sum(dim=-1)
        weighted_y = mask_y_proj * y_coords.view(1, 1, -1)
        total_mass = mask_y_proj.sum(dim=-1, keepdim=True) + 1e-6

        soft_y_weights = F.softmax(mask_y_proj * 10, dim=-1)
        y_min_soft = (soft_y_weights * y_coords.view(1, 1, -1)).sum(dim=-1)

        inv_weights = F.softmax(-mask_y_proj * 10, dim=-1)
        y_max_soft = H - 1 - (inv_weights * (H - 1 - y_coords).view(1, 1, -1)).sum(dim=-1)

        length = (y_max_soft - y_min_soft).clamp(min=1.0)

        masks_for_slicing = masks.view(B * T, H, W).unsqueeze(1)
        masks_resized = F.interpolate(
            masks_for_slicing,
            size=(self.num_slices, W),
            mode='bilinear',
            align_corners=False
        )
        masks_resized = masks_resized.squeeze(1).view(B, T, self.num_slices, W)

        widths = masks_resized.sum(dim=-1)

        return length, widths


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
        total_width = torch.sum(widths, dim=2)
        slice_height = length / widths.shape[2]
        area = total_width * slice_height
        volume = (8.0 * (area ** 2)) / (3.0 * 3.14159 * (length + 1e-6))
        return volume


class SegmentationDecoder(nn.Module):
    """
    U-Net style decoder that upsamples from hidden states to full resolution masks.
    Upsampling path: 7x7 -> 14x14 -> 28x28 -> 56x56 -> 112x112
    """
    def __init__(self, hidden_dim=256, output_size=112, spatial_input=False):
        super().__init__()
        self.output_size = output_size
        self.spatial_input = spatial_input

        if self.spatial_input:
            self.proj = nn.Conv2d(hidden_dim, 64, kernel_size=1)
        else:
            self.pre_deconv = nn.Linear(hidden_dim, 64 * 7 * 7)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.Conv2d(8, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        """
        Args:
            x: (B*T, hidden_dim) [2D] OR (B*T, hidden_dim, H, W) [4D]
        Returns:
            masks: (B*T, 1, output_size, output_size) logits
        """
        if self.spatial_input:
            x = self.proj(x)
        else:
            x = self.pre_deconv(x)
            x = x.view(-1, 64, 7, 7)
        
        x = self.decoder(x)

        if x.shape[-1] != self.output_size:
            x = F.interpolate(x, size=(self.output_size, self.output_size), mode='bilinear', align_corners=False)

        return x


@register_model("segment_tracker")
class EchoSegmentTracker(nn.Module):
    """
    Temporal Segmentation model for echocardiogram analysis.
    Predicts binary masks instead of keypoints, then derives volume via MaskSlicer.
    """
    @classmethod
    def from_config(cls, cfg):
        model_cfg = cfg.get('model', {})
        return cls(
            backbone=model_cfg.get('backbone', 'convnext_tiny'),
            hidden_dim=model_cfg.get('hidden_dim', 256),
            output_size=model_cfg.get('output_size', 112),
            pretrained=model_cfg.get('pretrained', True)
        )

    def __init__(self, backbone='convnext_tiny', hidden_dim=256, output_size=112, pretrained=True):
        super().__init__()

        self.encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.feature_dim = self.encoder.num_features

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

        self.decoder = SegmentationDecoder(hidden_dim=hidden_dim, output_size=output_size)
        self.slicer = MaskSlicer(num_slices=20)
        self.calculator = AreaLengthVolumeCalculator()

        logger.info(f"EchoSegmentTracker initialized: backbone={backbone}, hidden_dim={hidden_dim}, output_size={output_size}")

    def forward(self, x, lengths=None):
        """
        Args:
            x: (B, C, T, H, W) input video
            lengths: (B,) actual frame counts for variable-length sequences
        Returns:
            masks: (B, 1, T, H, W) sigmoid probabilities
            volume: (B, T) volume per frame
            ef: (B, 1) ejection fraction
        """
        B, C, T, H, W = x.shape

        x_flat = x.transpose(1, 2).reshape(B * T, C, H, W)
        features = self.encoder(x_flat)
        features = features.view(B, T, -1)

        features_t = features.permute(0, 2, 1)
        features_t = self.temporal_injector(features_t)
        features = features_t.permute(0, 2, 1)

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

        mask_logits = self.decoder(hidden_states.reshape(B * T, -1))
        mask_logits = mask_logits.view(B, T, 1, mask_logits.shape[-2], mask_logits.shape[-1])
        mask_logits = mask_logits.permute(0, 2, 1, 3, 4)

        masks = torch.sigmoid(mask_logits)

        masks_for_slicer = masks.permute(0, 2, 1, 3, 4)
        long_axis, slice_widths = self.slicer(masks_for_slicer)
        volume = self.calculator(long_axis, slice_widths)

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

        return mask_logits, volume, ef
