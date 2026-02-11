import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from src.registry import register_model
from src.utils.logging import get_logger
from src.models.segment_tracker import SegmentationDecoder

logger = get_logger()


@register_model("temporal_segment_tracker")
class TemporalEchoSegmentTracker(nn.Module):
    """
    Late Temporal Modeling for echocardiogram segmentation.

    Architecture:
        - Backbone with spatial features preserved (no global pooling)
        - Path A (Segmentation): frame-wise, adapter -> pool -> decoder
        - Path B (Regression): sequence-wise, adapter -> pool -> GRU -> regressor
    """

    @classmethod
    def from_config(cls, cfg):
        model_cfg = cfg.get('model', {})
        return cls(
            backbone=model_cfg.get('backbone', 'convnext_tiny'),
            hidden_dim=model_cfg.get('hidden_dim', 256),
            output_size=model_cfg.get('output_size', 112),
            pretrained=model_cfg.get('pretrained', True),
        )

    def __init__(
        self,
        backbone: str = 'convnext_tiny',
        hidden_dim: int = 256,
        output_size: int = 112,
        pretrained: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool='',
        )
        self.feature_dim = self.encoder.num_features

        # Channel adapter: feature_dim -> hidden_dim
        self.adapter = nn.Conv2d(self.feature_dim, hidden_dim, kernel_size=1)

        # Path A: Segmentation (frame-wise, spatial map -> decoder)
        self.decoder = SegmentationDecoder(
            hidden_dim=hidden_dim, output_size=output_size
        )

        # Path B: Volume Regression (temporal sequence)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
        )
        self.volume_regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # [Generic Vol 1, Generic Vol 2]
        )

        logger.info(
            "TemporalEchoSegmentTracker initialized: "
            f"backbone={backbone}, hidden_dim={hidden_dim}, "
            f"feature_dim={self.feature_dim}"
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None, **kwargs):
        """
        Args:
            x: (B, C, T, H, W) padded input video
            lengths: (B,) actual frame counts per sample
        Returns:
            dict with mask_logits, pred_edv, pred_esv, pred_ef
        """
        B, C, T, H, W = x.shape

        # Fold time into batch for spatial encoding
        x_flat = x.transpose(1, 2).reshape(B * T, C, H, W)

        # Backbone → spatial feature maps (B*T, feature_dim, H', W')
        features = self.encoder(x_flat)
        features = self.adapter(features)  # (B*T, hidden_dim, H', W')

        # ----- Path A: Segmentation (spatial features -> decoder) -----
        # Pass spatial features directly to decoder
        mask_logits = self.decoder(features)  # (B*T, 1, H_out, W_out)

        # ----- Path B: Mask-Guided Pooling for Regression -----
        # 1. Downsample mask logits to feature size
        # features shape: (N, C, H', W')
        feat_h, feat_w = features.shape[-2:]
        
        # Detach gradient for mask guidance?? 
        # Plan says: "Wrap this block in torch.set_grad_enabled(True)... so that gradients from the regression loss flow back through the mask"
        # So we do NOT detach.
        
        mask_down = F.interpolate(
            mask_logits, size=(feat_h, feat_w), mode='bilinear', align_corners=False
        )
        mask_prob = torch.sigmoid(mask_down) # (N, 1, H', W')
        
        # 2. Weighted Average
        # (N, C, H', W') * (N, 1, H', W') -> sum spatial dims
        numerator = (features * mask_prob).sum(dim=(-2, -1)) # (N, C)
        denominator = mask_prob.sum(dim=(-2, -1)) + 1e-6 # (N, 1)
        
        feat_vec = numerator / denominator # (N, C) aka (B*T, hidden_dim)

        # Reshape for Temporal Model
        mask_logits = mask_logits.view(
            B, T, 1, mask_logits.shape[-2], mask_logits.shape[-1]
        ).transpose(1, 2) # (B, 1, T, H_out, W_out)

        feat_vec = feat_vec.view(B, T, -1)  # (B, T, hidden_dim)

        if lengths is not None:
            lengths_cpu = lengths.to('cpu').int()
            packed = nn.utils.rnn.pack_padded_sequence(
                feat_vec, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.gru(packed)
            gru_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=T
            )

            # Average only valid frames
            mask_t = torch.arange(T, device=x.device)[None, :] < lengths[:, None]
            mask_t = mask_t.unsqueeze(-1).float()
            video_embedding = (
                (gru_out * mask_t).sum(dim=1)
                / mask_t.sum(dim=1).clamp(min=1e-6)
            )
        else:
            gru_out, _ = self.gru(feat_vec)
            video_embedding = gru_out.mean(dim=1)

        # ----- Path C: Sorted Volume Regression -----
        pred_vols = F.softplus(self.volume_regressor(video_embedding)) # (B, 2)
        
        # Sort: max -> EDV, min -> ESV
        pred_edv, _ = pred_vols.max(dim=1, keepdim=True)
        pred_esv, _ = pred_vols.min(dim=1, keepdim=True)

        return {
            "mask_logits": mask_logits,
            "hidden_features": feat_vec, # This is now the pooled features
            "pred_edv": pred_edv,
            "pred_esv": pred_esv,
            "pred_ef": (
                (pred_edv - pred_esv)
                / (pred_edv + 1e-6)
            ),
        }
