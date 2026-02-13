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
    Late Temporal Modeling for echocardiogram segmentation and volume regression.

    Architecture:
        - Backbone: Spatial features preserved (no global pooling).
        - Path A (Segmentation): Frame-wise adapter -> SegmentationDecoder.
        - Path B (Regression): Sequence-wise adapter -> Mask-Guided Pooling -> GRU -> Regressor.
    """

    @classmethod
    def from_config(cls, cfg: dict) -> "TemporalEchoSegmentTracker":
        """Creates an instance from a configuration dictionary."""
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
        """
        Initializes the TemporalEchoSegmentTracker.

        Args:
            backbone: Name of the timm backbone model.
            hidden_dim: Dimension of the hidden layer.
            output_size: Size of the output spatial dimensions.
            pretrained: Whether to use pretrained backbone weights.
            **kwargs: Additional arguments.
        """
        super().__init__()

        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool='',
        )
        self.feature_dim = self.encoder.num_features
        self.adapter = nn.Conv2d(self.feature_dim, hidden_dim, kernel_size=1)

        # Path A: Segmentation
        self.decoder = SegmentationDecoder(
            hidden_dim=hidden_dim, output_size=output_size, spatial_input=True
        )

        # Path B: Volume Regression
        self.gru = nn.GRU(
            input_size=hidden_dim + 1,  # +1 for Area feature
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
            f"TemporalEchoSegmentTracker initialized: backbone={backbone}, "
            f"hidden_dim={hidden_dim}, feature_dim={self.feature_dim}"
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None, **kwargs) -> dict:
        """
        Forward pass of the model.

        Args:
            x: Input tensor of shape (B, C, T, H, W).
            lengths: Actual frame counts per sample of shape (B,).
            **kwargs: Additional arguments.

        Returns:
            Dictionary containing:
                - mask_logits: Segmentation masks (B, T, 1, H, W).
                - hidden_features: Pooled features used for regression.
                - pred_edv: Predicted End-Diastolic Volume.
                - pred_esv: Predicted End-Systolic Volume.
                - pred_ef: Predicted Ejection Fraction.
        """
        B, C, T, H, W = x.shape

        # Fold time into batch for spatial encoding
        x_flat = x.transpose(1, 2).reshape(B * T, C, H, W)

        # Backbone extraction
        features = self.encoder(x_flat)
        features = self.adapter(features)  # (B*T, hidden_dim, H', W')

        # Path A: Segmentation
        mask_logits = self.decoder(features)  # (B*T, 1, H_out, W_out)

        # Path B: Regression with Gradient Shield and Area-Aware Pooling
        feat_reg = features.detach()
        mask_reg = mask_logits.detach()

        feat_h, feat_w = feat_reg.shape[-2:]
        mask_down = F.interpolate(
            mask_reg, size=(feat_h, feat_w), mode='bilinear', align_corners=False
        )
        mask_prob = torch.sigmoid(mask_down)

        # Mask-Guided Pooling: Weighted Average (Texture) + Sum (Area)
        numerator = (feat_reg * mask_prob).sum(dim=(-2, -1))
        area = mask_prob.sum(dim=(-2, -1)) + 1e-6

        avg_feat = numerator / area
        area_feat = torch.log(area)

        # Concatenate features and reshape for GRU
        feat_vec = torch.cat([avg_feat, area_feat], dim=1)
        feat_vec = feat_vec.view(B, T, -1)

        # Reshape masks to (B, T, 1, H, W)
        mask_logits = mask_logits.view(
            B, T, 1, mask_logits.shape[-2], mask_logits.shape[-1]
        ).transpose(1, 2)

        # Temporal Modeling
        if lengths is not None:
            lengths_cpu = lengths.to('cpu').int()
            packed = nn.utils.rnn.pack_padded_sequence(
                feat_vec, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.gru(packed)
            gru_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=x.size(2)
            )

            # Average pooling over valid frames only
            # Create mask dynamically based on current batch's T
            mask_t = (
                torch.arange(x.size(2), device=x.device)[None, :] < lengths[:, None]
            ).float().unsqueeze(-1)

            video_embedding = (gru_out * mask_t).sum(dim=1) / mask_t.sum(dim=1).clamp(min=1e-6)
        else:
            gru_out, _ = self.gru(feat_vec)
            video_embedding = gru_out.mean(dim=1)

        # Volume Regression and Sorting
        pred_vols = F.softplus(self.volume_regressor(video_embedding))
        pred_edv, _ = pred_vols.max(dim=1, keepdim=True)
        pred_esv, _ = pred_vols.min(dim=1, keepdim=True)

        return {
            "mask_logits": mask_logits,
            "hidden_features": avg_feat.view(B, T, -1),
            "pred_edv": pred_edv,
            "pred_esv": pred_esv,
            "pred_ef": (pred_edv - pred_esv) / (pred_edv + 1e-6),
        }
