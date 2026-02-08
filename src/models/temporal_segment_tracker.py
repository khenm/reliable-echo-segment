import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from src.registry import register_model
from src.utils.logging import get_logger
from src.models.segment_tracker import SegmentationDecoder
from src.models.temporal_shift import TemporalShiftModule, ConvLSTM

logger = get_logger()


@register_model("temporal_segment_tracker")
class TemporalEchoSegmentTracker(nn.Module):
    """
    Temporal Segmentation with TSM + ConvLSTM bottleneck.
    
    Architecture:
        1. TSM before spatial encoding (inject temporal context to 2D conv)
        2. ConvNeXt Encoder (spatial features per frame)
        3. Conv1D Temporal Injector
        4. Optional ConvLSTM at bottleneck (spatial-temporal learning)
        5. GRU for sequence modeling
        6. Decoder for mask upsampling
    """

    @classmethod
    def from_config(cls, cfg):
        model_cfg = cfg.get('model', {})
        return cls(
            backbone=model_cfg.get('backbone', 'convnext_tiny'),
            hidden_dim=model_cfg.get('hidden_dim', 256),
            output_size=model_cfg.get('output_size', 112),
            pretrained=model_cfg.get('pretrained', True),
            shift_fraction=model_cfg.get('shift_fraction', 0.125),
            use_convlstm=model_cfg.get('use_convlstm', True)
        )

    def __init__(
        self,
        backbone: str = 'convnext_tiny',
        hidden_dim: int = 256,
        output_size: int = 112,
        pretrained: bool = True,
        shift_fraction: float = 0.125,
        use_convlstm: bool = True
    ):
        super().__init__()
        self.use_convlstm = use_convlstm

        self.tsm = TemporalShiftModule(shift_fraction=shift_fraction)

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

        if use_convlstm:
            self.bottleneck_lstm = ConvLSTM(
                input_dim=self.feature_dim,
                hidden_dim=self.feature_dim,
                kernel_size=3
            )
            logger.info("Using ConvLSTM bottleneck for spatial-temporal features")

        self.temporal = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )

        self.decoder = SegmentationDecoder(hidden_dim=hidden_dim, output_size=output_size)
        self.volume_regressor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)  # Output: [EDV, ESV] (Normalized)
        )

        logger.info(
            f"TemporalEchoSegmentTracker initialized: backbone={backbone}, "
            f"hidden_dim={hidden_dim}, shift_fraction={shift_fraction}, "
            f"use_convlstm={use_convlstm}"
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None, return_features: bool = False):
        """
        Args:
            x: (B, C, T, H, W) input video
            lengths: (B,) actual frame counts
            return_features: If True, returns features
        Returns:
            dict: {
                "mask_logits": (B, 1, T, H, W),
                "pred_edv": (B, 1),
                "pred_esv": (B, 1),
                "pred_ef": (B, 1),
                "features": Optional (B, T, D),
                "hidden_states": (B, T, D)
            }
        """
        B, C, T, H, W = x.shape

        x_time = x.permute(0, 2, 1, 3, 4)
        x_shifted = self.tsm(x_time)

        x_flat = x_shifted.reshape(B * T, C, H, W)
        features = self.encoder(x_flat)

        if features.dim() == 4:
            spatial_h, spatial_w = features.shape[2], features.shape[3]
            features_spatial = features.view(B, T, self.feature_dim, spatial_h, spatial_w)

            if self.use_convlstm:
                features_lstm = self.bottleneck_lstm(features_spatial)
                features = features_lstm.mean(dim=(3, 4))
            else:
                features = features.view(B, T, -1)
                features = features[:, :, :self.feature_dim]
        else:
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
        
        if lengths is not None:
            mask = torch.arange(T, device=x.device)[None, :] < lengths[:, None]
            mask = mask.unsqueeze(-1).float()
            video_embedding = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        else:
            video_embedding = hidden_states.mean(dim=1)

        pred_vols = self.volume_regressor(video_embedding)
        pred_vols = F.softplus(pred_vols) 
        
        pred_edv = pred_vols[:, 0:1]
        pred_esv = pred_vols[:, 1:2]

        pred_ef = (pred_edv - pred_esv) / (pred_edv + 1e-6)
        
        return {
            "mask_logits": mask_logits,
            "pred_edv": pred_edv,
            "pred_esv": pred_esv,
            "pred_ef": pred_ef,
            "features": features if return_features else None,
            "hidden_states": hidden_states
        }
