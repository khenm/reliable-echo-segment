import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from src.registry import register_model
from src.utils.logging import get_logger
from src.models.segment_tracker import SegmentationDecoder
from src.models.mamba_wrapper import MambaBlock

logger = get_logger()

@register_model("cardiac_mamba")
class CardiacMamba(nn.Module):
    """
    Streaming Cardiac Model using Mamba (State Space Models).
    
    Architecture:
    1. Spatial Backbone (2D CNN): Extracts features per frame.
    2. Segmentation Head (2D U-Net): Generates masks per frame.
    3. Temporal Module (Mamba): Processes sequence of features to predict volume and phase.
    
    Capabilities:
    - Parallel Training: Sees whole clip, gradients flow through time in log(N).
    - Streaming Inference: Updates hidden state frame-by-frame, constant memory.
    """

    @classmethod
    def from_config(cls, cfg: dict) -> "CardiacMamba":
        model_cfg = cfg.get('model', {})
        return cls(
            backbone=model_cfg.get('backbone', 'efficientnet_b0'),
            pretrained=model_cfg.get('pretrained', True),
            hidden_dim=model_cfg.get('hidden_dim', 256),
            output_size=model_cfg.get('output_size', 112),
            d_state=model_cfg.get('d_state', 16),
            d_conv=model_cfg.get('d_conv', 4),
            expand=model_cfg.get('expand', 2),
            chunk_size=model_cfg.get('chunk_size', 32),
        )

    def __init__(
        self,
        backbone: str = 'efficientnet_b0',
        pretrained: bool = True,
        hidden_dim: int = 256,
        output_size: int = 112,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        chunk_size: int = 32,
    ):
        super().__init__()
        
        # 1. Spatial Encoder (Backbone)
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool='', # Keep spatial dims
            features_only=True # Return list of features for U-Net
        )
        
        # Get feature info
        feature_info = self.encoder.feature_info.info
        encoder_channels = [x['num_chs'] for x in feature_info]
        self.feature_dim = encoder_channels[-1]

        # Adapter to hidden_dim
        self.adapter = nn.Conv2d(self.feature_dim, hidden_dim, kernel_size=1)

        # 2. Segmentation Decoder (Frame-wise)
        self.decoder = SegmentationDecoder(
            hidden_dim=hidden_dim, 
            output_size=output_size, 
            spatial_input=True
        )

        # 3. Temporal Module (Mamba)
        # Mamba takes (B, L, D)
        self.temporal_mamba = MambaBlock(
            d_model=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            chunk_size=chunk_size,
        )

        # 4. Heads
        self.vol_head = nn.Linear(hidden_dim + 1, 1)
        self.phase_head = nn.Linear(hidden_dim, 3) # [Background, ES, ED]

        logger.info(f"CardiacMamba initialized with backbone={backbone}")

    def forward_spatial(self, x):
        """
        Processes a batch of frames (B*T) spatially.
        Returns:
            mask_logits: (B*T, 1, H, W)
            features: (B*T, D) - Pooled feature vector for temporal model
        """
        # x: (N, C, H, W) where N = B*T
        enc_feats = self.encoder(x) 
        bottleneck = enc_feats[-1]
        
        # Adapter
        adapted = self.adapter(bottleneck) # (N, hidden_dim, H', W')
        
        # Segmentation
        mask_logits = self.decoder(adapted) # (N, 1, H_out, W_out)
        mask_probs = torch.sigmoid(mask_logits)
        
        mask_probs_down = F.adaptive_avg_pool2d(mask_probs, adapted.shape[2:])
        
        weighted_adapted = adapted * (mask_probs_down + 0.1)
        pooled = F.adaptive_avg_pool2d(weighted_adapted, 1).flatten(1)
        
        raw_area = mask_probs.sum(dim=(2, 3)) / (mask_probs.shape[2] * mask_probs.shape[3])
        
        return mask_logits, pooled, raw_area

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None, **kwargs) -> dict:
        """
        Parallel Training Forward Pass.
        
        Args:
            x: (B, C, T, H, W)
            lengths: (B,) number of valid frames
            
        Returns:
            dict with logs, including 'pred_vol_curve', 'pred_phase', 'mask_logits'
        """
        B, C, T, H, W = x.shape
        
        # Fold time into batch
        x_flat = x.transpose(1, 2).reshape(B * T, C, H, W)
        
        # 1. Spatial Processing (Parallel over all frames)
        mask_logits_flat, tokens_flat, raw_area_flat = self.forward_spatial(x_flat)
        
        # Unfold time
        # mask_logits: (B, T, 1, H_out, W_out)
        mask_logits = mask_logits_flat.view(B, T, 1, *mask_logits_flat.shape[2:])
        # tokens: (B, T, D)
        tokens = tokens_flat.view(B, T, -1)
        raw_area = raw_area_flat.view(B, T, -1)
        
        # 2. Temporal Processing (Mamba Parallel Scan)
        temporal_out = self.temporal_mamba(tokens) # (B, T, D)
        
        # 3. Heads
        vol_input = torch.cat([temporal_out, raw_area], dim=-1)
        vol_curve = F.softplus(self.vol_head(vol_input)) # (B, T, 1) -> Positive volume
        phase_logits = self.phase_head(temporal_out)        # (B, T, 3)
        
        # 4. Aggregation
        phase_probs = F.softmax(phase_logits, dim=-1)
        
        prob_es = phase_probs[:, :, 1] # (B, T)
        prob_ed = phase_probs[:, :, 2] # (B, T)
        
        vols = vol_curve.squeeze(-1) # (B, T)
        
        if lengths is not None:
            # Mask out invalid frames so they don't contribute
            mask_t = (torch.arange(T, device=x.device)[None, :] < lengths[:, None]).float()
            prob_es = prob_es * mask_t
            prob_ed = prob_ed * mask_t
            
            # Normalize probabilities over the valid sequence length
            prob_es = prob_es / (prob_es.sum(dim=1, keepdim=True) + 1e-6)
            prob_ed = prob_ed / (prob_ed.sum(dim=1, keepdim=True) + 1e-6)
        else:
            # Normalize probabilities over the whole temporal sequence
            prob_es = prob_es / (prob_es.sum(dim=1, keepdim=True) + 1e-6)
            prob_ed = prob_ed / (prob_ed.sum(dim=1, keepdim=True) + 1e-6)

        # Soft Expected Volumes
        pred_edv = torch.sum(prob_ed * vols, dim=1) # (B,)
        pred_esv = torch.sum(prob_es * vols, dim=1) # (B,)

        return {
            "mask_logits": mask_logits.transpose(1, 2), 
            "pred_vol_curve": vol_curve,
            "pred_phase": phase_logits,
            "pred_edv": pred_edv,
            "pred_esv": pred_esv,
            "pred_ef": (pred_edv - pred_esv) / (pred_edv + 1e-6),
            "hidden_features": temporal_out
        }

    def step(self, x: torch.Tensor, state=None):
        """
        Streaming Inference Step.
        
        Args:
            x: Single frame (B, C, H, W)
            state: Hidden state from previous step (Mamba state)
            
        Returns:
            dict results, next_state
        """
        # 1. Spatial
        mask_logits, token, raw_area = self.forward_spatial(x)
        # token: (B, D)
        
        # 2. Temporal Step
        # Mamba step might need (B, 1, D)
        token_seq = token.unsqueeze(1) # (B, 1, D)
        
        temporal_out_seq, next_state = self.temporal_mamba.step(token_seq, state)
        # temporal_out_seq: (B, 1, D)
        
        temporal_out = temporal_out_seq.squeeze(1) # (B, D)
        
        # 3. Heads
        vol_input = torch.cat([temporal_out, raw_area.view(-1, 1)], dim=-1)
        vol = F.softplus(self.vol_head(vol_input)) # (B, 1)
        phase = self.phase_head(temporal_out)         # (B, 3)
        
        return {
            "mask_logits": mask_logits, # (B, 1, H, W)
            "pred_vol": vol,
            "pred_phase": phase,
            "hidden_features": temporal_out
        }, next_state
