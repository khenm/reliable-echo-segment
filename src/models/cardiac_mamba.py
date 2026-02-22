import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm

from src.registry import register_model
from src.utils.logging import get_logger
from src.models.segment_tracker import SegmentationDecoder
from src.models.mamba2.block import Mamba2Block

logger = get_logger()

class MambaBlock(nn.Module):
    """
    Wrapper for Mamba-2 Block.
    """
    def __init__(self, d_model, **kwargs):
        super().__init__()
        self.inner = Mamba2Block(dim=d_model, **kwargs)

    def forward(self, x):
        return self.inner(x)
    
    def step(self, x, state=None):
        return self.inner.step(x, state)

class FourierVolumeHead(nn.Module):
    def __init__(self, hidden_dim, K=3):
        super().__init__()
        self.K = K
        self.timescale = nn.Parameter(torch.tensor([15.0]))
        self.phase_vel_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.coeff_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1 + 2 * K)
        )

    def generate_wave(self, coeffs, phase, T):
        """Helper to reconstruct the wave from coefficients and phase."""
        a0 = F.softplus(coeffs[:, :1])
        harmonics = coeffs[:, 1:]
        
        vol_curve = a0.unsqueeze(1).repeat(1, T, 1)
        for n in range(1, self.K + 1):
            an = harmonics[:, 2*(n-1) : 2*(n-1)+1].unsqueeze(1)
            bn = harmonics[:, 2*(n-1)+1 : 2*(n-1)+2].unsqueeze(1)
            vol_curve = vol_curve + an * torch.cos(n * phase) + bn * torch.sin(n * phase)
            
        return torch.relu(vol_curve)

    def forward(self, tokens, is_streaming=False, prev_phase=None, coeff_ema=None):
        B, T, D = tokens.shape
        raw_vel = self.phase_vel_head(tokens)
        phase_vel = raw_vel * (2 * math.pi / self.timescale)
        
        if not is_streaming:
            phase = torch.cumsum(phase_vel, dim=1)
            coeffs = self.coeff_head(tokens.mean(dim=1))
        else:
            phase = (torch.zeros(B, 1, 1, device=tokens.device) if prev_phase is None else prev_phase) + phase_vel
            curr_coeffs = self.coeff_head(tokens.mean(dim=1))
            coeffs = curr_coeffs if coeff_ema is None else (0.95 * coeff_ema + 0.05 * curr_coeffs)
            
        vol_curve = self.generate_wave(coeffs, phase, T)
        return vol_curve, phase, phase_vel, coeffs

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
        self.fourier_vol_head = FourierVolumeHead(hidden_dim, K=3)
        self.phase_head = nn.Linear(hidden_dim, 3) # [Background, ES, ED]
        
        # 5. Mask Refinement
        self.refinement_conv = nn.Sequential(
            nn.Conv2d(hidden_dim + 1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)
        )

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
        
        weighted_adapted = adapted * (mask_probs_down.detach() + 0.1)
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
        
        # Unfold time for Mamba
        # tokens: (B, T, D)
        tokens = tokens_flat.view(B, T, -1)
        raw_area = raw_area_flat.view(B, T, -1)
        
        # 2. Temporal Processing (Mamba Parallel Scan)
        temporal_out = self.temporal_mamba(tokens) # (B, T, D)
        
        # 2.5 Mask Refinement
        # Broadcast temporal features to spatial dimensions
        temporal_out_spatial = temporal_out.view(B*T, -1).unsqueeze(-1).unsqueeze(-1) # (B*T, D, 1, 1)
        temporal_out_spatial = temporal_out_spatial.expand(-1, -1, mask_logits_flat.shape[2], mask_logits_flat.shape[3]) # (B*T, D, H, W)
        
        # Combine draft mask and temporal features
        refined_input = torch.cat([mask_logits_flat, temporal_out_spatial], dim=1) # (B*T, D+1, H, W)
        
        # Generate final refined masks
        final_mask_logits_flat = self.refinement_conv(refined_input) # (B*T, 1, H, W)
        
        # Unfold time
        # mask_logits: (B, T, 1, H_out, W_out)
        mask_logits = final_mask_logits_flat.view(B, T, 1, *final_mask_logits_flat.shape[2:])
        # tokens: (B, T, D)
        tokens = tokens_flat.view(B, T, -1)
        raw_area = raw_area_flat.view(B, T, -1)
        
        # 3. Heads
        vol_curve, phase, phase_vel, coeffs = self.fourier_vol_head(temporal_out)
        phase_logits = self.phase_head(temporal_out)        # (B, T, 3)
        
        # 4. Aggregation
        vols = vol_curve.squeeze(-1) # (B, T)
        if lengths is not None:
            mask_t = (torch.arange(T, device=x.device)[None, :] < lengths[:, None]).float()
            vols_max = torch.where(mask_t > 0, vols, torch.tensor(-1e9, device=vols.device))
            vols_min = torch.where(mask_t > 0, vols, torch.tensor(1e9, device=vols.device))
            pred_edv = vols_max.max(dim=1)[0]
            pred_esv = vols_min.min(dim=1)[0]
        else:
            pred_edv = vols.max(dim=1)[0]
            pred_esv = vols.min(dim=1)[0]
            
        return {
            "mask_logits": mask_logits.transpose(1, 2), 
            "pred_vol_curve": vol_curve,
            "pred_raw_area": raw_area,
            "pred_phase_vel": phase_vel,
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
            state: Dictionary containing Mamba hidden state, phase, and moving average coeffs
            
        Returns:
            dict results, next_state
        """
        if state is None:
            state = {
                'mamba_state': None,
                'prev_phase': None,
                'coeff_ema': None
            }
            
        mamba_state = state.get('mamba_state')
        prev_phase = state.get('prev_phase')
        coeff_ema = state.get('coeff_ema')

        # 1. Spatial
        mask_logits, token, raw_area = self.forward_spatial(x)
        # token: (B, D)
        
        # 2. Temporal Step
        token_seq = token.unsqueeze(1) # (B, 1, D)
        
        temporal_out_seq, next_mamba_state = self.temporal_mamba.step(token_seq, mamba_state)
        # temporal_out_seq: (B, 1, D)
        
        temporal_out = temporal_out_seq # Keeping sequence dim (B, 1, D) for Fourier Head
        
        # 2.5 Mask Refinement
        temporal_out_spatial = temporal_out.view(x.shape[0], -1).unsqueeze(-1).unsqueeze(-1) # (B, D, 1, 1)
        temporal_out_spatial = temporal_out_spatial.expand(-1, -1, mask_logits.shape[2], mask_logits.shape[3]) # (B, D, H, W)
        refined_input = torch.cat([mask_logits, temporal_out_spatial], dim=1) # (B, D+1, H, W)
        mask_logits = self.refinement_conv(refined_input) # (B, 1, H, W)
        
        vol_curve, phase, phase_vel, coeff_ema = self.fourier_vol_head(
            temporal_out, 
            is_streaming=True, 
            prev_phase=prev_phase, 
            coeff_ema=coeff_ema
        )
        vol = vol_curve
        
        # Anchor Phase classification head
        temporal_out_flat = temporal_out.squeeze(1) # (B, D)
        phase_logits = self.phase_head(temporal_out_flat) # (B, 3)
        
        next_state = {
            'mamba_state': next_mamba_state,
            'prev_phase': phase,
            'coeff_ema': coeff_ema
        }
        
        return {
            "mask_logits": mask_logits, # (B, 1, H, W)
            "pred_vol": vol.squeeze(1), # (B, 1)
            "pred_phase": phase_logits,
            "pred_phase_vel": phase_vel.squeeze(1),
            "hidden_features": temporal_out_flat
        }, next_state
