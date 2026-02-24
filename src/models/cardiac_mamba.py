import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.registry import register_model
from src.utils.logging import get_logger
from src.models.mamba2.block import Mamba2Block
from src.models.mamba2.vss import PureMambaEncoder, PureMambaDecoder

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
            
        return F.softplus(vol_curve)

    def forward(self, tokens, is_streaming=False, prev_phase=None, coeff_ema=None):
        B, T, D = tokens.shape
        raw_vel = self.phase_vel_head(tokens)
        phase_vel = raw_vel * (2 * math.pi / (torch.abs(self.timescale) + 1e-6))
        
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
    Streaming Cardiac Model using a Unified Pure Mamba Architecture.
    """
    @classmethod
    def from_config(cls, cfg: dict) -> "CardiacMamba":
        model_cfg = cfg.get('model', {})
        return cls(
            hidden_dim=model_cfg.get('hidden_dim', 256),
            output_size=model_cfg.get('output_size', 112),
            d_state=model_cfg.get('d_state', 16),
            d_conv=model_cfg.get('d_conv', 4),
            expand=model_cfg.get('expand', 2),
            chunk_size=model_cfg.get('chunk_size', 32),
        )

    def __init__(
        self,
        hidden_dim: int = 256,
        output_size: int = 112,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        chunk_size: int = 32,
    ):
        super().__init__()
        
        self.output_size = output_size
        
        # 1. Spatial Encoder (Pure Mamba)
        self.encoder = PureMambaEncoder(
            in_channels=3,
            embed_dim=96,
            depths=[2, 2, 4, 2],
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            chunk_size=chunk_size
        )
        self.feature_dim = self.encoder.feature_dims[-1]

        # Clean Linear Adapter mapping spatial bottleneck -> temporal token
        self.temporal_adapter = nn.Linear(self.feature_dim, hidden_dim)

        # 2. Segmentation Decoder (Frame-wise Native Projection)
        self.decoder = PureMambaDecoder(
            encoder_channels=self.encoder.feature_dims, 
            output_channels=1,
            depths=[1, 1, 1, 1],
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            chunk_size=chunk_size
        )

        # 3. Temporal Module (Mamba)
        self.temporal_mamba = MambaBlock(
            d_model=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            chunk_size=chunk_size,
        )

        # 4. Multi-Task Heads
        self.fourier_vol_head = FourierVolumeHead(hidden_dim, K=3)

        logger.info(f"CardiacMamba initialized with PureMamba backbone")

    def forward_spatial(self, x):
        enc_feats = self.encoder(x) 
        bottleneck = enc_feats[-1] 
        
        mask_logits = self.decoder(enc_feats)
        
        mask_probs = torch.sigmoid(mask_logits)
        mask_downsampled = F.adaptive_avg_pool2d(mask_probs, bottleneck.shape[2:])
        
        attended_bottleneck = bottleneck * mask_downsampled
        pooled = F.adaptive_avg_pool2d(attended_bottleneck, 1).flatten(1) 
        temporal_tokens = self.temporal_adapter(pooled)          
        
        if self.output_size is not None and mask_logits.shape[-1] != self.output_size:
            mask_logits = F.interpolate(
                mask_logits, 
                size=(self.output_size, self.output_size), 
                mode='bilinear', 
                align_corners=False
            )
            
        return mask_logits, temporal_tokens

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None, **kwargs) -> dict:
        B, C, T, H, W = x.shape
        x_flat = x.transpose(1, 2).reshape(B * T, C, H, W)
        
        mask_logits_flat, tokens_flat = self.forward_spatial(x_flat)
        
        mask_logits = mask_logits_flat.view(B, T, 1, *mask_logits_flat.shape[2:])
        tokens = tokens_flat.view(B, T, -1)
        
        temporal_out = self.temporal_mamba(tokens) 
        
        vol_curve, phase, phase_vel, coeffs = self.fourier_vol_head(temporal_out)
        
        vols = vol_curve.squeeze(-1)
        if lengths is not None:
            mask_t = (torch.arange(T, device=x.device)[None, :] < lengths[:, None])
            pred_edv, pred_esv = self.differentiable_reduce(vols, mask=mask_t, tau=10.0)
        else:
            pred_edv, pred_esv = self.differentiable_reduce(vols, mask=None, tau=10.0)
            
        return {
            "mask_logits": mask_logits.transpose(1, 2), 
            "pred_vol_curve": vol_curve,
            "pred_phase_vel": phase_vel,
            "pred_edv": pred_edv,
            "pred_esv": pred_esv,
            "pred_ef": (pred_edv - pred_esv) / torch.clamp(pred_edv, min=1.0),
            "hidden_features": temporal_out
        }

    def step(self, x: torch.Tensor, state=None):
        if state is None:
            state = {'mamba_state': None, 'prev_phase': None, 'coeff_ema': None}
            
        mamba_state = state.get('mamba_state')
        prev_phase = state.get('prev_phase')
        coeff_ema = state.get('coeff_ema')

        # 1. Spatial
        mask_logits, token = self.forward_spatial(x)
        
        # 2. Temporal Step
        token_seq = token.unsqueeze(1) 
        temporal_out_seq, next_mamba_state = self.temporal_mamba.step(token_seq, mamba_state)
        temporal_out = temporal_out_seq 
        
        # 3. Heads
        vol_curve, phase, phase_vel, coeff_ema = self.fourier_vol_head(
            temporal_out, is_streaming=True, prev_phase=prev_phase, coeff_ema=coeff_ema
        )
        
        temporal_out_flat = temporal_out.squeeze(1) 
        
        next_state = {
            'mamba_state': next_mamba_state,
            'prev_phase': phase,
            'coeff_ema': coeff_ema
        }
        
        return {
            "mask_logits": mask_logits, 
            "pred_vol": vol_curve.squeeze(1), 
            "pred_phase_vel": phase_vel.squeeze(1),
            "hidden_features": temporal_out_flat
        }, next_state

    def differentiable_reduce(self, vols, mask=None, tau=1.0):
        """
        Computes a differentiable soft-maximum and soft-minimum across the sequence.
        tau (temperature): Controls the sharpness of the approximation. 
        """
        vols_centered = vols - vols.mean(dim=1, keepdim=True)
        
        logits_max = vols_centered / tau
        logits_min = -vols_centered / tau
        
        if mask is not None:
            mask_fill = (mask == 0)
            safe_min = torch.finfo(logits_max.dtype).min
            logits_max = logits_max.masked_fill(mask_fill, safe_min)
            logits_min = logits_min.masked_fill(mask_fill, safe_min)
            
        attn_weights_max = F.softmax(logits_max, dim=1)
        soft_val_max = torch.sum(attn_weights_max * vols, dim=1)
        
        attn_weights_min = F.softmax(logits_min, dim=1)
        soft_val_min = torch.sum(attn_weights_min * vols, dim=1)
        
        return soft_val_max, soft_val_min