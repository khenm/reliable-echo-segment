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

class KalmanDynamicsHead(nn.Module):
    """
    Kalman-Inspired Fusion Head for Volume Prediction.
    Mimics a Predict-Update cycle using temporal features as the prior
    and mask-derived area (z_t) as the measurement.
    """
    def __init__(self, hidden_dim: int, feature_dim: int = 1):
        super().__init__()
        
        # Prior prediction \hat{v}_{t|t-1} from the Mamba state (temporal prior)
        self.prior_head_raw = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 1),
        )
        
        self.meas_head = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, 1),
        )
        
        # Kalman Gain K_t gating mechanism derived from state and current token
        self.gain_head = nn.Sequential(
            nn.Linear(hidden_dim + feature_dim, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid() # Gain represents confidence from 0 to 1
        )
        
        # Scalar projection from continuous volume to sensor measurement space
        self.W = nn.Parameter(torch.tensor([1.0]))
        
        # Learned initial prior state for t=0
        self.init_state = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(self, temporal_out, z_t, temporal_tokens=None):
        """
        temporal_out: (B, T, D) - Output from temporal Mamba block (acts as state).
        z_t: (B, T, 1) - Geometric observation from mask area.
        temporal_tokens: (B, T, D) - Original features going into Mamba (optional, used for gain).
        """
        B, T, _ = temporal_out.shape
        
        # 1. Construct prior state corresponding to t 
        # (temporal_out at t-1 is the prior for t)
        padded_state = torch.cat([
            self.init_state.repeat(B, 1, 1), 
            temporal_out[:, :-1, :]
        ], dim=1)
        
        # Predict Prior \hat{v}_{t|t-1} (log-volume space softplus)
        v_prior = F.softplus(self.prior_head_raw(padded_state)) # (B, T, 1)
        
        # 2. Kalman Gain K_t calculation
        if temporal_tokens is None:
            gain_feats = torch.cat([padded_state, z_t], dim=-1)
        else:
            gain_feats = torch.cat([padded_state, z_t], dim=-1) # We use z_t instead of temporal_tokens to save dimension mismatches or simplify
            
        K_t = self.gain_head(gain_feats) # (B, T, 1)
        
        # Measure volume directly using our dedicated mlp
        m_t = F.softplus(self.meas_head(z_t))
        
        # 3. Update (Posterior) v_t
        # v_t = \hat{v}_{t|t-1} + K_t * (m_t - W * \hat{v}_{t|t-1})
        v_post = v_prior + K_t * (m_t - self.W * v_prior)
        
        # Clamp to valid normalized volume range (soft clamping allowed large max)
        return torch.clamp(v_post, min=1e-6, max=10.0)
    
    def step(self, temporal_out_t, z_t, state=None):
        """Streaming step for Kalman inference."""
        B = temporal_out_t.shape[0]
        
        # State holds the Mamba output from the PREVIOUS step
        if state is None:
            prev_state = self.init_state.repeat(B, 1, 1)
        else:
            prev_state = state
            
        v_prior = F.softplus(self.prior_head_raw(prev_state)) # (B, 1, 1)
        
        # z_t could be missing the sequences dimension here
        z_t_unsqueeze = z_t.unsqueeze(1) if z_t.ndim == 2 else z_t
        gain_feats = torch.cat([prev_state, z_t_unsqueeze], dim=-1)
        K_t = self.gain_head(gain_feats) # (B, 1, 1)
        
        m_t = F.softplus(self.meas_head(z_t_unsqueeze))
        v_post = v_prior + K_t * (m_t - self.W * v_prior)
        v_post = torch.clamp(v_post, min=1e-6, max=10.0)
        
        # The current temporal_out_t becomes the prior state for the NEXT step
        next_state = temporal_out_t 
        
        return v_post, next_state

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
        self.temporal_adapter = nn.Linear(self.feature_dim + 1, hidden_dim)

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
        self.kalman_head = KalmanDynamicsHead(hidden_dim, feature_dim=1)

        logger.info(f"CardiacMamba initialized with PureMamba backbone and KalmanDynamicsHead")

    def forward_spatial(self, x):
        enc_feats = self.encoder(x) 
        bottleneck = enc_feats[-1] 
        
        mask_logits = self.decoder(enc_feats)
        
        mask_probs = torch.sigmoid(mask_logits)

        B_T = mask_probs.shape[0]
        # Sum is more correlated with cavity area than mean which gets easily diluted by background.
        z_t = mask_probs.view(B_T, -1).sum(dim=1, keepdim=True) / 1000.0 # scale slightly so it's not huge
        
        mask_downsampled = F.adaptive_avg_pool2d(mask_probs, bottleneck.shape[2:])
        
        attended_bottleneck = bottleneck * mask_downsampled
        pooled = F.adaptive_avg_pool2d(attended_bottleneck, 1).flatten(1) 
        
        # Append sensor reading z_t to pooled features
        pooled_with_z = torch.cat([pooled, z_t], dim=1)
        temporal_tokens = self.temporal_adapter(pooled_with_z)          
        
        if self.output_size is not None and mask_logits.shape[-1] != self.output_size:
            mask_logits = F.interpolate(
                mask_logits, 
                size=(self.output_size, self.output_size), 
                mode='bilinear', 
                align_corners=False
            )
            
        return mask_logits, temporal_tokens, z_t

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None, **kwargs) -> dict:
        B, C, T, H, W = x.shape
        x_flat = x.transpose(1, 2).reshape(B * T, C, H, W)
        
        mask_logits_flat, tokens_flat, z_t_flat = self.forward_spatial(x_flat)
        
        mask_logits = mask_logits_flat.view(B, T, 1, *mask_logits_flat.shape[2:])
        tokens = tokens_flat.view(B, T, -1)
        z_t = z_t_flat.view(B, T, 1)
        
        temporal_out = self.temporal_mamba(tokens) 
        
        vol_curve = self.kalman_head(temporal_out, z_t)
        
        vols = vol_curve.squeeze(-1)
    
        if lengths is not None:
            mask_t = (torch.arange(T, device=x.device)[None, :] < lengths[:, None])
            pred_edv, pred_esv = self.differentiable_reduce(vols, mask=mask_t, tau=1.0)
        else:
            pred_edv, pred_esv = self.differentiable_reduce(vols, mask=None, tau=1.0)

        return {
            "mask_logits": mask_logits.transpose(1, 2), 
            "pred_vol_curve": vol_curve,
            "pred_edv": pred_edv,
            "pred_esv": pred_esv,
            "pred_ef": (pred_edv - pred_esv) / torch.clamp(pred_edv, min=1e-3),
            "hidden_features": temporal_out
        }

    def step(self, x: torch.Tensor, state=None):
        if state is None:
            state = {'mamba_state': None, 'kalman_state': None}
            
        mamba_state = state.get('mamba_state')
        kalman_state = state.get('kalman_state')

        # 1. Spatial
        mask_logits, token, z_t = self.forward_spatial(x)
        
        # 2. Temporal Step
        token_seq = token.unsqueeze(1) 
        temporal_out_seq, next_mamba_state = self.temporal_mamba.step(token_seq, mamba_state)
        temporal_out = temporal_out_seq 
        
        # 3. Heads
        temporal_out_flat = temporal_out.squeeze(1)
        vol_out, next_kalman_state = self.kalman_head.step(temporal_out_flat, z_t, state=kalman_state)
        
        next_state = {
            'mamba_state': next_mamba_state,
            'kalman_state': next_kalman_state,
        }
        
        return {
            "mask_logits": mask_logits, 
            "pred_vol": vol_out.squeeze(-1), 
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