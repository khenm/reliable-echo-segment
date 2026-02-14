import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.logging import get_logger

logger = get_logger()

# Try importing the official CUDA-optimized Mamba
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
    logger.info("Successfully imported mamba_ssm (CUDA).")
except ImportError:
    MAMBA_AVAILABLE = False
    logger.warning("mamba_ssm not found. Using MambaFallback (Pure PyTorch, Slower).")

class MambaFallback(nn.Module):
    """
    A pure PyTorch implementation of a Mamba-like block for development on non-CUDA devices (e.g. Mac).
    This approximates the functionality of the Mamba block but is NOT numerically identical.
    It uses a GRU-based approximation to mimic the selective state space behavior:
    1. Processing sequences in parallel (forward).
    2. Processing streams step-by-step (step).

    Note: This is a dev-proxy. Do not use for production training on Server.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        
        # We use a GRU to simulate the recurrent "selection" mechanism of Mamba
        # Mamba projects up -> Conv1d -> SSM -> project down
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # The "State Space" part - approximated by a GRU cell for simplicity in fallback
        # Real Mamba uses A, B, C, Delta parameters. 
        # Here we just want something that has state and processes sequences.
        self.gru = nn.GRU(
            input_size=self.d_inner,
            hidden_size=self.d_inner,
            batch_first=True
        )

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Args:
            x: (B, L, D)
        """
        B, L, D = x.shape
        
        # 1. Project
        x_and_res = self.in_proj(x)  # (B, L, 2*d_inner)
        x_in, res = x_and_res.split(self.d_inner, dim=-1)
        
        # 2. Conv1d (Causal by manually slicing)
        x_conv = x_in.transpose(1, 2) # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :L] # Construct Causal Conv
        x_conv = self.act(x_conv.transpose(1, 2)) # (B, L, d_inner)
        
        # 3. SSM (Approximated by GRU)
        if torch.is_autocast_enabled():
            x_ssm, _ = self.gru(x_conv.float())
            x_ssm = x_ssm.to(x_conv.dtype)
        else:
            x_ssm, _ = self.gru(x_conv)
        
        # 4. Gating
        x_out = x_ssm * self.act(res)
        
        # 5. Out Project
        return self.out_proj(x_out)

    def step(self, x, h_prev=None):
        """
        Streaming Inference Step.
        Args:
           x: (B, 1, D) or (B, D)
           h_prev: Hidden state (B, 1, d_inner) - GRU format
        """
        is_unbatched = x.dim() == 2
        if is_unbatched:
            x = x.unsqueeze(1) # (B, 1, D)
            
        B, L, D = x.shape
        assert L == 1, "Step mode only supports sequence length 1"
        
        # 1. Project
        x_and_res = self.in_proj(x)
        x_in, res = x_and_res.split(self.d_inner, dim=-1)
        
        # 2. Conv - For step mode, we'd ideally cache the conv state too.
        # For this simplified fallback, we'll ignore the convolution memory 
        # and just treat it as a pointwise op or identity for simplicity
        # (Implementing a full rolling buffer for Conv1d in fallback is overkill for dev)
        x_conv = self.act(x_in) 
        
        # 3. SSM (GRU Step)
        if h_prev is None:
            # GRU expects (num_layers, B, hidden)
            h_prev = torch.zeros(1, B, self.d_inner, device=x.device)
            
        # Run GRU on single step
        # GRU Input: (B, 1, d_inner)
        # GRU Hidden: (1, B, d_inner)
        if torch.is_autocast_enabled():
            x_ssm, h_new = self.gru(x_conv.float(), h_prev.float())
            x_ssm = x_ssm.to(x_conv.dtype)
            h_new = h_new.to(x_conv.dtype)
        else:
            x_ssm, h_new = self.gru(x_conv, h_prev)
        
        # 4. Gating
        x_out = x_ssm * self.act(res)
        
        # 5. Out Project
        out = self.out_proj(x_out)
        
        return out, h_new


class MambaBlock(nn.Module):
    """
    Wrapper that selects either the official Mamba or the Fallback.
    """
    def __init__(self, d_model, **kwargs):
        super().__init__()
        if MAMBA_AVAILABLE:
            self.inner = Mamba(d_model=d_model, **kwargs)
        else:
            self.inner = MambaFallback(d_model=d_model, **kwargs)

    def forward(self, x):
        return self.inner(x)
    
    def step(self, x, state=None):
        if MAMBA_AVAILABLE:
            # Official Mamba step signature is mamba.step(x, state)
            # But the official repo primarily exposes `forward`. 
            # Step generation usually requires `mamba_ssm.utils.generation`.
            # For simplicity, we assume we might need a custom step wrapper or 
            # the official implementation supports it.
            # *Correction*: Official Mamba implementation doesn't have a direct `step` method on the Module.
            # It uses `inference_params`.
            # For this wrapper, we'll need to handle that if using official Mamba.
            # However, for now, we will assume standard forward usage for training.
            # For inference, if official Mamba is used, we might need `allocate_inference_cache`.
            
            # TODO: Implement official Mamba step logic using inference_params
            # For now, we just pass through to ensure code structure is valid.
            # Real streaming with official Mamba requires keeping track of conv_state and ssm_state.
            return self.inner(x) # Placeholder for now
        else:
            return self.inner.step(x, state)
