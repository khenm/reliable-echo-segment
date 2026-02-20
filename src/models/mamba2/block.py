# Copyright (c) 2024, Tri Dao, Albert Gu.

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from .ssd import mamba_chunk_scan_combined

class Mamba2(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=64,
        d_conv=4,
        expand=2,
        headdim=64,
        d_ssm=None,
        ngroups=1,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        bias=False,
        conv_bias=True,
        norm_before_gate=False,
        chunk_size=128,
        use_mem_eff_path=True,
        layer_idx=None, 
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.headdim = headdim
        self.ngroups = ngroups
        self.chunk_size = chunk_size
        
        # d_ssm is the total dimension of the SSM states.
        # d_inner must be divisible by headdim
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
    
        self.d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        
        self.in_proj = nn.Linear(self.d_model, self.d_in_proj, bias=bias)
        
        self.conv1d = nn.Conv1d(
            in_channels=self.d_in_proj,
            out_channels=self.d_in_proj,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_in_proj,
            padding=d_conv - 1,
        )
        
        # SSM Parameters
        # A: (nheads)
        self.A_log = nn.Parameter(torch.log(torch.rand(self.nheads) * (A_init_range[1] - A_init_range[0]) + A_init_range[0]))
        
        # D: (nheads) or (d_inner)? D is skip connection per channel.
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # dt bias
        # Initialize dt bias so that softplus(dt + bias) is around init range
        # dt = exp(rand_uniform(log(dt_min), log(dt_max)))
        # inv_softplus(dt) = dt_bias (approx)
        dt = torch.exp(
            torch.rand(self.nheads) * (torch.log(torch.tensor(dt_max)) - torch.log(torch.tensor(dt_min)))
            + torch.log(torch.tensor(dt_min))
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: ln(exp(y) - 1)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias.requires_grad = True

        # Norm
        self.norm = nn.RMSNorm(self.d_inner)

        # Out projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

    def forward(self, u, inference_params=None):
        """
        u: (batch, seqlen, dim)
        """
        batch, seqlen, dim = u.shape

        # 1. Project
        zxbcdt = self.in_proj(u) # (B, L, D_in_proj)
        
        # 2. Conv1d
        # Rearrange for conv
        zxbcdt = zxbcdt.transpose(1, 2) # (B, D, L)
        zxbcdt = self.conv1d(zxbcdt)[:, :, :seqlen] # Causal padding means we slice off the end
        zxbcdt = zxbcdt.transpose(1, 2) # (B, L, D)
        
        d_inner = self.d_inner
        n_g_state = self.ngroups * self.d_state
        
        # Split logic
        z, x, B, C, dt = torch.split(
            zxbcdt, 
            [d_inner, d_inner, n_g_state, n_g_state, self.nheads],
            dim=-1
        )

        # 4. SSM
        y, last_state = mamba_chunk_scan_combined(
            z=None, 
            x=x, 
            B=B, 
            C=C, 
            dt=dt, 
            dt_bias=self.dt_bias,
            A=self.A_log,
            D=self.D,
            chunk_size=self.chunk_size,
            d_state=self.d_state
        )
        
        # 5. Output Project
        # Correct Order: Norm(y) -> Gate(y * silu(z)) -> OutProj
        y = self.norm(y) 
        out = y * F.silu(z)
        out = self.out_proj(out)
        
        return out

    def step(self, u, state=None):
        """
        Streaming Inference Step.
        Args:
            u: (B, 1, D) or (B, D)
            state: (h_prev, conv_state)
                   h_prev: (B, H, N=1, D_state, head_dim) (recurrence state) or (B, H, d_state, head_dim)
                   conv_state: (B, D_in, D_conv) (convolution buffer)
        """
        # Handle input shape
        if u.dim() == 2:
            u = u.unsqueeze(1) # (B, 1, D)
        B, L, D = u.shape
        assert L == 1, "Mamba2.step only supports sequence length 1"
        
        # Init state if None
        if state is None:
            # h_prev shape: (B, nheads, d_state, headdim)
            h_prev = torch.zeros(B, self.nheads, self.d_state, self.headdim, device=u.device, dtype=u.dtype)
            # conv_state shape: (B, d_in_proj, d_conv)
            conv_state = torch.zeros(B, self.d_in_proj, self.d_conv, device=u.device, dtype=u.dtype)
        else:
            h_prev, conv_state = state

        # 1. Project
        zxbcdt = self.in_proj(u) # (B, 1, D_in)
        
        xt = zxbcdt.transpose(1, 2) # (B, D_in, 1)
        conv_state = torch.cat([conv_state[:, :, 1:], xt], dim=2)
        
        # (B, D_in, K) * (D_in, 1, K) -> (B, D_in, K) -> sum(2) -> (B, D_in)
        x_conv = torch.sum(conv_state * self.conv1d.weight.squeeze(1).unsqueeze(0), dim=-1)
        if self.conv1d.bias is not None:
             x_conv = x_conv + self.conv1d.bias
             
        # Reshape back to (B, 1, D_in) for split
        zxbcdt = x_conv.unsqueeze(1) # (B, 1, D_in)

        # 3. Split
        d_inner = self.d_inner
        n_g_state = self.ngroups * self.d_state
        
        z, x, B_ssm, C_ssm, dt = torch.split(
            zxbcdt, 
            [d_inner, d_inner, n_g_state, n_g_state, self.nheads],
            dim=-1
        )
        
        # Reshape inputs for heads
        # x corresponds to u in SSD math
        # x: (B, 1, H*P) -> (B, H, 1, P)
        b, l, dim = x.shape
        x = rearrange(x, "b l (h p) -> b h l p", h=self.nheads) # (B, H, 1, P)
        
        # Discretize
        # dt: (B, 1, H) -> (B, H)
        dt = F.softplus(dt + self.dt_bias).squeeze(1) # (B, H)
        
        # A: (H) -> dA: (B, H)
        dA = torch.exp(self.A_log * dt) # (B, H)
        
        # B_ssm: (B, 1, G*N) -> (B, H, 1, N)
        B_reshaped = B_ssm.view(b, l, self.ngroups, self.d_state)
        C_reshaped = C_ssm.view(b, l, self.ngroups, self.d_state)
        
        if self.ngroups < self.nheads:
             K_rep = self.nheads // self.ngroups
             B_reshaped = repeat(B_reshaped, "b l g n -> b l (g k) n", k=K_rep)
             C_reshaped = repeat(C_reshaped, "b l g n -> b l (g k) n", k=K_rep)
        
        B_reshaped = rearrange(B_reshaped, "b l h n -> b h l n") # (B, H, 1, N)
        C_reshaped = rearrange(C_reshaped, "b l h n -> b h l n") # (B, H, 1, N)
        
        # Apply dt to B
        # dt: (B, H) -> (B, H, 1, 1)
        B_reshaped = B_reshaped * dt.view(b, self.nheads, 1, 1)
        
        # SSM Step (Recurrence)
        # h_t = h_{t-1} * dA + B * x
        # h_{t-1}: (B, H, N, P) or (B, H, N, P) -> our state is (B, H, N, P)
        # dA: (B, H) -> (B, H, 1, 1)
        
        dA_broad = dA.view(b, self.nheads, 1, 1)
        
        Bx = torch.einsum("bhln, bhlp -> bhnp", B_reshaped, x)
        
        h_new = h_prev * dA_broad + Bx
        
        # Output y = C * h_t
        # C: (B, H, 1, N)
        # h_new: (B, H, N, P)
        # y: (B, H, 1, P)
        
        y = torch.einsum("bhln, bhnp -> bhlp", C_reshaped, h_new)
        
        # Skip D
        # x is (B, H, 1, P)
        # D is (H*P) -> (H, P)
        D_r = self.D.view(1, self.nheads, 1, self.headdim)
        y = y + x * D_r
        
        # Reshape y: (B, 1, H*P)
        y = rearrange(y, "b h l p -> b l (h p)")
        
        # Norm + Gating
        y = self.norm(y)
        out = y * F.silu(z)
        out = self.out_proj(out)
        
        return out, (h_new, conv_state)

class Mamba2Block(nn.Module):
    def __init__(self, dim, mixer_cls=Mamba2, norm_cls=nn.RMSNorm, fused_add_norm=False, **kwargs):
        super().__init__()
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(d_model=dim, **kwargs)
        self.fused_add_norm = fused_add_norm

    def forward(self, x, inference_params=None):
        # Validation / simple residual
        residual = x
        x = self.norm(x)
        x = self.mixer(x, inference_params=inference_params)
        return x + residual

    def step(self, x, state=None):
        # Validation / simple residual
        # x: (B, 1, D)
        residual = x
        x = self.norm(x)
        x, next_state = self.mixer.step(x, state)
        return x + residual, next_state
