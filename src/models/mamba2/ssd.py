# Copyright (c) 2024, Tri Dao, Albert Gu.
# Minimal implementation of Mamba-2's SSD (Structured State Space Duality) engine.

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

def segsum(x):
    """
    Stable segment sum calculation.
    """
    T = x.size(-1)
    x_cumsum = torch.cumsum(x, dim=-1)
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
    return x_segsum

def ssd_minimal_discrete(u, A, B, C, block_len=64, initial_states=None):
    """
    Arguments:
        u: (batch, n_heads, seqlen, head_dim)
        A: (batch, n_heads, seqlen)
        B: (batch, n_heads, seqlen, d_state)
        C: (batch, n_heads, seqlen, d_state)
    """
    batch, n_heads, seqlen, head_dim = u.shape
    d_state = B.shape[-1]
    
    # Force Float32 for stability
    org_dtype = u.dtype
    u = u.float()
    A = A.float()
    B = B.float()
    C = C.float()

    # 1. Chunking
    u_chunked = rearrange(u, "b h (n c) d -> b h n c d", c=block_len)
    A_chunked = rearrange(A, "b h (n c) -> b h n c", c=block_len)
    B_chunked = rearrange(B, "b h (n c) d -> b h n c d", c=block_len)
    C_chunked = rearrange(C, "b h (n c) d -> b h n c d", c=block_len)

    # 2. Intra-chunk attention
    A_cumsum = torch.cumsum(A_chunked, dim=-1)
    diff = A_cumsum[:, :, :, :, None] - A_cumsum[:, :, :, None, :]
    mask = torch.tril(torch.ones(block_len, block_len, device=u.device))
    diff = torch.where(mask == 1, diff, torch.tensor(-float('inf'), device=diff.device))
    
    # 1-Semiseparable Mask
    # L[i,j] = exp(A_i - A_j)
    L = torch.exp(diff)

    # Intra-chunk output
    # M = (C @ B^T) * L
    M = torch.einsum("bhncd, bhnwd -> bhncw", C_chunked, B_chunked) * L
    y_intra = torch.einsum("bhncw, bhnwd -> bhncd", M, u_chunked)

    # 3. Inter-chunk recurrence
    # Decay for the whole chunk
    G_chunk_decay = torch.exp(A_cumsum[:, :, :, -1]) # (b, h, n)
    
    # State contribution: L_last * B * u
    L_last = torch.exp(A_cumsum[:, :, :, -1, None] - A_cumsum[:, :, :, :])
    B_decayed = B_chunked * L_last[..., None]
    state_updates = torch.einsum("bhncd, bhnce -> bhnde", B_decayed, u_chunked)
    
    # 4. Recurrence Loop
    if initial_states is None:
        curr_state = torch.zeros(batch, n_heads, d_state, head_dim, device=u.device, dtype=torch.float32)
    else:
        curr_state = initial_states.float()

    for i in range(state_updates.shape[2]):
        L_from_start = torch.exp(A_cumsum[:, :, i, :]) # (b, h, c)
        C_decayed = C_chunked[:, :, i] * L_from_start[..., None] # (b, h, c, d_state)
        
        y_inter_chunk = torch.einsum("bhcd, bhde -> bhce", C_decayed, curr_state)
        y_intra[:, :, i] += y_inter_chunk

        # 2. Update state for NEXT chunk
        curr_state = curr_state * G_chunk_decay[:, :, i, None, None] + state_updates[:, :, i]

    y = rearrange(y_intra, "b h n c d -> b h (n c) d")
    return y.to(org_dtype), curr_state.to(org_dtype)

def mamba_chunk_scan_combined(z, x, B, C, dt, dt_bias, A, D, chunk_size=128, d_state=64, seq_idx=None):
    """
    Combined wrapper for Mamba-2 SSD.
    
    Args:
        z: (batch, seqlen, n_heads * head_dim) - projection skip
        x: (batch, seqlen, n_heads * head_dim) - input
        B: (batch, seqlen, n_groups * d_state)
        C: (batch, seqlen, n_groups * d_state)
        dt: (batch, seqlen, n_heads)
        dt_bias: (n_heads,)
        A: (n_heads,)
        D: (n_heads * head_dim,)
        chunk_size: int
        d_state: int - SSM state dimension
        
    Returns:
        out: (batch, seqlen, n_heads * head_dim)
    """
    batch, seqlen, dim = x.shape
    n_heads = dt.shape[-1]
    head_dim = dim // n_heads
    
    # Reshape inputs
    u = rearrange(x, "b l (h d) -> b h l d", h=n_heads)
    
    # 1. Discretize A and dt
    dt = F.softplus(dt + dt_bias) # (B, L, H)
    dt = rearrange(dt, "b l h -> b h l")
    
    # A is (H,). Discretize A_t = A * dt
    dA = torch.exp(A).view(1, n_heads, 1) * dt # (B, H, L)
    dA = -dA # Decay must be negative for stability in exp
    
    # 2. Discretize B and C
    # B: (B, L, G*N). We need (B, H, L, N).
    
    # Check groups
    B_dim = B.shape[-1]
    assert B_dim % d_state == 0, f"B dimension {B_dim} must be divisible by d_state {d_state}"
    ngroups = B_dim // d_state
    
    assert n_heads % ngroups == 0, f"n_heads {n_heads} must be divisible by ngroups {ngroups}"
    
    # Reshape B: (B, L, G, N)
    B_reshaped = B.view(batch, seqlen, ngroups, d_state)
    C_reshaped = C.view(batch, seqlen, ngroups, d_state)
    
    # Broadcast to heads
    # (B, L, G, N) -> (B, L, H, N) where H = G * K
    if ngroups < n_heads:
        K = n_heads // ngroups
        B_reshaped = repeat(B_reshaped, "b l g n -> b l (g k) n", k=K)
        C_reshaped = repeat(C_reshaped, "b l g n -> b l (g k) n", k=K)
        
    # Transpose to (B, H, L, N)
    B_reshaped = rearrange(B_reshaped, "b l h n -> b h l n")
    C_reshaped = rearrange(C_reshaped, "b l h n -> b h l n")
    
    # Apply dt to B
    B_reshaped = B_reshaped * dt.unsqueeze(-1)
    
    # 3. Running SSD
    # ssd_minimal_discrete expects A as (B, H, L)
    y, last_state = ssd_minimal_discrete(u, dA, B_reshaped, C_reshaped, block_len=chunk_size, initial_states=None)
    
    # 4. Final skip connection (D) and gating (z)
    y = rearrange(y, "b h l d -> b l (h d)")
    
    # D residual
    D = D.view(1, 1, -1)
    y = y + x * D
    
    # Gating
    if z is not None:
        out = y * F.silu(z)
    else:
        out = y
    
    return out, last_state

# Global compilation is disabled to prevent recompilation loops with varying hierarchical head sizes
# try:
#     ssd_minimal_discrete = torch.compile(ssd_minimal_discrete, fullgraph=True)
#     mamba_chunk_scan_combined = torch.compile(mamba_chunk_scan_combined)
# except Exception as e:
#     import logging
#     logging.warning(f"SSD: torch.compile failed or not available, using eager mode. {e}")
