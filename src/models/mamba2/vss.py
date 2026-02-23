import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.mamba2.block import Mamba2

def cross_scan(x: torch.Tensor) -> torch.Tensor:
    """
    Scans a 2D tensor in 4 directions.
    Args:
        x: (B, C, H, W)
    Returns:
        (B, 4, C, H*W)
    """
    B, C, H, W = x.shape
    x1 = x.view(B, C, -1)
    x2 = torch.flip(x, dims=[2, 3]).view(B, C, -1)
    x3 = x.transpose(2, 3).contiguous().view(B, C, -1)
    x4 = torch.flip(x.transpose(2, 3), dims=[2, 3]).contiguous().view(B, C, -1)
    return torch.stack([x1, x2, x3, x4], dim=1)

def un_scan(xs: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Inverts the cross-scan and aggregates across directions.
    Args:
        xs: (B, 4, C, H*W)
        H: spatial height
        W: spatial width
    Returns:
        (B, C, H, W)
    """
    B, _, C, L = xs.shape
    x1 = xs[:, 0].view(B, C, H, W)
    x2 = torch.flip(xs[:, 1].view(B, C, H, W), dims=[2, 3])
    x3 = xs[:, 2].view(B, C, W, H).transpose(2, 3)
    x4 = torch.flip(xs[:, 3].view(B, C, W, H).transpose(2, 3), dims=[2, 3])
    return (x1 + x2 + x3 + x4) / 4.0

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=96, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        return self.proj(x)

class VSSBlock(nn.Module):
    def __init__(self, d_model, **kwargs):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.mamba = Mamba2(d_model=d_model, **kwargs)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        residual = x
        
        # 1. Norm
        x = x.permute(0, 2, 3, 1) # (B, H, W, C)
        x = self.norm(x.to(self.norm.weight.dtype))
        x = x.permute(0, 3, 1, 2) # (B, C, H, W)
        
        # 2. Cross-Scan
        xs = cross_scan(x) # (B, 4, C, H*W)
        
        # 3. Mamba processing
        xs = xs.transpose(2, 3).contiguous() # (B, 4, L, C)
        L = H * W
        xs = xs.view(B * 4, L, C)           # (B*4, L, C)
        
        chunk_size = self.mamba.chunk_size
        pad_len = (chunk_size - (L % chunk_size)) % chunk_size
        if pad_len > 0:
            xs = F.pad(xs, (0, 0, 0, pad_len))
            
        ys = self.mamba(xs)                  # (B*4, L+pad, C)
        
        if pad_len > 0:
            ys = ys[:, :L, :]
        
        # 4. Un-Scan
        ys = ys.view(B, 4, -1, C).transpose(2, 3).contiguous() # (B, 4, C, L)
        y = un_scan(ys, H, W) # (B, C, H, W)
        
        # 5. Residual
        return y + residual

class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.RMSNorm(4 * dim)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Padding if necessary
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, W % 2, 0, H % 2))
            H, W = H + (H % 2), W + (W % 2)
            
        x0 = x[:, :, 0::2, 0::2]
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]
        
        x = torch.cat([x0, x1, x2, x3], dim=1)
        x = x.permute(0, 2, 3, 1) # (B, H/2, W/2, 4*C)
        
        x = self.norm(x.to(self.norm.weight.dtype))
        x = self.reduction(x)
        
        x = x.permute(0, 3, 1, 2).contiguous() # (B, 2*C, H/2, W/2)
        return x

class PatchExpanding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.expand = nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)
        
    def forward(self, x):
        return self.expand(x)

class PureMambaEncoder(nn.Module):
    def __init__(self, in_channels=3, embed_dim=96, depths=[2, 2, 4, 2], **kwargs):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size=4)
        
        self.stages = nn.ModuleList()
        self.merges = nn.ModuleList()
        
        dim = embed_dim
        for i in range(len(depths)):
            blocks = nn.ModuleList([
                VSSBlock(d_model=dim, **kwargs) for _ in range(depths[i])
            ])
            self.stages.append(nn.Sequential(*blocks))
            
            if i < len(depths) - 1:
                self.merges.append(PatchMerging(dim))
                dim = dim * 2
            else:
                self.merges.append(nn.Identity())
                
        self.out_channels = dim
        self.feature_dims = [embed_dim * (2**i) for i in range(len(depths))]
                
    def forward(self, x):
        x = self.patch_embed(x)
        features = []
        for i in range(len(self.stages)):
            x = self.stages[i](x)
            features.append(x)
            if i < len(self.stages) - 1:
                x = self.merges[i](x)
        return features

class PureMambaDecoder(nn.Module):
    def __init__(self, encoder_channels, output_channels=1, depths=[1, 1, 1, 1], **kwargs):
        super().__init__()
        self.stages = nn.ModuleList()
        self.expands = nn.ModuleList()
        self.adapters = nn.ModuleList()
        
        encoder_channels_rev = encoder_channels[::-1]
        
        dim = encoder_channels_rev[0]
        for i in range(len(depths)):
            if i > 0:
                target_dim = encoder_channels_rev[i]
                self.expands.append(PatchExpanding(dim))
                concat_dim = (dim // 2) + target_dim
                self.adapters.append(nn.Conv2d(concat_dim, target_dim, kernel_size=1, bias=False))
                dim = target_dim
                
            blocks = nn.ModuleList([
                VSSBlock(d_model=dim, **kwargs) for _ in range(depths[i])
            ])
            self.stages.append(nn.Sequential(*blocks))
            
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
            nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
            nn.Conv2d(dim // 4, output_channels, kernel_size=1)
        )
        
    def forward(self, encoder_features):
        features = encoder_features[::-1]
        x = features[0]
        
        for i in range(len(self.stages)):
            if i > 0:
                skip = features[i]
                x = self.expands[i-1](x)
                
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
                    
                x = torch.cat([x, skip], dim=1)
                x = self.adapters[i-1](x)
                
            x = self.stages[i](x)
            
        x = self.final_upsample(x)
        return x
