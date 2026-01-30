import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Any, Tuple
from src.registry import register_model

class ConvBlock(nn.Module):
    """
    Standard convolution block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UpBlock(nn.Module):
    """
    Upsampling block: Bilinear Up -> Concat -> ConvBlock
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

@register_model("unet_tcm")
class UNet2D(nn.Module):
    """
    2D U-Net with ResNet backbone and Dual Heads (Segmentation + EF).
    """
    def __init__(self, 
                 backbone_name: str = 'resnet34', 
                 num_classes: int = 4, 
                 in_channels: int = 1,
                 pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        # Encoder (Backbone)
        if backbone_name == 'resnet34':
            base_model = models.resnet34(pretrained=pretrained)
        else:
            raise NotImplementedError(f"Backbone {backbone_name} not implemented yet.")
        
        # Adjust first conv if input is not RGB
        if in_channels != 3:
            base_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.enc0 = nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu)
        self.pool = base_model.maxpool
        self.enc1 = base_model.layer1
        self.enc2 = base_model.layer2
        self.enc3 = base_model.layer3
        self.enc4 = base_model.layer4

        # Decoder
        self.up1 = UpBlock(512 + 256, 256)
        self.up2 = UpBlock(256 + 128, 128)
        self.up3 = UpBlock(128 + 64, 64)
        self.up4 = UpBlock(64 + 64, 64)

        # Heads
        self.seg_head = nn.Conv2d(64, num_classes, kernel_size=1)
        
        # EF Head (Regression)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Temporal Aggregation via GRU
        self.temporal_agg = nn.GRU(input_size=512, hidden_size=128, batch_first=True)
        
        self.ef_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid() 
        )

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]):
        return cls(
            backbone_name=cfg['model'].get('backbone', 'resnet34'),
            num_classes=cfg['data'].get('num_classes', 4),
            in_channels=cfg['model'].get('in_channels', 1),
            pretrained=cfg['model'].get('pretrained', True)
        )

    def forward(self, x: torch.Tensor, return_features: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns:
            logits: Segmentation logits (B, num_classes, H, W)
            ef: Predicted EF scalar (B, 1)
            features: Bottleneck features (B, 512)
        """
        is_video = False
        if x.ndim == 5:
            is_video = True
            B, C, T, H, W = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        
        # Encoder
        x0 = self.enc0(x)      
        x_pool = self.pool(x0) 
        
        x1 = self.enc1(x_pool) 
        x2 = self.enc2(x1)     
        x3 = self.enc3(x2)     
        x4 = self.enc4(x3)     

        features = self.global_pool(x4).flatten(1)
        
        # Decoder
        d4 = self.up1(x4, x3)
        d3 = self.up2(d4, x2)
        d2 = self.up3(d3, x1)
        d1 = self.up4(d2, x0)
        
        logits = self.seg_head(d1)
        logits = F.interpolate(logits, scale_factor=2, mode='bilinear', align_corners=True)

        if is_video:
            _, n_cls, h, w = logits.shape
            logits = logits.view(B, T, n_cls, h, w).permute(0, 2, 1, 3, 4)
            
            features_video = features.view(B, T, -1)
            
            _, hidden = self.temporal_agg(features_video) 
            ef = self.ef_head(hidden.squeeze(0))
        else:
            features_seq = features.unsqueeze(1)
            _, hidden = self.temporal_agg(features_seq)
            ef = self.ef_head(hidden.squeeze(0))

        return logits, ef, features
