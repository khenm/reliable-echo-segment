import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from src.models.registry import ModelRegistry

class UpBlock3D(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        # x: (B, C, T, H, W)
        # skip: (B, C_skip, T_skip, H_skip, W_skip)
        
        # Upsample x to match skip size
        if x.shape[2:] != skip.shape[2:]:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
            
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

@ModelRegistry.register("r2plus1d")
class EchoR2Plus1D(nn.Module):
    """
    R(2+1)D-18 model adapted for Dual Output:
    1. Scalar Regression (EF prediction).
    2. Segmentation (Mask prediction) using a U-Net style decoder.
    """
    def __init__(self, pretrained=True, progress=True, num_classes=2):
        super().__init__()
        
        # Load pre-trained R(2+1)D-18
        weights = R2Plus1D_18_Weights.DEFAULT if pretrained else None
        base = r2plus1d_18(weights=weights, progress=progress)
        
        # Encoder Stages
        self.stem = base.stem
        self.layer1 = base.layer1 # 64
        self.layer2 = base.layer2 # 128
        self.layer3 = base.layer3 # 256
        self.layer4 = base.layer4 # 512
        self.avgpool = base.avgpool
        
        # Regression Head (EF)
        # Original fc in_features=512
        in_features = base.fc.in_features
        self.fc_ef = nn.Linear(in_features, 1)
        
        # Initialize EF head
        nn.init.normal_(self.fc_ef.weight, 0, 0.01)
        nn.init.constant_(self.fc_ef.bias, 0)
        
        # Segmentation Decoder
        # Layer 4 out: 512 channels
        # Layer 3 out: 256 channels
        # Layer 2 out: 128 channels
        # Layer 1 out: 64 channels
        # Stem out: 64 channels
        
        self.up1 = UpBlock3D(512, 256, 256)
        self.up2 = UpBlock3D(256, 128, 128)
        self.up3 = UpBlock3D(128, 64, 64)
        self.up4 = UpBlock3D(64, 64, 32) # Skip from stem (64)
        
        self.final_conv = nn.Conv3d(32, num_classes, kernel_size=1)
        
    @classmethod
    def from_config(cls, cfg):
         num_classes = cfg['data'].get('num_classes', 2)
         return cls(pretrained=True, progress=True, num_classes=num_classes)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input video tensor of shape (B, C, T, H, W).
            
        Returns:
            tuple: (ef_pred, seg_pred)
                - ef_pred (B, 1): Predicted scalar EF.
                - seg_pred (B, NumClasses, T, H, W): Predicted segmentation logits.
        """
        # Encoder
        x0 = self.stem(x)      # 64
        x1 = self.layer1(x0)   # 64
        x2 = self.layer2(x1)   # 128
        x3 = self.layer3(x2)   # 256
        x4 = self.layer4(x3)   # 512
        
        # Regression Branch
        pool = self.avgpool(x4)
        flat = pool.flatten(1)
        ef_pred = self.fc_ef(flat)
        
        # Segmentation Branch
        d1 = self.up1(x4, x3)
        d2 = self.up2(d1, x2)
        d3 = self.up3(d2, x1)
        d4 = self.up4(d3, x0)
        
        seg_pred = self.final_conv(d4)
        
        # Final upsample to match input size if needed
        if seg_pred.shape[2:] != x.shape[2:]:
            seg_pred = nn.functional.interpolate(seg_pred, size=x.shape[2:], mode='trilinear', align_corners=False)
            
        return ef_pred, seg_pred
