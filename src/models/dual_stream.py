import torch
import torch.nn as nn
import torch.nn.functional as F
from src.registry import register_model
from src.models.unet_2d import UNet2D
from src.models.components.temporal_memory import TemporalMemoryBank
from src.utils.simpson import DifferentiableSimpson

@register_model("dual_stream")
class DualStreamModel(nn.Module):
    """
    Dual-Stream Temporal Architecture for Echocardiography.
    Stream 1: Frame Stream (2D U-Net) -> Segmentation & Frame-wise Metrics
    Stream 2: Temporal Stream (Transformer) -> EF Prediction
    """
    def __init__(self, 
                 backbone_name='resnet34', 
                 num_classes=4, 
                 in_channels=1, 
                 pretrained=True,
                 seq_hidden_dim=128,
                 seq_layers=2):
        super().__init__()
        
        # Frame Stream: 2D U-Net
        self.unet = UNet2D(backbone_name=backbone_name, 
                           num_classes=num_classes, 
                           in_channels=in_channels, 
                           pretrained=pretrained)
        
        # Temporal Stream: Memory Bank
        self.memory_bank = TemporalMemoryBank(input_dim=512, 
                                              hidden_dim=seq_hidden_dim, 
                                              num_layers=seq_layers)
        
        # Clinical Consistency
        self.simpson = DifferentiableSimpson()
        self.num_classes = num_classes

    @classmethod
    def from_config(cls, cfg):
        return cls(
            backbone_name=cfg['model'].get('backbone', 'resnet34'),
            num_classes=cfg['data'].get('num_classes', 4),
            in_channels=cfg['model'].get('in_channels', 1),
            pretrained=cfg['model'].get('pretrained', True),
            seq_hidden_dim=cfg['model'].get('seq_hidden_dim', 128),
            seq_layers=cfg['model'].get('seq_layers', 2)
        )

    def forward(self, x, return_features=False):
        """
        Args:
            x: (B, C, T, H, W)
            return_features: If True, returns features for additional processing.
            
        Returns:
            If return_features=False:
                (ef_seq, seg_logits, ef_simpson)
            If return_features=True:
                ((ef_seq, seg_logits, ef_simpson), features_seq)
        """
        B, C, T, H, W = x.shape
        
        # Frame Stream (Flatten Time)
        x_flat = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        
        # U-Net Forward
        seg_logits_flat, _, features_flat = self.unet(x_flat)
        
        # Reshape back to 3D
        seg_logits = seg_logits_flat.reshape(B, T, self.num_classes, H, W).permute(0, 2, 1, 3, 4)
        features_seq = features_flat.reshape(B, T, -1)
        
        # Temporal Stream
        ef_seq = self.memory_bank(features_seq)
        
        # Simpson's EF (Internal Calculation for Consistency)
        probs = F.softmax(seg_logits, dim=1)
        
        # Assuming Class 1 is LV (Left Ventricle)
        lv_mask = probs[:, 1, :, :, :]
        
        volumes = self.simpson(lv_mask)
        ef_simpson = self.simpson.calculate_ef(volumes)
        
        if return_features:
             return (ef_seq, seg_logits, ef_simpson), features_seq
             
        return ef_seq, seg_logits, ef_simpson
