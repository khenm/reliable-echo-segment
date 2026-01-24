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
        
        # 1. Frame Stream: 2D U-Net
        # We reuse the UNet2D class but we might ignore its internal EF head
        self.unet = UNet2D(backbone_name=backbone_name, 
                           num_classes=num_classes, 
                           in_channels=in_channels, 
                           pretrained=pretrained)
        
        # 2. Temporal Stream: Memory Bank
        # U-Net bottleneck is 512 for ResNet34
        self.memory_bank = TemporalMemoryBank(input_dim=512, 
                                              hidden_dim=seq_hidden_dim, 
                                              num_layers=seq_layers)
        
        # 3. Clinical Consistency
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
        Returns:
            If return_features=False:
                (ef_seq, segmentation)
            If return_features=True:
                ((ef_seq, segmentation), features_seq)
            
            We also implicitly calculate EF_Simpson for loss, but usually forward supports training/inference.
            The user wants 'Consistency Loss'. We should return EF_Simpson too?
            
            Standard interface usually return predictions.
            Let's return a dictionary or tuple? 
            The existing trainer expects (ef, seg).
            But we need EF_Simpson for the loss. 
            
            Let's return:
            (ef_seq, segmentation, ef_simpson)
            
            Wait, existing trainer code in `trainer.py` probably expects `output, target` structure. 
            Modifying the return signature might break `trainer.py`.
            I should check `src/trainer.py` next.
            For now, I'll return a special object or standard tuple and handle extra outputs via a side channel or extended tuple.
            
            Let's assume we can return (ef_seq, seg_logits, ef_simpson) and update trainer.
        """
        B, C, T, H, W = x.shape
        
        # 1. Frame Stream (Flatten Time)
        x_flat = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        
        # U-Net Forward
        # logits: (BT, NumClasses, H, W)
        # ef_frame: (BT, 1) - ignored
        # features: (BT, 512)
        seg_logits_flat, _, features_flat = self.unet(x_flat)
        
        # Reshape back to 3D
        seg_logits = seg_logits_flat.reshape(B, T, self.num_classes, H, W).permute(0, 2, 1, 3, 4) # (B, C, T, H, W)
        features_seq = features_flat.reshape(B, T, -1) # (B, T, 512)
        
        # 2. Temporal Stream
        ef_seq = self.memory_bank(features_seq) # (B, 1)
        
        # 3. Simpson's EF (Internal Calculation for Consistency)
        # Softmax for probability
        probs = F.softmax(seg_logits, dim=1)
        
        # Assuming Class 1 is LV (Left Ventricle)
        # Check if num_classes > 1. If binary, maybe channel 0? standard is channel 1 for class 1.
        lv_mask = probs[:, 1, :, :, :] # (B, T, H, W)
        
        volumes = self.simpson(lv_mask) # (B, T)
        ef_simpson = self.simpson.calculate_ef(volumes) # (B, 1)
        
        if return_features:
             # Returning features for Martingale/Auditor
             return (ef_seq, seg_logits, ef_simpson), features_seq
             
        return ef_seq, seg_logits, ef_simpson
