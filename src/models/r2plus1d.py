import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

class EchoR2Plus1D(nn.Module):
    """
    R(2+1)D-18 model adapted for Regression (EF prediction) on EchoNet-Dynamic.
    """
    def __init__(self, pretrained=True, progress=True):
        """
        Args:
            pretrained (bool): If True, returns a model pre-trained on Kinetics-400.
            progress (bool): If True, displays a progress bar of the download to stderr.
        """
        super().__init__()
        
        # Load pre-trained R(2+1)D-18
        weights = R2Plus1D_18_Weights.DEFAULT if pretrained else None
        self.base_model = r2plus1d_18(weights=weights, progress=progress)
        
        # R(2+1)D-18 has a linear layer at model.fc with in_features=512, out_features=400 (Kinetics)
        # We replace it with a regression head: in_features=512 -> out_features=1 (EF)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, 1)
        
        # Initialize the new layer
        nn.init.normal_(self.base_model.fc.weight, 0, 0.01)
        nn.init.constant_(self.base_model.fc.bias, 0)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input video tensor of shape (B, C, T, H, W).
                            Values should be normalized (usually 0-1 or standardized).
            
        Returns:
            torch.Tensor: Predicted scalar EF (B, 1).
        """
        return self.base_model(x)
