
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.registry import register_model

class TemporalGate(nn.Module):
    """
    Computes a gating weight based on the cosine similarity between features of consecutive frames.
    
    w_temp = Sigmoid(CosineSim(f_t, f_{t-1}) - threshold)
    
    If similarity is high, w_temp -> 1 (enforce consistency).
    If similarity is low (large motion/cut), w_temp -> 0 (relax consistency).
    """
    def __init__(self, threshold: float = 0.7, scale: float = 10.0):
        super().__init__()
        self.threshold = threshold
        self.scale = scale # steeper sigmoid

    def forward(self, features_t: torch.Tensor, features_t_minus_1: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features_t: (B, D)
            features_t_minus_1: (B, D)
            
        Returns:
            w_temp: (B, 1) Gating weight.
        """
        # Normalize features
        f_t_norm = F.normalize(features_t, p=2, dim=1)
        f_prev_norm = F.normalize(features_t_minus_1, p=2, dim=1)
        
        # Cosine similarity
        similarity = (f_t_norm * f_prev_norm).sum(dim=1, keepdim=True) # (B, 1)
        
        # Gating
        # Shifted sigmoid: 1 / (1 + exp(-scale * (sim - threshold)))
        gate = torch.sigmoid(self.scale * (similarity - self.threshold))
        
        return gate, similarity

class TemporalConsistencyLoss(nn.Module):
    """
    Computes temporal consistency loss between predicted frames, weighted by the gate.
    
    L_temp = w_temp * MSE(pred_t, pred_{t-1})
    """
    def __init__(self, method: str = 'mse'):
        super().__init__()
        self.method = method
        if method == 'mse':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError(f"Method {method} not implemented")

    def forward(self, pred_t: torch.Tensor, pred_t_minus_1: torch.Tensor, gate: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            pred_t: (B, C, H, W)
            pred_t_minus_1: (B, C, H, W)
            gate: (B, 1) or None.
            
        Returns:
            Scalar loss (mean over batch).
        """
        loss = self.loss_fn(pred_t, pred_t_minus_1) # (B, C, H, W)
        
        # Reduce to (B, )
        loss = loss.view(loss.size(0), -1).mean(dim=1)
        
        if gate is not None:
            loss = loss * gate.squeeze(1)
            
        return loss.mean()
