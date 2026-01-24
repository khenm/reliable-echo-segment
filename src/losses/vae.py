import torch
import torch.nn as nn
from torch import Tensor

class KLLoss(nn.Module):
    """
    Kullback-Leibler Divergence Loss for VAEs.
    
    Computes the KL divergence between the learned latent distribution 
    and a standard Gaussian prior N(0, I).
    
    Args:
        weight: Weight multiplier for the loss term.
    """
    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """
        Computes the KL divergence loss.
        
        Args:
            mu: Mean of the latent distribution, shape [B, latent_dim].
            log_var: Log variance of the latent distribution, shape [B, latent_dim].
            
        Returns:
            Weighted KL divergence loss (scalar).
        """
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        kl_loss = torch.mean(kl_div)
        return self.weight * kl_loss
