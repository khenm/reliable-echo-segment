import torch
import torch.nn as nn

class KLLoss(nn.Module):
    """
    Kullback-Leibler Divergence Loss for VAEs.
    Computes the KL divergence between the learned latent distribution and a standard Gaussian.
    """
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, mu, log_var):
        # KL Divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        # Sum over latent dim, mean over batch usually
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        kl_loss = torch.mean(kl_div)
        return self.weight * kl_loss
