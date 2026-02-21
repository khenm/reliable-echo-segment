import torch
import torch.nn as nn
import math

class HomoscedasticUncertaintyWeighting(nn.Module):
    """
    Applies homoscedastic uncertainty weighting to balance multiple loss components dynamically.
    Instead of manually tuning weights for all loss components, this introduces learnable
    variance parameters for each component.
    
    L_total = sum_k ( 1/(2*sigma_k^2) * L_k + log(sigma_k) )
    We optimize over s_k = log(sigma_k^2), so precision = exp(-s_k)
    L_total = sum_k ( 0.5 * exp(-s_k) * L_k + 0.5 * s_k )
    """
    def __init__(self, manual_weights: dict):
        super().__init__()
        self.log_vars = nn.ParameterDict()
        for k, w in manual_weights.items():
            if w > 0:
                init_val = -math.log(2.0 * w)
            else:
                init_val = 0.0 # Will not be weighted if weight is 0
            self.log_vars[k] = nn.Parameter(torch.tensor([init_val], dtype=torch.float32))
            
    def forward(self, losses_dict):
        """
        Computes the dynamically weighted total loss.
        """
        total_loss = 0.0
        effective_weights = {}
        
        for k, var in self.log_vars.items():
            if k in losses_dict:
                precision = torch.exp(-torch.clamp(var, min=-4.0, max=4.0))
                total_loss += 0.5 * precision * losses_dict[k] + 0.5 * var
                
                # Store the effective weight for logging
                effective_weights[k] = (0.5 * precision).detach().cpu().item()
                
        return total_loss, effective_weights
