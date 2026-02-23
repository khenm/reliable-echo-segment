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
    def __init__(self, manual_weights: dict, elastic_alpha: float = 0.1):
        super().__init__()
        self.elastic_alpha = elastic_alpha
        self.log_vars = nn.ParameterDict()
        self.initial_vars = {}
        for k, w in manual_weights.items():
            if w > 0:
                init_val = -math.log(2.0 * w)
            else:
                init_val = 0.0 # Will not be weighted if weight is 0
            self.log_vars[k] = nn.Parameter(torch.tensor([init_val], dtype=torch.float32))
            self.initial_vars[k] = init_val
            
    def forward(self, losses_dict):
        """
        Computes the dynamically weighted total loss.
        """
        total_loss = 0.0
        effective_weights = {}
        penalty = 0.0
        
        for k, var in self.log_vars.items():
            if k in losses_dict:
                clamped_var = torch.clamp(var, min=-4.0, max=4.0)
                precision = torch.exp(-clamped_var)
                total_loss += 0.5 * precision * losses_dict[k] + 0.5 * clamped_var
                
                # Store the effective weight for logging
                init_var = self.initial_vars[k]
                penalty += self.elastic_alpha * (clamped_var - init_var) ** 2

                effective_weights[k] = (0.5 * precision).detach().cpu().item()
                
        return total_loss + penalty, effective_weights
