import math
import torch
import torch.nn as nn

class HomoscedasticUncertaintyWeighting(nn.Module):
    """
    Balances multiple loss components dynamically using homoscedastic uncertainty.
    Optimizes variance parameters for each loss component to avoid manual weight tuning.
    """
    def __init__(self, manual_weights: dict, elastic_alpha: float = 0.1):
        super().__init__()
        self.elastic_alpha = elastic_alpha
        self.log_vars = nn.ParameterDict()
        self.initial_vars = {}
        
        for key, weight in manual_weights.items():
            init_val = -math.log(2.0 * weight) if weight > 0 else 0.0
            self.log_vars[key] = nn.Parameter(torch.tensor([init_val], dtype=torch.float32))
            self.initial_vars[key] = init_val
            
    def forward(self, losses_dict: dict):
        """
        Computes the dynamically weighted total loss with an elastic penalty 
        to prevent variance drift.
        """
        total_loss = 0.0
        effective_weights = {}
        penalty = 0.0
        
        for key, var in self.log_vars.items():
            if key in losses_dict:
                clamped_var = torch.clamp(var, min=-4.0, max=4.0)
                precision = torch.exp(-clamped_var)
                
                total_loss += 0.5 * precision * losses_dict[key] + 0.5 * clamped_var
                penalty += self.elastic_alpha * (clamped_var - self.initial_vars[key]) ** 2
                effective_weights[key] = (0.5 * precision).detach().cpu().item()
                
        return total_loss + penalty, effective_weights