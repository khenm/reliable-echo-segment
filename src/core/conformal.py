import numpy as np
import torch

class ConformalCalibrator:
    def __init__(self, scores, masks):
        """
        Initializes the ConformalCalibrator.
        
        Args:
            scores: (N, H, W) Calibration scores.
            masks: (N, H, W) Ground truth masks.
        """
        self.best_lambda = None
        self.lambdas = np.linspace(0, 1, 101) # Grid of thresholds
        
    def _compute_dice(self, score_map, mask, thresh):
        """
        Computes Dice for a single image/mask pair at a threshold.
        score_map: (H, W) or (N, H, W)
        mask: (H, W) or (N, H, W)
        """
        pred = (score_map >= thresh).float()
        
        intersection = (pred * mask).sum(dim=(-1, -2))
        union = pred.sum(dim=(-1, -2)) + mask.sum(dim=(-1, -2))
        
        # Dice = 2*Int / (Union + epsilon)
        dice = (2.0 * intersection) / (union + 1e-8)
        return dice

    def calibrate(self, scores, masks, alpha=0.1):
        """
        Finds the optimal lambda_hat such that Risk <= alpha.
        Risk = 1 - Dice.
        
        scores: (N, H, W) torch tensor or numpy array, probabilities of positive class.
        masks: (N, H, W) torch tensor or numpy array, binary ground truth.
        """
        if not torch.is_tensor(scores):
            scores = torch.tensor(scores)
        if not torch.is_tensor(masks):
            masks = torch.tensor(masks)
            
        scores = scores.float()
        masks = masks.float()
        
        n = scores.shape[0]

        self.risks = []
        valid_lambdas = []
        
        # Hoeffding Bound parameters
        delta = 0.05 
        
        for lam in self.lambdas:
            dices = self._compute_dice(scores, masks, lam)
            losses = 1.0 - dices
            r_hat = losses.mean().item()
            
            # Hoeffding Upper Bound
            # P(R > r_hat + t) <= exp(-2nt^2) = delta
            # -2nt^2 = ln(delta) => t^2 = -ln(delta)/(2n) => t = sqrt(-ln(delta)/(2n))
            bound = r_hat + np.sqrt(-np.log(delta) / (2 * n))
            
            self.risks.append({
                "lambda": lam,
                "empirical_risk": r_hat,
                "upper_bound": bound
            })
            
            if bound <= alpha:
                valid_lambdas.append(lam)
        
        if not valid_lambdas:
            print("Warning: No lambda found satisfying the guarantee. Returning lambda with min risk.")
            # Fallback: lambda with min empirical risk
            best_idx = np.argmin([x['empirical_risk'] for x in self.risks])
            self.best_lambda = self.risks[best_idx]['lambda']
        else:
            # Pick the valid lambda that minimizes empirical risk
            valid_stats = [x for x in self.risks if x['lambda'] in valid_lambdas]
            best_valid = min(valid_stats, key=lambda x: x['empirical_risk'])
            self.best_lambda = best_valid['lambda']
            
        print(f"Optimal lambda: {self.best_lambda}")
        return self.best_lambda

    def get_risk_curve(self):
        """Returns the computed risks for plotting."""
        return self.risks
