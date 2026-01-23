import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Any
import logging

from src.registry import register_tta_component

logger = logging.getLogger(__name__)

# Correct import path logic
try:
    from src.registry import register_tta_component
except ImportError:
    # Fallback if I messed up the thinking, but since I just created it, it should work.
    # However, for robustness:
    def register_tta_component(name):
        def decorator(cls):
            return cls
        return decorator

@register_tta_component("conformal_calibrator")
class ConformalCalibrator:
    """
    Implements Conformal Prediction for classification (Prediction Sets).
    Uses Adaptive Conformal Inference (ACI) for online adaptation.
    """
    def __init__(self, alpha: float = 0.05, gamma: float = 0.01, num_classes: int = 10, **kwargs):
        """
        Args:
            alpha (float): Target error rate (e.g., 0.05 for 95% coverage).
            gamma (float): Step size for ACI update.
            num_classes (int): Total number of classes.
            **kwargs: Ignored additional arguments for config compatibility.
        """
        self.target_alpha = alpha
        self.current_alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        
        # Calibration storage
        self.cal_scores = np.array([])
        self.q_hat: Optional[float] = None
        
        logger.info(f"Initialized ConformalCalibrator(alpha={alpha}, gamma={gamma}, n_classes={num_classes})")

    def calibrate(self, valid_loader: torch.utils.data.DataLoader, model: torch.nn.Module, device: torch.device):
        """
        Phase 4a: Offline Calibration (Conformalize the Source).
        Compute non-conformity scores (1 - true_class_prob) on validation data.
        """
        logger.info("Starting Conformal Calibration...")
        model.eval()
        scores_list = []
        
        with torch.no_grad():
            for batch in valid_loader:
                # Handle varying data loader return formats (inputs, targets) or (inputs, targets, meta)
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0], batch[1]
                else:
                    raise ValueError("DataLoader must return tuple/list with (inputs, targets, ...)")
                
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                # Support models returning tuple (logits, features) or just logits
                out = model(inputs)
                if isinstance(out, tuple):
                    logits = out[0]
                else:
                    logits = out
                
                probs = F.softmax(logits, dim=1)
                
                # Get probability of the TRUE class
                # gather: extracts the value at the index of the target
                # targets shape: (B), unsqueeze -> (B, 1)
                # gather result: (B, 1), squeeze -> (B)
                true_class_probs = probs.gather(1, targets.unsqueeze(1)).squeeze()
                
                # Score = 1 - probability of true class
                # (Lower probability = Higher non-conformity)
                batch_scores = 1.0 - true_class_probs
                scores_list.append(batch_scores.cpu().numpy())
                
        if not scores_list:
            logger.warning("Calibration data empty.")
            return

        self.cal_scores = np.concatenate(scores_list)
        
        # Calculate initial quantile for the target alpha
        # We want the score q such that (1-alpha) of samples have score <= q
        # Standard formulation: q_level = ceil((n+1)(1-alpha)) / n
        n = len(self.cal_scores)
        q_level = np.ceil((n + 1) * (1 - self.target_alpha)) / n
        
        # Clip q_level to [0, 1]
        q_level = min(max(q_level, 0.0), 1.0)
        
        self.q_hat = np.quantile(self.cal_scores, q_level, method='higher') # 'higher' == interpolation='higher' in newer numpy
        
        logger.info(f"Calibration Complete. N={n}, Initial Quantile (q_hat): {self.q_hat:.4f}")

    def predict_interval(self, logits: torch.Tensor, martingale_stable: bool = True) -> Tuple[List[List[int]], float]:
        """
        Phase 4b: Online Interval Prediction with ACI.
        
        Args:
            logits (torch.Tensor): Prediction logits (B, C)
            martingale_stable (bool): Flag from Auditor. If False, freeze adaptation.
        
        Returns:
            prediction_sets (List[List[int]]): List of class indices for each sample.
            q_val (float): The threshold used.
        """
        probs = F.softmax(logits, dim=1)
        batch_size = probs.shape[0]
        prediction_sets = []
        
        # 1. Determine Threshold
        if self.q_hat is None:
             # Fallback if not calibrated
             logger.warning("ConformalCalibrator not calibrated! Using default 0.95 threshold estimate.")
             current_q = 0.95
        else:
             current_q = self.quantile_from_alpha(self.current_alpha)

        # 2. Construct Sets
        # Rule: include class if prob >= (1 - current_q)
        # Because score <= q <=> 1 - prob <= q <=> prob >= 1 - q
        prob_threshold = 1.0 - current_q
        
        pseudo_labels = torch.argmax(probs, dim=1)
        
        prob_np = probs.detach().cpu().numpy()
        pseudo_labels_np = pseudo_labels.detach().cpu().numpy()
        
        for i in range(batch_size):
            # Get all classes above threshold
            classes = np.nonzero(prob_np[i] >= prob_threshold)[0]
            
            # Fallback: If set is empty, include top-1
            if len(classes) == 0:
                classes = np.array([pseudo_labels_np[i]])
                
            prediction_sets.append(classes.tolist())
            
        # 3. Adaptive Conformal Inference (ACI) Update
        if martingale_stable:
            # Simplified ACI for TTA:
            # Assume pseudo-label (top-1) is the ground truth
            # Error = 1 if pseudo_label NOT in set, else 0
            
            avg_err = 0.0
            for i, p_label in enumerate(pseudo_labels_np):
                if p_label not in prediction_sets[i]:
                    avg_err += 1.0
            
            avg_err /= batch_size
            
            # Update Rule: alpha_{t+1} = alpha_t + gamma * (err - target_alpha)
            self.current_alpha = self.current_alpha + self.gamma * (avg_err - self.target_alpha)
            
            # Clip alpha to sensible range [0.001, 1.0]
            self.current_alpha = np.clip(self.current_alpha, 0.001, 0.999)

        return prediction_sets, current_q

    def quantile_from_alpha(self, alpha: float) -> float:
        """Helper to look up quantile from calibration table dynamically."""
        if len(self.cal_scores) == 0:
            return 0.95
            
        q_level = 1.0 - alpha
        q_level = min(max(q_level, 0.0), 1.0)
        return np.quantile(self.cal_scores, q_level, method='higher')
