import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Union, Any
import logging
from tqdm import tqdm

from src.registry import register_tta_component

logger = logging.getLogger(__name__)

@register_tta_component("conformal_calibrator")
class ConformalCalibrator:
    """
    Implements Conformal Prediction for classification (Prediction Sets).
    Uses Adaptive Conformal Inference (ACI) for online adaptation.
    """
    def __init__(self, alpha: float = 0.05, gamma: float = 0.01, num_classes: int = 10, **kwargs: Any):
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

    def calibrate(self, valid_loader: torch.utils.data.DataLoader, model: torch.nn.Module, device: torch.device) -> None:
        """
        Phase 4a: Offline Calibration (Conformalize the Source).
        Compute non-conformity scores (1 - true_class_prob) on validation data.
        
        Args:
            valid_loader: DataLoader for validation set.
            model: The model to calibrate.
            device: Computation device.
        """
        logger.info("Starting Conformal Calibration...")
        model.eval()
        scores_list = []
        
        with torch.no_grad():
            pbar = tqdm(valid_loader, desc="Conformal Calibration")
            for batch in pbar:
                # Handle varying data loader return formats (inputs, targets) or (inputs, targets, meta)
                if isinstance(batch, dict):
                    # Try common keys
                    if "video" in batch:
                        inputs = batch["video"]
                    elif "image" in batch:
                        inputs = batch["image"]
                    else:
                        # Fallback: assume first value is input
                        inputs = list(batch.values())[0]

                    if "target" in batch:
                        targets = batch["target"]
                    elif "label" in batch:
                        targets = batch["label"]
                    else:
                        # Fallback: assume second value is target
                        targets = list(batch.values())[1]
                elif isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0], batch[1]
                else:
                    raise ValueError("DataLoader must return tuple/list or dict with (inputs, targets, ...)")
                
                inputs, targets = inputs.to(device), targets.to(device, dtype=torch.long)
                
                # Forward pass
                out = model(inputs)
                if isinstance(out, tuple):
                    logits = out[0]
                else:
                    logits = out
                
                probs = F.softmax(logits, dim=1)
                
                # Get probability of the TRUE class
                true_class_probs = probs.gather(1, targets.unsqueeze(1)).squeeze()
                
                # Score = 1 - probability of true class
                batch_scores = 1.0 - true_class_probs
                scores_list.append(batch_scores.cpu().numpy())
                
        if not scores_list:
            logger.warning("Calibration data empty.")
            return

        self.cal_scores = np.concatenate(scores_list)
        
        # Calculate initial quantile for the target alpha
        n = len(self.cal_scores)
        q_level = np.ceil((n + 1) * (1 - self.target_alpha)) / n
        
        # Clip q_level to [0, 1]
        q_level = min(max(q_level, 0.0), 1.0)
        
        self.q_hat = np.quantile(self.cal_scores, q_level, method='higher')
        
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
             logger.warning("ConformalCalibrator not calibrated! Using default 0.95 threshold estimate.")
             current_q = 0.95
        else:
             current_q = self.quantile_from_alpha(self.current_alpha)

        # 2. Construct Sets
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
