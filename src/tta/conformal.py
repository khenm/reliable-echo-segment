import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Union, Any
import logging
from tqdm import tqdm
from abc import ABC, abstractmethod

from src.registry import register_tta_component

logger = logging.getLogger(__name__)

class BaseConformalCalibrator(ABC):
    """
    Abstract Base Class for Conformal Calibrators.
    """
    def __init__(self, alpha: float = 0.05, gamma: float = 0.01, num_classes: Optional[int] = None, **kwargs: Any):
        """
        Args:
            alpha (float): Target error rate (e.g., 0.05 for 95% coverage).
            gamma (float): Step size for ACI update.
            num_classes (int, optional): Total number of classes.
            **kwargs: Ignored additional arguments.
        """
        self.target_alpha = alpha
        self.current_alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        
        # Calibration storage
        self.cal_scores = np.array([])
        self.q_hat: Optional[float] = None
        
        logger.info(f"Initialized {self.__class__.__name__}(alpha={alpha}, gamma={gamma})")

    @abstractmethod
    def calibrate(self, valid_loader: torch.utils.data.DataLoader, model: torch.nn.Module, device: torch.device) -> None:
        """
        Phase 4a: Offline Calibration (Conformalize the Source).
        Compute non-conformity scores on validation data.
        """
        pass

    @abstractmethod
    def predict(self, logits: torch.Tensor, martingale_stable: bool = True) -> Tuple[Any, float]:
        """
        Phase 4b: Online Prediction with ACI.
        
        Args:
            logits (torch.Tensor): Prediction logits.
            martingale_stable (bool): Flag from Auditor. If False, freeze adaptation/ACI.
        
        Returns:
            prediction (Any): Conformal prediction (Sets, Interval, or Mask).
            q_val (float): The threshold used.
        """
        pass
    
    def quantile_from_alpha(self, alpha: float) -> float:
        """Helper to look up quantile from calibration table dynamically."""
        if len(self.cal_scores) == 0:
            return 0.95
            
        q_level = 1.0 - alpha
        q_level = min(max(q_level, 0.0), 1.0)
        return np.quantile(self.cal_scores, q_level, method='higher')

    def _update_alpha(self, error_rate: float):
        """Standard ACI update rule."""
        # Update Rule: alpha_{t+1} = alpha_t + gamma * (err - target_alpha)
        self.current_alpha = self.current_alpha + self.gamma * (error_rate - self.target_alpha)
        
        # Clip alpha to sensible range [0.001, 1.0]
        self.current_alpha = np.clip(self.current_alpha, 0.001, 0.999)


@register_tta_component("conformal_calibrator")  # Legacy name
@register_tta_component("classification_calibrator")
class ClassificationCalibrator(BaseConformalCalibrator):
    """
    Implements Conformal Prediction for classification (Prediction Sets).
    Uses Adaptive Conformal Inference (ACI) for online adaptation.
    """
    def __init__(self, alpha: float = 0.05, gamma: float = 0.01, num_classes: int = 10, **kwargs: Any):
        super().__init__(alpha=alpha, gamma=gamma, num_classes=num_classes, **kwargs)

    def calibrate(self, valid_loader: torch.utils.data.DataLoader, model: torch.nn.Module, device: torch.device) -> None:
        logger.info("Starting Classification Conformal Calibration...")
        model.eval()
        scores_list = []
        
        with torch.no_grad():
            pbar = tqdm(valid_loader, desc="Conformal Calibration")
            for batch in pbar:
                # Handle varying data loader return formats
                if isinstance(batch, dict):
                    if "video" in batch: inputs = batch["video"]
                    elif "image" in batch: inputs = batch["image"]
                    else: inputs = list(batch.values())[0]

                    if "target" in batch: targets = batch["target"]
                    elif "label" in batch: targets = batch["label"]
                    else: targets = list(batch.values())[1]
                elif isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0], batch[1]
                else:
                    raise ValueError("DataLoader must return tuple/list or dict")
                
                inputs, targets = inputs.to(device), targets.to(device, dtype=torch.long)
                
                # Forward pass
                out = model(inputs)
                logits = out[0] if isinstance(out, tuple) else out
                
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
        
        # Calculate initial quantile
        n = len(self.cal_scores)
        q_level = np.ceil((n + 1) * (1 - self.target_alpha)) / n
        q_level = min(max(q_level, 0.0), 1.0)
        
        self.q_hat = np.quantile(self.cal_scores, q_level, method='higher')
        logger.info(f"Calibration Complete. N={n}, Initial Quantile (q_hat): {self.q_hat:.4f}")

    def predict(self, logits: torch.Tensor, martingale_stable: bool = True) -> Tuple[List[List[int]], float]:
        """
        Returns:
            prediction_sets (List[List[int]]): List of class indices for each sample.
            q_val (float): The threshold used.
        """
        probs = F.softmax(logits, dim=1)
        batch_size = probs.shape[0]
        prediction_sets = []
        
        # 1. Determine Threshold
        if self.q_hat is None:
             logger.warning("Calibrator not calibrated! Using default 0.95.")
             current_q = 0.95
        else:
             current_q = self.quantile_from_alpha(self.current_alpha)

        # 2. Construct Sets
        prob_threshold = 1.0 - current_q
        
        pseudo_labels = torch.argmax(probs, dim=1)
        
        prob_np = probs.detach().cpu().numpy()
        pseudo_labels_np = pseudo_labels.detach().cpu().numpy()
        
        for i in range(batch_size):
            classes = np.nonzero(prob_np[i] >= prob_threshold)[0]
            if len(classes) == 0:
                classes = np.array([pseudo_labels_np[i]])
            prediction_sets.append(classes.tolist())
            
        # 3. ACI Update
        if martingale_stable:
            avg_err = 0.0
            for i, p_label in enumerate(pseudo_labels_np):
                if p_label not in prediction_sets[i]:
                    avg_err += 1.0
            avg_err /= batch_size
            
            self._update_alpha(avg_err)

        return prediction_sets, current_q
    
    # Alias for backward compatibility (if called directly)
    def predict_interval(self, logits, martingale_stable=True):
        return self.predict(logits, martingale_stable)


@register_tta_component("regression_calibrator")
class RegressionCalibrator(BaseConformalCalibrator):
    """
    Implements Conformal Prediction for Regression (EF Estimation).
    Outputs a prediction interval [y_hat - q, y_hat + q].
    """
    def __init__(self, alpha: float = 0.05, gamma: float = 0.01, **kwargs: Any):
        super().__init__(alpha=alpha, gamma=gamma, **kwargs)

    def calibrate(self, valid_loader, model, device):
        logger.info("Starting Regression Conformal Calibration...")
        model.eval()
        residuals = []
        
        with torch.no_grad():
            for batch in valid_loader:
                # Handle varying data formats simply for regression (assuming [0]=input, [1]=target)
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0], batch[1]
                elif isinstance(batch, dict):
                     inputs = list(batch.values())[0]
                     targets = list(batch.values())[1]
                
                inputs, targets = inputs.to(device), targets.to(device)
                
                preds = model(inputs)
                if isinstance(preds, tuple): preds = preds[0]
                
                # Non-conformity score: Absolute Error
                batch_residuals = torch.abs(preds.squeeze() - targets.squeeze())
                residuals.extend(batch_residuals.cpu().numpy())
                
        self.cal_scores = np.array(residuals)
        
        n = len(self.cal_scores)
        if n == 0:
             logger.warning("Regression calibration empty.")
             self.q_hat = 0.5 # Default fallback
             return

        q_level = np.ceil((n + 1) * (1 - self.target_alpha)) / n
        self.q_hat = np.quantile(self.cal_scores, q_level)
        logger.info(f"Calibration Complete. Margin (q_hat): +/- {self.q_hat:.4f}")

    def predict(self, logits: torch.Tensor, martingale_stable: bool = True) -> Tuple[List[Tuple[float, float]], float]:
        """Returns [lower_bound, upper_bound] for EF."""
        preds = logits.detach().cpu().numpy().squeeze()
        if preds.ndim == 0: preds = np.array([preds]) # Handle batch size 1 scalar
        
        # Determine Threshold
        if self.q_hat is None:
            current_q = 0.5 # Fallback
        else:
            # We can apply ACI here too if we had a way to verify coverage online (requires ground truth)
            # For regression TTA without labels, we usually stick to static or unsupervised ACI.
            # Here we assume static for simplicity unless we have pseudo-label logic.
            current_q = self.q_hat 
        
        # Create Intervals
        lower_bounds = np.maximum(0.0, preds - current_q)
        upper_bounds = np.minimum(1.0, preds + current_q) # Assuming normalized target
        
        prediction_intervals = list(zip(lower_bounds, upper_bounds))
        return prediction_intervals, current_q

    def predict_interval(self, logits, martingale_stable=True):
        return self.predict(logits, martingale_stable)


@register_tta_component("segmentation_calibrator")
class SegmentationCalibrator(BaseConformalCalibrator):
    """
    Outputs a Core Mask (high confidence) and a Shadow Mask (uncertainty zone).
    """
    def __init__(self, alpha: float = 0.05, gamma: float = 0.01, **kwargs: Any):
        super().__init__(alpha=alpha, gamma=gamma, **kwargs)

    def calibrate(self, valid_loader, model, device):
        """
        Pixel-wise calibration.
        Score s_ij = 1 - p_ij(y_ij)
        """
        logger.info("Starting Segmentation Conformal Calibration (Pixel-wise)...")
        model.eval()
        scores_list = []
        
        # Limit calibration samples to avoid OOM for segmentation
        max_samples = 100 
        curr_samples = 0
        
        with torch.no_grad():
            for batch in valid_loader:
                if curr_samples >= max_samples: break
                
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0], batch[1]
                elif isinstance(batch, dict):
                     inputs = list(batch.values())[0]
                     targets = list(batch.values())[1] # Assuming targets are masks (B, H, W) or (B, 1, H, W)
                
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward
                logits = model(inputs)
                if isinstance(logits, tuple): logits = logits[0]
                
                probs = torch.sigmoid(logits) # (B, 1, H, W)
                
                # Check target shape
                if targets.ndim == 3: targets = targets.unsqueeze(1) # (B, 1, H, W)
                
                # Score = 1 - prob of true class. 
                # For binary: if y=1, s = 1-p. If y=0, s = 1-(1-p) = p.
                
                # Gather probability of true label
                # p_true = p if y=1 else (1-p)
                p_true = torch.where(targets > 0.5, probs, 1.0 - probs)
                
                batch_scores = 1.0 - p_true
                
                # Downsample scores to save memory? Or take random subset of pixels
                # Let's take random subset of 1000 pixels per image to estimate quantile
                flat_scores = batch_scores.flatten()
                if flat_scores.numel() > 10000:
                    idx = torch.randperm(flat_scores.numel())[:10000]
                    flat_scores = flat_scores[idx]
                    
                scores_list.append(flat_scores.cpu().numpy())
                curr_samples += inputs.shape[0]

        if not scores_list:
            logger.warning("Segmentation calibration empty.")
            self.q_hat = 0.1
            return

        self.cal_scores = np.concatenate(scores_list)
        n = len(self.cal_scores)
        
        # Calculate Quantile
        q_level = np.ceil((n + 1) * (1 - self.target_alpha)) / n
        self.q_hat = np.quantile(self.cal_scores, q_level)
        logger.info(f"Segmentation Calibration Complete. q_hat: {self.q_hat:.4f}")

    def predict(self, logits: torch.Tensor, martingale_stable: bool = True) -> Tuple[Tuple[torch.Tensor, torch.Tensor], float]:
        """
        Returns:
            (core_mask, shadow_mask): Tensors
            q_val: Threshold
        """
        probs = torch.sigmoid(logits) # (B, C, H, W)
        
        if self.q_hat is None:
             current_q = 0.1 # default
        else:
             current_q = self.q_hat
             
        # Upper threshold = 1 - q_hat (high confidence)
        # Lower threshold = q_hat (low confidence, if q_hat is small error)
        # Wait, if q_hat is score threshold (e.g. 0.05), then 1-p_true <= 0.05 => p_true >= 0.95
        # So included set is {y | p(y) >= 1 - q_hat}
        
        # For segmentation, we often want the range of valid probabilities?
        # Let's stick to the logic:
        # Core: p >= 1 - q_hat
        # Shadow: q_hat <= p < 1 - q_hat  (Ambiguous region)
        
        upper_thresh = 1.0 - current_q
        lower_thresh = current_q
        
        core_mask = (probs >= upper_thresh).int()
        shadow_mask = ((probs >= lower_thresh) & (probs < upper_thresh)).int()
        
        return (core_mask, shadow_mask), current_q

    def predict_mask(self, logits, martingale_stable=True):
        return self.predict(logits, martingale_stable)

# Alias for backward compatibility
ConformalCalibrator = ClassificationCalibrator

@register_tta_component("audit_calibrator")
class AuditCalibrator(BaseConformalCalibrator):
    """
    Wraps multiple calibrators for hybrid models (e.g., Regression + Segmentation).
    Returns a dictionary of predictions: {'regression': ..., 'segmentation': ...}
    """
    def __init__(self, calibrators: Dict[str, BaseConformalCalibrator], **kwargs: Any):
        super().__init__(**kwargs)
        self.calibrators = calibrators
        logger.info(f"Initialized AuditCalibrator with: {list(calibrators.keys())}")

    def calibrate(self, valid_loader, model, device):
        logger.info("Starting Audit (Hybrid) Calibration...")
        for name, cal in self.calibrators.items():
            logger.info(f"Calibrating sub-component: {name}")
            # Note: We rely on the sub-calibrator to handle the model's output correctly.
            # However, standard calibrators expect model(x) -> logits.
            # If model returns tuple ((ef, seg), features) or (ef, seg), we might need to wrap/shim the model
            # so the sub-calibrator sees what it expects.
            
            # Simple shim: Wrap model to return specific output for each calibrator
            if name == 'regression':
                # Expects scalar/regression output
                # Model returns (ef, seg) normally.
                shim_model = self._create_shim(model, index=0)
            elif name == 'segmentation':
                # Expects mask output
                shim_model = self._create_shim(model, index=1)
            else:
                shim_model = model
            
            cal.calibrate(valid_loader, shim_model, device)
            
    def predict(self, logits: Any, martingale_stable: bool = True) -> Tuple[Dict[str, Any], float]:
        """
        Args:
            logits: Tuple (ef_pred, seg_pred)
        """
        results = {}
        # We assume order matches what we expect or we explicitly unpack
        # logits is (ef, seg)
        
        if not isinstance(logits, (tuple, list)):
            raise ValueError("AuditCalibrator expects tuple/list logits.")
            
        # Hardcoded for now based on R2Plus1D dual output: (ef, seg)
        # TODO: Make this more flexible if needed
        
        q_vals = []
        
        if 'regression' in self.calibrators:
            res, q = self.calibrators['regression'].predict(logits[0], martingale_stable)
            results['regression'] = res
            q_vals.append(q)
            
        if 'segmentation' in self.calibrators:
            res, q = self.calibrators['segmentation'].predict(logits[1], martingale_stable)
            results['segmentation'] = res
            q_vals.append(q)
            
        # Return max q_val or avg? Usually just return one for logging, or both?
        # Contract says return float. Let's return the regression one if present, or max.
        final_q = max(q_vals) if q_vals else 0.0
        
        return results, final_q

    def _create_shim(self, model, index):
        class Shim(torch.nn.Module):
            def __init__(self, m, idx):
                super().__init__()
                self.model = m
                self.idx = idx
            def forward(self, x):
                out = self.model(x)
                if isinstance(out, tuple):
                    return out[self.idx]
                return out
        return Shim(model, index)
