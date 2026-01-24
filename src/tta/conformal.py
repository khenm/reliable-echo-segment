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
    def predict(self, logits: torch.Tensor, martingale_stable: bool = True, audit_score: Optional[float] = None, audit_epsilon: Optional[float] = None) -> Tuple[Any, float]:
        """
        Phase 4b: Online Prediction with ACI.
        
        Args:
            logits (torch.Tensor): Prediction logits.
            martingale_stable (bool): Flag from Auditor. If False, freeze adaptation/ACI.
            audit_score (float, optional): Current risk score from SelfAuditor.
            audit_epsilon (float, optional): Expected risk threshold.
        
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
        # Update Rule: alpha_{t+1} = alpha_t - gamma * (err - target_alpha)
        # We subtract because if err > target, we are under-covering, so we need wider intervals.
        # Wider intervals correspond to higher quantile (1-alpha), which means smaller alpha.
        self.current_alpha = self.current_alpha - self.gamma * (error_rate - self.target_alpha)
        
        # Clip alpha to sensible range [0.001, 1.0]
        self.current_alpha = np.clip(self.current_alpha, 0.001, 0.999)


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
                
                # Ensure 1D for storage
                batch_scores_np = batch_scores.cpu().numpy()
                if batch_scores_np.ndim == 0:
                    batch_scores_np = np.expand_dims(batch_scores_np, axis=0)
                    
                scores_list.append(batch_scores_np)
                
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

    def predict(self, logits: torch.Tensor, martingale_stable: bool = True, audit_score: Optional[float] = None, audit_epsilon: Optional[float] = None) -> Tuple[List[List[int]], float]:
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
    def predict_interval(self, logits, martingale_stable=True, audit_score=None, audit_epsilon=None):
        return self.predict(logits, martingale_stable, audit_score, audit_epsilon)


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
                # Handle varying data formats
                if isinstance(batch, dict):
                     if "video" in batch: inputs = batch["video"]
                     elif "image" in batch: inputs = batch["image"]
                     else: inputs = list(batch.values())[0]

                     if "target" in batch: targets = batch["target"]
                     elif "ef" in batch: targets = batch["ef"]
                     else: targets = list(batch.values())[1]
                elif isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0], batch[1]
                
                inputs, targets = inputs.to(device), targets.to(device)
                
                preds = model(inputs)
                if isinstance(preds, tuple): preds = preds[0]
                
                # Non-conformity score: Absolute Error
                batch_residuals = torch.abs(preds.squeeze() - targets.squeeze())
                
                # Ensure 1D for iterable extension
                batch_residuals_np = batch_residuals.cpu().numpy()
                if batch_residuals_np.ndim == 0:
                     batch_residuals_np = np.expand_dims(batch_residuals_np, axis=0)
                     
                residuals.extend(batch_residuals_np)
                
        self.cal_scores = np.array(residuals)
        
        n = len(self.cal_scores)
        if n == 0:
             logger.warning("Regression calibration empty.")
             self.q_hat = 0.5 # Default fallback
             return

        q_level = np.ceil((n + 1) * (1 - self.target_alpha)) / n
        q_level = min(max(q_level, 0.0), 1.0)
        self.q_hat = np.quantile(self.cal_scores, q_level)
        logger.info(f"Calibration Complete. Margin (q_hat): +/- {self.q_hat:.4f}")

    def predict(self, logits: torch.Tensor, martingale_stable: bool = True, audit_score: Optional[float] = None, audit_epsilon: Optional[float] = None) -> Tuple[List[Tuple[float, float]], float]:
        """Returns [lower_bound, upper_bound] for EF."""
        preds = logits.detach().cpu().numpy().squeeze()
        if preds.ndim == 0: preds = np.array([preds]) # Handle batch size 1 scalar
        
        # Determine Threshold
        if self.q_hat is None:
            current_q = 0.5 # Fallback
        else:
            # Use dynamic quantile based on ACI alpha
            current_q = self.quantile_from_alpha(self.current_alpha) 
        
        # Create Intervals
        lower_bounds = np.maximum(0.0, preds - current_q)
        upper_bounds = np.minimum(1.0, preds + current_q) # Assuming normalized target
        
        prediction_intervals = list(zip(lower_bounds, upper_bounds))

        # 3. ACI Update via Auditor Proxy
        if martingale_stable and audit_score is not None and audit_epsilon is not None:
            # Using a binary proxy for robustness
            proxy_error = 1.0 if audit_score > audit_epsilon else 0.0
            self._update_alpha(proxy_error)

        return prediction_intervals, current_q

    def predict_interval(self, logits, martingale_stable=True, audit_score=None, audit_epsilon=None):
        return self.predict(logits, martingale_stable, audit_score, audit_epsilon)


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
        # max_samples = 100 
        # curr_samples = 0
        
        with torch.no_grad():
            for batch in valid_loader:
                inputs = batch["video"].to(device)
                targets = batch["label"].to(device)      # Shape: (B, 1, T, H, W)
                frame_mask = batch["frame_mask"].to(device) # Shape: (B, T)
                
                logits = model(inputs)
                if isinstance(logits, tuple): logits = logits[1] # R2Plus1D seg output is index 1
                
                probs = torch.sigmoid(logits) # (B, 1, T, H, W)
                
                # 2. Filter out UNLABELED frames using the frame_mask
                # Flatten the batch and time dimensions: (B*T, 1, H, W)
                _, C, _, H, W = probs.shape
                probs_flat = probs.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
                targets_flat = targets.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
                frame_mask_flat = frame_mask.view(-1)
                
                # Get indices of frames that actually have ground truth
                valid_indices = torch.nonzero(frame_mask_flat).squeeze()
                
                if valid_indices.numel() == 0:
                    continue # Skip video if no labeled frames found
                    
                probs_valid = probs_flat[valid_indices]
                targets_valid = targets_flat[valid_indices]

                # 3. Calculate Conformal Scores on VALID pixels only
                # Score = 1 - p(true_class)
                p_true = torch.where(targets_valid > 0.5, probs_valid, 1.0 - probs_valid)
                batch_scores = 1.0 - p_true
                
                # Subsample pixels to prevent Out-Of-Memory (OOM)
                flat_scores = batch_scores.flatten()
                if flat_scores.numel() > 10000:
                    idx = torch.randperm(flat_scores.numel())[:10000]
                    flat_scores = flat_scores[idx]
                    
                scores_list.append(flat_scores.cpu().numpy())

    def predict(self, logits: torch.Tensor, martingale_stable: bool = True, audit_score: Optional[float] = None, audit_epsilon: Optional[float] = None) -> Tuple[Tuple[torch.Tensor, torch.Tensor], float]:
        """
        Returns:
            (core_mask, shadow_mask): Tensors
            q_val: Threshold
        """
        probs = torch.sigmoid(logits) # (B, C, H, W)
        
        # 1. Determine Threshold (Dynamic Feedback Loop)
        base_q = self.q_hat if self.q_hat is not None else 0.1
        
        sensitivity = 0.05
        if martingale_stable and audit_score is not None:
            # Dynamic relaxation
            current_q = base_q * (1.0 + sensitivity * np.log1p(audit_score))
        else:
            current_q = 0.99

        # Safety Cap
        current_q = min(current_q, 0.99)
        
        upper_thresh = 1.0 - current_q
        lower_thresh = current_q
        
        core_mask = (probs >= upper_thresh).int()
        shadow_mask = ((probs >= lower_thresh) & (probs < upper_thresh)).int()
        
        return (core_mask, shadow_mask), current_q

    def predict_mask(self, logits, martingale_stable=True, audit_score=None, audit_epsilon=None):
        return self.predict(logits, martingale_stable, audit_score, audit_epsilon)

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
            
    def predict(self, logits: Any, martingale_stable: bool = True, audit_score: Optional[float] = None, audit_epsilon: Optional[float] = None) -> Tuple[Dict[str, Any], float]:
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
        
        q_vals = []
        
        if 'regression' in self.calibrators:
            res, q = self.calibrators['regression'].predict(logits[0], martingale_stable, audit_score, audit_epsilon)
            results['regression'] = res
            q_vals.append(q)
            
        if 'segmentation' in self.calibrators:
            res, q = self.calibrators['segmentation'].predict(logits[1], martingale_stable, audit_score, audit_epsilon)
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
