import numpy as np
import torch
from sklearn.metrics import roc_curve, roc_auc_score, r2_score, mean_absolute_error, mean_squared_error

def compute_dice_coefficient(gt, pr, label_idx=1):
    """
    Computes Dice coefficient for a specific class index.
    
    Args:
        gt (np.ndarray): Ground truth mask.
        pr (np.ndarray): Predicted mask.
        label_idx (int): Class label to evaluate.
        
    Returns:
        float: Dice coefficient.
    """
    gt_bin = (gt == label_idx)
    pr_bin = (pr == label_idx)
    inter = np.logical_and(gt_bin, pr_bin).sum()
    denom = gt_bin.sum() + pr_bin.sum()
    if denom == 0:
        return 1.0
    return 2.0 * inter / (denom + 1e-8)

def calculate_ef_from_areas(ed_area, es_area):
    """
    Calculates Ejection Fraction (EF) using area-based approximation.
    
    Args:
        ed_area (float): End-diastolic area.
        es_area (float): End-systolic area.
        
    Returns:
        float: Calculated Ejection Fraction.
    """
    if ed_area <= 0:
        return 0.0
    return (ed_area - es_area) / ed_area * 100.0

def calculate_ef_interval(ed_vol, es_vol, delta_ed, delta_es, epsilon=1e-6):
    """
    Calculates the uncertainty interval for Ejection Fraction (EF) using Taylor Expansion.
    
    Formula:
        Delta EF approx sqrt( (dEF/dVs * Delta Vs)^2 + (dEF/dVd * Delta Vd)^2 )
        dEF/dVs = -1 / Vd
        dEF/dVd = Vs / Vd^2
        
    Args:
        ed_vol (float): End-diastolic volume (or area).
        es_vol (float): End-systolic volume (or area).
        delta_ed (float): Uncertainty in ED volume.
        delta_es (float): Uncertainty in ES volume.
        epsilon (float): Small value to prevent division by zero.
        
    Returns:
        float: The calculated uncertainty radius for EF (Delta EF) in percentage points.
    """
    if ed_vol < epsilon:
        # If Vd is effectively zero, EF is undefined or 0. 
        # Return a large uncertainty or 0 depending on philosophy. 
        # Returning 0 to avoid NaNs, assuming EF is just 0.
        return 0.0
        
    # Partial derivatives
    # f = 1 - Vs/Vd
    # df/dVs = -1/Vd
    # df/dVd = Vs / Vd^2
    
    term_vs = (-1.0 / ed_vol) * delta_es
    term_vd = (es_vol / (ed_vol ** 2)) * delta_ed
    
    # Combined uncertainty (radius)
    delta_ef = np.sqrt(term_vs**2 + term_vd**2)
    
    # Scale to percentage
    return delta_ef * 100.0

def get_bland_altman_stats(ref, pred):
    """
    Computes Bland-Altman statistics: Pearson correlation, Bias, and Limits of Agreement.
    
    Args:
        ref (list or np.ndarray): Reference values.
        pred (list or np.ndarray): Predicted values.
        
    Returns:
        tuple: (Pearson r, Bias, Limits of Agreement)
    """
    ref = np.asarray(ref, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(ref) & np.isfinite(pred)
    ref, pred = ref[mask], pred[mask]

    if ref.size < 2:
        return np.nan, np.nan, np.nan

    r = np.corrcoef(ref, pred)[0, 1]
    diff = pred - ref
    bias = diff.mean()
    loa = 1.96 * diff.std(ddof=1)
    
    return r, bias, loa

def get_classification_metrics(y_true, y_pred, labels):
    """
    Computes classification metrics including confusion matrix, recall, and precision.
    
    Args:
        y_true (list): Ground truth labels.
        y_pred (list): Predicted labels.
        labels (list): List of class labels.
        
    Returns:
        tuple: (Conflict Matrix, Recall, Precision)
    """
    n_labels = len(labels)
    cm = np.zeros((n_labels, n_labels), dtype=int)
    
    for t, p in zip(y_true, y_pred):
        if t in labels and p in labels:
            ti = labels.index(t)
            pi = labels.index(p)
            cm[ti, pi] += 1
            
    # Per-class stats
    diag = np.diag(cm)
    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)
    
    recall = np.divide(diag, row_sum, out=np.zeros_like(diag, dtype=float), where=row_sum > 0)
    precision = np.divide(diag, col_sum, out=np.zeros_like(diag, dtype=float), where=col_sum > 0)
    
    return cm, recall, precision

def get_roc_auc_low_ef(y_ref, y_pred_val, threshold=45.0):
    """
    Calculates ROC and AUC for detecting low EF.
    
    The predicted EF is inverted (threshold - pred) to use as a score, 
    since lower EF implies higher probability of the 'low EF' condition.
    
    Args:
        y_ref (list): Ground truth EF values.
        y_pred_val (list): Predicted EF values.
        threshold (float): Threshold to define 'low EF' class.
        
    Returns:
        tuple: (FPR, TPR, AUC)
    """
    y_true_bin = (np.array(y_ref) < threshold).astype(int)
    scores = (threshold - np.array(y_pred_val))
    
    if len(np.unique(y_true_bin)) < 2:
        return np.array([0, 1]), np.array([0, 1]), np.nan
        
    fpr, tpr, _ = roc_curve(y_true_bin, scores)
    auc = roc_auc_score(y_true_bin, scores)
    return fpr, tpr, auc


class SkeletalError:
    """
    Computes the Mean Euclidean Distance (Pixel Error) for skeletal keypoints.
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.total_dist = 0.0
        self.count = 0
        
    def __call__(self, preds, targets):
        """
        Args:
            preds (torch.Tensor): Predicted keypoints (B, T, N, 2).
            targets (torch.Tensor): Ground truth keypoints (B, T, N, 2).
        """
        # Calculate Euclidean distance per point
        # diff: (B, T, N, 2)
        diff = preds.float() - targets.float()
        
        # dist: (B, T, N)
        dist = torch.norm(diff, dim=-1)
        
        self.total_dist += dist.sum().item()
        self.count += dist.numel()
        
    def aggregate(self):
        if self.count == 0:
            return 0.0
        return self.total_dist / self.count

class R2Score:
    """
    Computes the Coefficient of Determination (R^2 Score).
    Uses sklearn.metrics.r2_score internally.
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.preds = []
        self.targets = []
        
    def __call__(self, preds, targets):
        """
        Args:
            preds (torch.Tensor or np.ndarray): Predicted values.
            targets (torch.Tensor or np.ndarray): Ground truth values.
        """
        if isinstance(preds, torch.Tensor):
            preds = preds.float().detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.float().detach().cpu().numpy()
            
        self.preds.extend(preds.reshape(-1).tolist())
        self.targets.extend(targets.reshape(-1).tolist())
        
    def aggregate(self):
        if not self.preds:
            return 0.0
        return r2_score(self.targets, self.preds)

class MAE:
    """
    Computes Mean Absolute Error.
    """
    def __init__(self, reduction="mean"):
        self.reduction = reduction
        self.reset()
        
    def reset(self):
        self.preds = []
        self.targets = []
        
    def __call__(self, preds, targets):
        if isinstance(preds, torch.Tensor):
            preds = preds.float().detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.float().detach().cpu().numpy()
            
        self.preds.extend(preds.reshape(-1).tolist())
        self.targets.extend(targets.reshape(-1).tolist())
        
    def aggregate(self):
        if not self.preds:
            return 0.0
        return mean_absolute_error(self.targets, self.preds)

class RMSE:
    """
    Computes Root Mean Squared Error.
    """
    def __init__(self, reduction="mean"):
        self.reduction = reduction
        self.reset()
        
    def reset(self):
        self.preds = []
        self.targets = []
        
    def __call__(self, preds, targets):
        if isinstance(preds, torch.Tensor):
            preds = preds.float().detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.float().detach().cpu().numpy()
            
        self.preds.extend(preds.reshape(-1).tolist())
        self.targets.extend(targets.reshape(-1).tolist())
        
    def aggregate(self):
        if not self.preds:
            return 0.0
        return np.sqrt(mean_squared_error(self.targets, self.preds))