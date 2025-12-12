import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

def compute_dice_coefficient(gt, pr, label_idx=1):
    """
    Computes Dice coefficient for a specific class index.
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
    Calculates Ejection Fraction (EF) assuming area acts as a proxy for volume 
    (Simpson's rule would be better, but this matches the original code logic).
    """
    if ed_area <= 0:
        return 0.0
    return (ed_area - es_area) / ed_area * 100.0

def get_bland_altman_stats(ref, pred):
    """
    Returns Pearson r, Bias (mean diff), and Limits of Agreement (1.96 * SD).
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
    Computes confusion matrix, recall, and precision for EF categorization.
    """
    # Simple confusion matrix using pandas is usually easiest, 
    # but here is a numpy implementation to match requirements
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
    Calculates ROC and AUC for detecting low EF (< threshold).
    y_ref: Ground truth EF values
    y_pred_val: Predicted EF values
    """
    y_true_bin = (np.array(y_ref) < threshold).astype(int)
    # Score: The lower the EF, the higher the likelihood of 'low EF'
    # We invert score so that lower EF = higher probability score for ROC
    scores = (threshold - np.array(y_pred_val))
    
    if len(np.unique(y_true_bin)) < 2:
        return np.array([0, 1]), np.array([0, 1]), np.nan
        
    fpr, tpr, _ = roc_curve(y_true_bin, scores)
    auc = roc_auc_score(y_true_bin, scores)
    return fpr, tpr, auc