import torch

def predict_with_guarantee(model, image, calibrator):
    """
    Predicts segmentation mask using the calibrated threshold (RCPS).
    
    Args:
        model (torch.nn.Module): Trained segmentation model.
        image (torch.Tensor): Input image tensor of shape (1, C, H, W).
        calibrator (ConformalCalibrator): Calibrated instance holding the optimal threshold.
        
    Returns:
        torch.Tensor: Segmentation mask of shape (H, W) with guaranteed risk control.
        
    Raises:
        ValueError: If the calibrator has not been calibrated.
    """
    model.eval()
    lambda_hat = calibrator.best_lambda
    
    if lambda_hat is None:
        raise ValueError("Calibrator has not been calibrated yet.")
        
    with torch.no_grad():
        logits = model(image)
        probs = torch.softmax(logits, dim=1)
        
        max_probs, preds = torch.max(probs, dim=1)
        
        # Apply threshold: set low-confidence pixels to background (0)
        mask = preds.clone()
        mask[max_probs < lambda_hat] = 0
        
        return mask
