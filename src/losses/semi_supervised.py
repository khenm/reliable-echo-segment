from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor
from src.registry import register_loss

@register_loss("EchoSemiSupervisedLoss")
class EchoSemiSupervisedLoss(nn.Module):
    """
    Semi-Supervised Loss for EchoNet-Dynamic.
    
    Combines:
    1. Supervised Loss (BCE + Dice) on labeled frames (ED, ES).
    2. Temporal Consistency Loss (MSE) on unlabeled frames.
    3. Entropy Minimization Loss on unlabeled frames.
    """
    def __init__(
        self,
        lambda_supervised: float = 1.0,
        lambda_consistency: float = 1.0,
        lambda_entropy: float = 0.1,
        foreground_class: int = 0,
        eps: float = 1e-6
    ) -> None:
        super().__init__()
        self.lambda_supervised = lambda_supervised
        self.lambda_consistency = lambda_consistency
        self.lambda_entropy = lambda_entropy
        
        self.bce = nn.BCEWithLogitsLoss() 
        from monai.losses import DiceLoss
        self.dice = DiceLoss(sigmoid=True, reduction="mean")
        
        self.mse = nn.MSELoss(reduction='none') 
        self.eps = eps

    def forward(
        self, 
        logits: Tensor, 
        targets: Tensor, 
        labeled_mask: Tensor = None
    ) -> Tuple[Tensor, dict]:
        # Ensure logits are (B, T, C, H, W)
        if logits.ndim == 5 and logits.shape[1] == 1:
            logits_perm = logits.permute(0, 2, 1, 3, 4)
            targets_perm = targets.permute(0, 2, 1, 3, 4)
        else:
             logits_perm = logits
             targets_perm = targets

        B, T, C, H, W = logits_perm.shape
        loss_sup = torch.tensor(0.0, device=logits.device)
        
        if labeled_mask is None:
             labeled_mask = torch.zeros((B, T), dtype=torch.bool, device=logits.device)
        else:
             labeled_mask = labeled_mask.bool()
            
        # 1. Supervised Loss
        logits_flat = logits_perm.reshape(-1, C, H, W)
        targets_flat = targets_perm.reshape(-1, C, H, W)
        mask_flat = labeled_mask.reshape(-1)
        
        logits_labeled = logits_flat[mask_flat]
        targets_labeled = targets_flat[mask_flat]
        
        if logits_labeled.shape[0] > 0:
            l_bce = self.bce(logits_labeled, targets_labeled.float())
            l_dice = self.dice(logits_labeled, targets_labeled)
            loss_sup = l_bce + l_dice
        
        # 2. Unsupervised: Temporal Consistency
        probs = torch.sigmoid(logits_perm)
        diffs = probs[:, 1:] - probs[:, :-1]
        
        se = diffs ** 2 
        se = se.mean(dim=(2, 3, 4)) # (B, T-1)
        
        unlabeled_mask = ~labeled_mask # (B, T)
        loss_consist = torch.tensor(0.0, device=logits.device)
        
        unlabeled_transitions = unlabeled_mask[:, :-1] # (B, T-1)
        
        if unlabeled_transitions.any():
            se_masked = se * unlabeled_transitions.float()
            loss_consist = se_masked.sum() / (unlabeled_transitions.sum() + self.eps)

        # 3. Unsupervised: Entropy Minimization
        probs_flat = probs.reshape(-1, C, H, W)
        unlabeled_mask_flat = unlabeled_mask.reshape(-1)
        
        probs_unlabeled = probs_flat[unlabeled_mask_flat]
        loss_entropy = torch.tensor(0.0, device=logits.device)
        
        if probs_unlabeled.shape[0] > 0:
            p = torch.clamp(probs_unlabeled, min=self.eps, max=1.0-self.eps)
            entropy = -(p * torch.log(p) + (1-p) * torch.log(1-p))
            loss_entropy = entropy.mean()

        total_loss = (
            self.lambda_supervised * loss_sup +
            self.lambda_consistency * loss_consist +
            self.lambda_entropy * loss_entropy
        )
        
        return total_loss, {
            "loss_sup": loss_sup, 
            "loss_consist": loss_consist, 
            "loss_entropy": loss_entropy
        }
