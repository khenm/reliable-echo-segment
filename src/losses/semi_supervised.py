from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor

class EchoSemiSupervisedLoss(nn.Module):
    """
    Semi-Supervised Loss for EchoNet-Dynamic.
    
    Combines:
    1. Supervised Loss (BCE + Dice) on labeled frames (ED, ES).
    2. Temporal Consistency Loss (MSE) on unlabeled frames.
    3. Entropy Minimization Loss on unlabeled frames.
    
    Args:
        lambda_supervised: Weight for supervised loss.
        lambda_consistency: Weight for temporal consistency loss.
        lambda_entropy: Weight for entropy minimization loss.
        pixel_spacing: Unused, kept for API consistency if needed.
    """
    def __init__(
        self,
        lambda_supervised: float = 1.0,
        lambda_consistency: float = 1.0,
        lambda_entropy: float = 0.1,
        foreground_class: int = 0, # Assuming binary segmentation, class 0 or 1 depending on setup.
        eps: float = 1e-6
    ) -> None:
        super().__init__()
        self.lambda_supervised = lambda_supervised
        self.lambda_consistency = lambda_consistency
        self.lambda_entropy = lambda_entropy
        
        # Supervised components
        # Note: BCEWithLogitsLoss takes logits.
        self.bce = nn.BCEWithLogitsLoss() 
        # DiceLoss expects probabilities if sigmoid=True is not set in it, or logits if we handle it.
        # Monai DiceLoss: sigmoid=True applies sigmoid to input.
        from monai.losses import DiceLoss
        self.dice = DiceLoss(sigmoid=True, reduction="mean")
        
        self.mse = nn.MSELoss(reduction='none') # For temporal consistency
        self.eps = eps

    def forward(
        self, 
        logits: Tensor, 
        targets: Tensor, 
        labeled_mask: Tensor = None
    ) -> Tuple[Tensor, dict]:
        """
        Args:
            logits: (B, 1, T, H, W) - Model output logits.
            targets: (B, 1, T, H, W) - Ground truth (sparse). 
                     or (B, 2, 1, H, W) if constructed differently, but usually full video tensor with zeros.
            labeled_mask: (B, T) - Boolean or Float mask where 1 indicates labeled frame.
                          If None, derived from targets if possible (assuming sparse targets are non-zero).
        
        Returns:
            total_loss (scalar), dict of component losses.
        """
        # Ensure logits are (B, 1, T, H, W) or (B, T, 1, H, W).
        # Trainer says seg_logits is (B, 1, T, H, W).
        # We will work with (B, T, ...) for easier temporal indexing.
        
        if logits.ndim == 5 and logits.shape[1] == 1:
            # (B, 1, T, H, W) -> (B, T, 1, H, W)
            logits_perm = logits.permute(0, 2, 1, 3, 4)
            targets_perm = targets.permute(0, 2, 1, 3, 4)
        else:
             logits_perm = logits
             targets_perm = targets

        B, T, C, H, W = logits_perm.shape
        
        # 1. Supervised Loss (only on labeled_mask)
        loss_sup = torch.tensor(0.0, device=logits.device)
        
        # Ensure labeled_mask is boolean (B, T)
        if labeled_mask is None:
             # Try to infer? Or just zeros.
             labeled_mask = torch.zeros((B, T), dtype=torch.bool, device=logits.device)
        else:
             labeled_mask = labeled_mask.bool()
            
        # --- SUPERVISED ---
        # Extract labeled frames
        # We flatten B and T to filter.
        logits_flat = logits_perm.reshape(-1, C, H, W) # (B*T, C, H, W)
        targets_flat = targets_perm.reshape(-1, C, H, W)
        mask_flat = labeled_mask.reshape(-1) # (B*T)
        
        # Select labeled
        logits_labeled = logits_flat[mask_flat]
        targets_labeled = targets_flat[mask_flat]
        
        if logits_labeled.shape[0] > 0:
            l_bce = self.bce(logits_labeled, targets_labeled.float())
            l_dice = self.dice(logits_labeled, targets_labeled)
            loss_sup = l_bce + l_dice
        
        # 2. Unsupervised: Temporal Consistency
        # MSE(prob[t], prob[t+1]) for unlabeled frames
        probs = torch.sigmoid(logits_perm) # (B, T, C, H, W)
        
        # Compute difference between t and t+1
        # diffs: (B, T-1, C, H, W)
        diffs = probs[:, 1:] - probs[:, :-1]
        
        # Calculate Squared Error
        se = diffs ** 2 # (B, T-1, C, H, W)
        se = se.mean(dim=(2, 3, 4)) # Average over C, H, W -> (B, T-1)
        
        unlabeled_mask = ~labeled_mask # (B, T)
        
        # We need mask for transitions. (B, T-1).
        # Let's say we count transition t->t+1 if t is unlabeled.
        loss_consist = torch.tensor(0.0, device=logits.device)
        
        # Transition mask: exclude transitions where Source is Labeled? 
        # Or just average over all valid unlabeled frames' contributions?
        # Let's align with: MSE(t, t+1) for all t where t is Unlabeled.
        unlabeled_transitions = unlabeled_mask[:, :-1] # (B, T-1)
        
        if unlabeled_transitions.any():
            se_masked = se * unlabeled_transitions.float()
            loss_consist = se_masked.sum() / (unlabeled_transitions.sum() + self.eps)

        # 3. Unsupervised: Entropy Minimization
        # Minimize H(p) = -sum(p * log(p)) on unlabeled frames
        # For binary: H(p) = -(p*log(p) + (1-p)*log(1-p))
        # p is prob of class 1.
        
        # Extract unlabeled probs
        # We can reuse probs (B, T, C, H, W)
        # Flatten to (B*T, ...) then mask
        probs_flat = probs.reshape(-1, C, H, W)
        unlabeled_mask_flat = unlabeled_mask.reshape(-1)
        
        probs_unlabeled = probs_flat[unlabeled_mask_flat]
        
        loss_entropy = torch.tensor(0.0, device=logits.device)
        
        if probs_unlabeled.shape[0] > 0:
            # Clamp for numerical stability in log
            p = torch.clamp(probs_unlabeled, min=self.eps, max=1.0-self.eps)
            entropy = -(p * torch.log(p) + (1-p) * torch.log(1-p))
            loss_entropy = entropy.mean()

        # Combine
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
