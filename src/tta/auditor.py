import torch
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)

class SelfAuditor:
    def __init__(self, feature_dim, alpha=0.5, delta=0.05, lambda_val=0.5):
        """
        Args:
            feature_dim (int): Dimension of the feature embedding Phi(x).
            alpha (float): Weighting between Entropy (0) and Drift (1).
            delta (float): Error tolerance (e.g., 0.05 means 95% confidence).
            lambda_val (float): Betting factor (aggressive vs conservative growth).
        """
        # Hyperparameters
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.delta = delta
        self.lambda_val = lambda_val
        self.threshold = 1.0 / delta  # Rejection threshold
        
        # State
        self.martingale = 1.0  # Initial wealth
        self.t = 0
        self.collapsed = False
        
        # Calibration Statistics (Populated in Step 1)
        self.mu_source = None  # Mean embedding of source
        self.epsilon = None    # Expected score on validation set
        
        # Buffer for running stats (optional, for debugging)
        self.scores = []

    def _compute_entropy(self, inputs):
        """
        Computes entropy for a batch of inputs (logits or tuple).
        Returns:
            entropy: (B,) tensor
            max_ent: float (normalization factor)
        """
        logits = inputs
        # 1. Handle Tuple (Hybrid Models)
        if isinstance(logits, (tuple, list)):
            # Heuristic: Find first tensor with ndim >= 3 (Spatial)
            # If none, find first tensor with ndim == 2 (Classification)
            # If none, take the first element
            selected = logits[0]
            for item in logits:
                if isinstance(item, torch.Tensor) and item.ndim >= 3:
                    selected = item
                    break
            logits = selected
            
        # 2. Compute Entropy
        if logits.ndim > 2:
            # Spatial/Temporal Segmentation (B, C, T, H, W) or (B, C, H, W)
            if logits.shape[1] == 1:
                # Binary Segmentation -> Use Sigmoid
                probs = torch.sigmoid(logits)
                ent = -(probs * torch.log(probs + 1e-10) + (1-probs) * torch.log(1-probs + 1e-10))
                # Average over all remaining dimensions (C, T, H, W)
                # Note: dim 1 is Channel (size 1), so averaging it is fine/noop
                dims = list(range(1, ent.ndim))
                ent = ent.mean(dim=dims)
                max_ent = np.log(2) # Max entropy for binary
            else:
                # Multiclass -> Use Softmax
                probs = F.softmax(logits, dim=1)
                ent = -torch.sum(probs * torch.log(probs + 1e-10), dim=1) # (B, T, H, W)
                dims = list(range(1, ent.ndim))
                ent = ent.mean(dim=dims) 
                
                num_classes = logits.shape[1]
                max_ent = np.log(num_classes)
            
        else:
            # Flat Classification (B, C) or Regression (B, 1)
            if logits.shape[1] == 1:
                # Binary/Regression
                # Assuming logits. Use Sigmoid.
                p = torch.sigmoid(logits)
                ent = -(p * torch.log(p + 1e-10) + (1-p) * torch.log(1-p + 1e-10))
                ent = ent.squeeze(1)
                max_ent = np.log(2)
            else:
                probs = F.softmax(logits, dim=1)
                ent = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
                max_ent = np.log(logits.shape[1])
                
        return ent, max_ent

    def calibrate(self, valid_loader, model, device):
        """
        Step 1: Calibration
        Run over source validation set to calculate mu_source and epsilon.
        
        Args:
            valid_loader (torch.utils.data.DataLoader): DataLoader for validation set.
            model (torch.nn.Module): The model to audit.
            device (torch.device): Device to run computation on.
        """
        model.eval()
        features_list = []
        entropies_list = []
        
        logger.info("Starting SelfAuditor Calibration...")
        max_ent_sample = 1.0

        with torch.no_grad():
            for batch in valid_loader:
                if isinstance(batch, dict):
                    if "video" in batch:
                        inputs = batch["video"]
                    elif "image" in batch:
                        inputs = batch["image"]
                    else:
                        inputs = list(batch.values())[0]
                elif isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                else:
                    inputs = batch
                    
                inputs = inputs.to(device)
                
                try:
                    out = model(inputs, return_features=True)
                    if isinstance(out, tuple):
                        logits, features = out
                    else:
                        logits = out
                        features = getattr(model, 'features', None)
                        if features is None:
                             raise ValueError("Model must return features or have 'features' attribute for SelfAuditor.")
                except TypeError:
                     logits = model(inputs)
                     raise ValueError("SelfAuditor requires model to support feature extraction via return_features=True or similar mechanism.")

                # 1. Collect Features for Drift
                features_list.append(features.cpu())
                
                # 2. Collect Entropy
                entropy, max_e = self._compute_entropy(logits)
                entropies_list.append(entropy.cpu())
                max_ent_sample = max_e

        if not features_list:
            logger.warning("Calibration data empty.")
            return

        # Compute Mu_Source
        all_features = torch.cat(features_list, dim=0)
        self.mu_source = torch.mean(all_features, dim=0).to(device)
        self.mu_source = F.normalize(self.mu_source, dim=0) 
        
        # Compute Expected Score (Epsilon)
        all_entropies = torch.cat(entropies_list, dim=0)
        
        # Normalize Entropy roughly to [0, 1]
        norm_entropies = all_entropies / max_ent_sample
        
        # Calculate Drift for all validation samples
        all_features_norm = F.normalize(all_features, dim=1)
        
        # Drift Calculation
        # drift = 1 - sim
        # (N, D) @ (D, 1) -> (N, 1)
        drift_scores = 1 - (all_features_norm @ self.mu_source.cpu().unsqueeze(1)).squeeze()
        
        # Composite Scores
        scores = self.alpha * norm_entropies + (1 - self.alpha) * drift_scores
        
        # Set Epsilon
        self.epsilon = scores.mean().item()
        
        # Safety margin
        self.epsilon *= 1.05 
        
        logger.info(f"Calibration Complete. Epsilon: {self.epsilon:.4f}, Threshold: {self.threshold:.2f}, MaxEnt: {max_ent_sample:.4f}")

    def update(self, logits, features):
        """
        Step 2 & 3: Calculate Score and Update Martingale
        Args:
            logits: Current batch logits
            features: Current batch features
        Returns:
            stop_adaptation (bool): True if model should reset/stop
        """
        if self.collapsed:
            return True

        # --- 1. Calculate Entropy Component ---
        entropy, max_ent = self._compute_entropy(logits)
        norm_entropy = entropy / max_ent

        # --- 2. Calculate Drift Component ---
        # Normalize features
        feat_norm = F.normalize(features, dim=1)
        # Cosine distance to source mean
        drift = 1 - torch.matmul(feat_norm, self.mu_source)
        
        # --- 3. Composite Score (s_t) ---
        # Average over the batch to get a single scalar signal for the time step t
        batch_score_t = self.alpha * norm_entropy.mean() + (1 - self.alpha) * drift.mean()
        
        # --- 4. Martingale Update (The Betting) ---
        # M_t = M_{t-1} * (1 + lambda * (s_t - epsilon))
        
        score_val = batch_score_t.item()
        self.scores.append(score_val)
        
        bet = 1 + self.lambda_val * (score_val - self.epsilon)
        
        # Safety clipping: Wealth cannot be negative
        bet = max(0, bet) 
        
        self.martingale *= bet
        self.t += 1
        
        # --- 5. Rejection Rule ---
        if self.martingale >= self.threshold:
            self.collapsed = True
            logger.warning(f"SelfAuditor: Collapse detected at step {self.t}! Martingale={self.martingale:.2f}")
            return True
            
        return False

    def reset_audit(self):
        """Resets the martingale (e.g. after a recovery intervention)"""
        self.martingale = 1.0
        self.collapsed = False
        logger.info("SelfAuditor: Martingale reset.")
