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
        with torch.no_grad():
            for inputs, _ in valid_loader:
                inputs = inputs.to(device)
                
                # Assume model returns (logits, features)
                # Note: Model must support return_features=True or return a tuple
                # We handle the case where the model might output just logits if not adjusted,
                # but per design it requires features.
                try:
                    out = model(inputs, return_features=True)
                    if isinstance(out, tuple):
                        logits, features = out
                    else:
                        # Fallback if model doesn't return tuple but has attribute
                        logits = out
                        features = getattr(model, 'features', None)
                        if features is None:
                             raise ValueError("Model must return features or have 'features' attribute for SelfAuditor.")
                except TypeError:
                     # If return_features is not a valid arg, assume standard forward
                     logits = model(inputs)
                     # Start of strict assumption: Model MUST provide features for Drift
                     raise ValueError("SelfAuditor requires model to support feature extraction via return_features=True or similar mechanism.")

                # 1. Collect Features for Drift
                features_list.append(features.cpu())
                
                # 2. Collect Entropy
                probs = F.softmax(logits, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
                entropies_list.append(entropy.cpu())

        if not features_list:
            logger.warning("Calibration data empty.")
            return

        # Compute Mu_Source
        all_features = torch.cat(features_list, dim=0)
        self.mu_source = torch.mean(all_features, dim=0).to(device)
        self.mu_source = F.normalize(self.mu_source, dim=0) # Normalize for Cosine
        
        # Compute Expected Score (Epsilon)
        all_entropies = torch.cat(entropies_list, dim=0)
        
        # Normalize Entropy roughly to [0, 1]
        # Using logits shape from last iteration to determine max_ent
        if logits is not None:
             max_ent = np.log(logits.shape[1])
        else:
             max_ent = 1.0 # Fallback
             
        norm_entropies = all_entropies / max_ent
        
        # Calculate Drift for all validation samples
        # Cosine Distance = 1 - Cosine Similarity
        # Normalize features per sample
        all_features_norm = F.normalize(all_features, dim=1)
        
        # Ensure mu_source is on CPU for this massive batch calculation if needed, 
        # or keep on device if generic. Let's start with CPU to avoid OOM on large Val sets.
        drift_scores = 1 - (all_features_norm @ self.mu_source.cpu().unsqueeze(1)).squeeze()
        
        # Composite Scores
        scores = self.alpha * norm_entropies + (1 - self.alpha) * drift_scores
        
        # Set Epsilon (Expected value under Null Hypothesis)
        self.epsilon = scores.mean().item()
        
        # Safety margin: slightly inflate epsilon to make betting harder (conservative)
        self.epsilon *= 1.05 
        
        logger.info(f"Calibration Complete. Epsilon: {self.epsilon:.4f}, Threshold: {self.threshold:.2f}")

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
        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        max_ent = np.log(logits.shape[1])
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
