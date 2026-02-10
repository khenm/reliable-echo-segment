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
        self.p_value = 0.5 # Default neutral p-value
        
        # Calibration Statistics (Populated in Step 1)
        self.mu_source = None  # Mean embedding of source
        self.epsilon = None    # Expected score on validation set
        self.calibration_scores = [] # Store all calibration scores for p-value calc
        
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
                    out = model(inputs)
                    if isinstance(out, dict):
                        logits = out['mask_logits']
                        # Derive features via spatial pooling of mask logits
                        if logits.ndim == 5:
                            B_o, C_o, T_o, H_o, W_o = logits.shape
                            flat = logits.permute(0, 2, 1, 3, 4).reshape(B_o * T_o, C_o, H_o, W_o)
                            features = flat.mean(dim=(2, 3))
                        else:
                            features = logits.mean(dim=tuple(range(2, logits.ndim)))
                    elif isinstance(out, tuple):
                        if len(out) == 2:
                            logits, features = out
                        elif len(out) == 3:
                            logits, _, features = out
                        else:
                            logits = out[0]
                            features = out[-1]
                    else:
                        logits = out
                        features = getattr(model, 'features', None)
                        if features is None:
                            raise ValueError("Model must return features or have 'features' attribute for SelfAuditor.")
                except TypeError:
                    logits = model(inputs)
                    raise ValueError("SelfAuditor requires model to support feature extraction via return_features=True or similar mechanism.")

                # 1. Handle Feature Shape (B, D, T) -> (B*T, D)
                if features.ndim == 3:
                     # (B, D, T) -> (B, T, D) -> (B*T, D)
                     features = features.permute(0, 2, 1).reshape(-1, features.shape[1])
                
                # 2. Handle Logits Shape (B, C, T, H, W) -> (B*T, C, H, W)
                if logits.ndim == 5:
                     # (B, C, T, H, W) -> (B, T, C, H, W) -> (B*T, C, H, W)
                     B, C, T, H, W = logits.shape
                     logits = logits.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)

                # 3. Collect Features
                features_list.append(features.cpu())
                
                # 4. Collect Entropy
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
        
        # Store for P-Value Calculation
        self.calibration_scores = scores.numpy().tolist()
        
        # Set Epsilon
        self.epsilon = scores.mean().item()
        
        # Safety margin
        self.epsilon *= 1.05 
        
        logger.info(f"Calibration Complete. Epsilon: {self.epsilon:.4f}, Threshold: {self.threshold:.2f}, MaxEnt: {max_ent_sample:.4f}")

    def update(self, logits, features):
        """
        Step 2 & 3: Calculate Score and Update Martingale
        Args:
            logits: Current batch logits. Can be (1, C, T, H, W) or (B, C).
            features: Current batch features.
        Returns:
            stop_adaptation (bool): True if model should reset/stop
        """
        if self.collapsed:
            return True

        # Handle Temporal Dimension for Video (T > 1)
        # Assuming batch size 1 for TTA usually.
        # If logits is (1, C, T, H, W)
        
        logits_seq = [logits]
        features_seq = [features]
        
        is_temporal = False
        
        # Unpack if temporal video
        if isinstance(logits, torch.Tensor) and logits.ndim == 5:
            # (B, C, T, H, W) -> Unpack T frames
            # Assume B=1
            T = logits.shape[2]
            logits_seq = [logits[:, :, t, :, :] for t in range(T)]
            is_temporal = True
            
            # Features: (B, D, T) usually. If (B, D, T), unpack.
            if isinstance(features, torch.Tensor):
                if features.ndim == 3: # (B, D, T)
                     features_seq = [features[:, :, t] for t in range(T)]
                else:
                     features_seq = [features] * T

        elif isinstance(logits, tuple) and len(logits) == 2:
             # Hybrid Tuple: (EF_Head, Seg_Logits)
             ef_head, seg_head = logits
             
             # Calculate Geometric EF
             geom_ef, _, _ = self._compute_geometric_ef(seg_head)
             head_ef = ef_head.item() if ef_head.numel() == 1 else ef_head.mean().item()
             
             # Store for trace
             if 'ef_head' not in self.current_trace: self.current_trace['ef_head'] = []
             if 'ef_geom' not in self.current_trace: self.current_trace['ef_geom'] = []
             self.current_trace['ef_head'].append(head_ef)
             self.current_trace['ef_geom'].append(geom_ef)
             
             # Disagreement Check
             disagreement = abs(head_ef - geom_ef)
             
             if disagreement > 0.15: # 15% threshold
                 logger.warning(f"Auditor: Geometric Breakdown! Head={head_ef:.2f}, Geom={geom_ef:.2f}, Diff={disagreement:.2f}")
                 self.martingale *= 2.0 # Penalty
                 
             # Use Seg Head for entropy
             logits = seg_head
             
             if seg_head.ndim == 5:
                 T = seg_head.shape[2]
                 logits_seq = [seg_head[:, :, t, :, :] for t in range(T)]
                 is_temporal = True
             
             if isinstance(features, torch.Tensor):
                 if features.ndim == 3:
                     features_seq = [features[:, :, t] for t in range(T)]
                 else:
                     features_seq = [features] * T

        # Store history for this update call
        self.current_trace = {'martingale': [], 'p_value': []}

        for i, (log_t, feat_t) in enumerate(zip(logits_seq, features_seq)):
            # --- 1. Calculate Entropy Component ---
            entropy, max_ent = self._compute_entropy(log_t)
            norm_entropy = entropy / max_ent
    
            # --- 2. Calculate Drift Component ---
            # Normalize features
            feat_norm = F.normalize(feat_t, dim=1)
            # Cosine distance to source mean
            drift = 1 - torch.matmul(feat_norm, self.mu_source)
            
            # --- 3. Composite Score (s_t) ---
            # Average over the batch (B=1 usually)
            batch_score_t = self.alpha * norm_entropy.mean() + (1 - self.alpha) * drift.mean()
            
            # --- 4. Martingale Update (The Betting) ---
            # M_t = M_{t-1} * (1 + lambda * (s_t - epsilon))
            
            score_val = batch_score_t.item()
            self.scores.append(score_val)
            
            # Compute P-Value
            if self.calibration_scores:
                cal_scores = np.array(self.calibration_scores)
                p_val = (np.sum(cal_scores >= score_val) + 1) / (len(cal_scores) + 1)
                self.p_value = p_val
            else:
                self.p_value = 0.5
            
            bet = 1 + self.lambda_val * (score_val - self.epsilon)
            
            # Safety clipping: Wealth cannot be negative
            bet = max(0, bet) 
            
            self.martingale *= bet
            self.t += 1
            
            self.current_trace['martingale'].append(self.martingale)
            self.current_trace['p_value'].append(self.p_value)
            
            # --- 5. Rejection Rule ---
            if self.martingale >= self.threshold:
                self.collapsed = True
                logger.warning(f"SelfAuditor: Collapse detected at step {self.t} (Frame {i})! Martingale={self.martingale:.2f}")
                return True
            
        return False

    def reset_audit(self):
        """Resets the martingale (e.g. after a recovery intervention)"""
        self.martingale = 1.0
        self.collapsed = False
        self.p_value = 0.5
        self.current_trace = {'martingale': [], 'p_value': [], 'ef_head': [], 'ef_geom': []}
        logger.info("SelfAuditor: Martingale reset.")

    def _compute_geometric_ef(self, seg_logits):
        """
        Computes EF using Simpson's Rule (Method of Disks) from segmentation logits.
        Assumes standard Apical 4-Chamber view orientation (Apex at top/bottom).
        Input: (T, H, W) or (1, T, H, W)
        Returns: EF (0.0-1.0), EDV, ESV
        """
        if seg_logits.ndim == 4: # (1, T, H, W)
            seg_logits = seg_logits.squeeze(0)
            
        prob = torch.sigmoid(seg_logits)
        mask = (prob > 0.5).float() # (T, H, W)

        diameters = mask.sum(dim=2) # (T, H) - sum over Width -> diameter per row
        
        # Area of disks: pi * (d/2)^2 = (pi/4) * d^2
        disk_areas = (torch.pi / 4.0) * (diameters ** 2)
        
        volumes = disk_areas.sum(dim=1) # (T,) - Sum over Height -> Volume per frame
        
        if volumes.max() <= 1.0:
            return 0.0, 0.0, 0.0
            
        edv = volumes.max()
        esv = volumes.min()
        
        ef = (edv - esv) / (edv + 1e-6)
        return ef.item(), edv.item(), esv.item()
