import torch
import torch.nn as nn
import torch.optim as optim
import copy
import logging
from .augmentations import VideoAugmentor
from src.tta.auditor import SelfAuditor
from src.tta.conformal import ConformalCalibrator
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class TTA_Engine(nn.Module):
    def __init__(self, model, lr=1e-4, n_augments=4, steps=1, optimizer_name="SGD"):
        super().__init__()
        self.model = model
        # Save initial state to reset after each video
        self.initial_state = copy.deepcopy(model.state_dict())
        
        self.lr = lr
        self.n_augments = n_augments
        self.steps = steps
        self.optimizer_name = optimizer_name
        
        self.augmentor = VideoAugmentor(hflip=True, shift_limit=0.1)
        
        # Configure model for Tent (Freeze non-norm layers)
        self.configure_model_for_tent()
        
        # Initialize optimizer
        self.reset_optimizer()

    def configure_model_for_tent(self):
        """
        Freezes all weights except BatchNorm/LayerNorm/GroupNorm affine parameters.
        Sets the model to train mode to update BN running stats.
        """
        self.model.train() # Enable BN stat updates
        self.model.requires_grad_(False) # Default freeze
        
        trainable_params = []
        for name, m in self.model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm)):
                if m.weight is not None: 
                    m.weight.requires_grad = True
                    trainable_params.append(m.weight)
                if m.bias is not None: 
                    m.bias.requires_grad = True
                    trainable_params.append(m.bias)
        
        self.trainable_params = trainable_params
        # logger.info(f"TTA Configured: {len(trainable_params)} parameters trainable (Norm layers).")

    def reset_optimizer(self):
        """Re-initializes the optimizer."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.optimizer_name == "SGD":
            self.optimizer = optim.SGD(params, lr=self.lr, momentum=0.9)
        elif self.optimizer_name == "Adam":
            self.optimizer = optim.Adam(params, lr=self.lr)
        else:
            self.optimizer = optim.SGD(params, lr=self.lr, momentum=0.9)

    def reset(self):
        """Restore original weights after a video is processed."""
        self.model.load_state_dict(self.initial_state)
        # Re-configure because load_state_dict might reset requires_grad flags depending on implementation,
        # but usually it doesn't affect requires_grad. However, we need to reset the optimizer state.
        self.reset_optimizer()
        # Ensure mode is correct
        self.configure_model_for_tent()

    def get_augmented_batch(self, x):
        """
        Create N augmented versions of input x.
        x: (1, C, T, H, W)
        Returns: (N, C, T, H, W)
        """
        # x is usually batch size 1 for inference
        x = x.repeat(self.n_augments, 1, 1, 1, 1) # (N, C, T, H, W)
        
        # Apply augmentations individually or in batch
        # Our augmentor handles batch
        x_aug = self.augmentor(x)
        return x_aug

    def forward_and_adapt(self, x_stream):
        """
        Adapt on x_stream and return final prediction.
        x_stream: (1, C, T, H, W)
        """
        # 1. Adapt Loop
        for step in range(self.steps):
            # Generate Augmented Views
            aug_inputs = self.get_augmented_batch(x_stream) # (N, ...)
            
            self.optimizer.zero_grad()
            outputs = self.model(aug_inputs) # (N, 1)
            
            # Variance Consistency Loss
            # specific for regression: minimize variance of predictions across augs
            # outputs shape: (N, 1)
            variance = torch.var(outputs)
            
            # Safety Check: If variance is extremely high initially, might be garbage input.
            # Skip update to avoid destroying weights.
            if step == 0 and variance > 100.0: # Arbitrary threshold, tune based on EF scale (0-100)
                logger.warning(f"Skipping TTA update: High initial variance {variance.item():.2f}")
                break

            loss = variance
            loss.backward()
            
            # Gradient Clipping (Safety)
            nn.utils.clip_grad_norm_(self.trainable_params, max_norm=1.0)
            
            self.optimizer.step()
        
        # 2. Final Inference (using updated weights)
        # Use simple test-time augmentation averaging for final prediction as well,
        # or just single forward pass on clean data. Standard TTA uses CLEAN data for final pred.
        with torch.no_grad():
            self.model.eval() # Use eval mode for final prediction to use learned stats
            final_pred = self.model(x_stream)
            
            # Switch back to train for next potential step (though we reset after video usually)
            self.model.train() 
            
        return final_pred


class SafeTTAEngine:
    """
    Implements the Safe-TTA Loop:
    1. Forward Pass
    2. Phase 3: Self-Auditor Check (Drift/Entropy)
    3. Phase 4: Conformal Prediction (Prediction Sets)
    4. Adaptation (Optional/Conditional)
    """
    def __init__(self, 
                 model: nn.Module, 
                 auditor: SelfAuditor, 
                 calibrator: ConformalCalibrator,
                 optimizer_name: str = "SGD",
                 lr: float = 1e-4):
        self.model = model
        self.auditor = auditor
        self.calibrator = calibrator
        self.lr = lr
        
        # Save initial state
        self.initial_state = copy.deepcopy(model.state_dict())
        self.optimizer_name = optimizer_name
        self.configure_optimization()

    def configure_optimization(self):
        """Sets up optimizer for the adaptation phase."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            # If model frozen, maybe unfreeze specific layers or warn
            # For now, assume model came pre-configured or we don't adapt
            self.optimizer = None
            return

        if self.optimizer_name == "SGD":
            self.optimizer = optim.SGD(params, lr=self.lr, momentum=0.9)
        elif self.optimizer_name == "Adam":
            self.optimizer = optim.Adam(params, lr=self.lr)
        else:
             self.optimizer = optim.SGD(params, lr=self.lr)

    def reset(self):
        """Resets model to initial state and resets auditor."""
        self.model.load_state_dict(self.initial_state)
        self.auditor.reset_audit()
        self.configure_optimization()

    def run_calibration(self, val_loader, device):
        """Runs the offline calibration for both Auditor and Conformal Calibrator."""
        print(f"DEBUG: Engine run_calibration called. Val Len: {len(val_loader)}")
        logger.info("Running Safe-TTA Calibration Phase...")
        self.auditor.calibrate(val_loader, self.model, device)
        self.calibrator.calibrate(val_loader, self.model, device)

    def predict_step(self, x_t, adapt=True):
        """
        Processes a single test batch x_t.
        Returns:
            prediction_sets: List of sets
            q_val: Quantile used
            collapsed: Boolean indicating if model collapsed
        """
        # 1. Forward Pass & Audit (No Grad)
        with torch.no_grad():
            self.model.eval() # Eval for feature extraction initially
            
            # Forward Pass (Need logits and features)
            try:
                out = self.model(x_t, return_features=True)
                if isinstance(out, tuple):
                    if len(out) == 3:
                         val1 = out[0]
                         val2 = out[1]
                         features = out[2]
                         
                         if val1.ndim >= 4: 
                             # (seg, ef) -> Swap for AuditCalibrator (expects ef, seg)
                             logits = (val2, val1)
                         else:
                             # (ef, seg) -> Keep
                             logits = (val1, val2)
                             
                    elif len(out) == 2:
                         logits, features = out
                    else:
                         logits = out[0]
                         features = out[-1]
                else:
                    logits = out
                    features = getattr(self.model, 'features', None)
                    if features is None:
                        raise RuntimeError("Model features not found.")
            except TypeError:
                logits = self.model(x_t)
                features = getattr(self.model, 'last_features', torch.zeros_like(logits))

            # 2. Phase 3: Audit
            # Auditor handles tuple unpacking internally to compute entropy from the most relevant output (e.g. segmentation)
            is_collapsed = self.auditor.update(logits, features)
            
            final_output = None
            q_used = 1.0
            
            if is_collapsed:
                logger.warning("SafeTTA: Collapse detected. Outputting max uncertainty.")
                # Recovery: Reset model
                self.model.load_state_dict(self.initial_state)
                self.auditor.reset_audit()
                
                # final_output remains None, main.py should handle it, but return stats for plotting
                audit_stats = getattr(self.auditor, 'current_trace', {
                    'martingale': [self.auditor.martingale],
                    'p_value': [self.auditor.p_value]
                })
                return None, 1.0, True, audit_stats
                
            else:
                # 3. Phase 4: Conformal Output
                # Calibrator (including AuditCalibrator) handles tuple logits
                
                # Get latest audit score and epsilon
                audit_score = self.auditor.scores[-1] if self.auditor.scores else None
                audit_epsilon = self.auditor.epsilon
                
                print(f"DEBUG: Engine Predict - AuditScore={audit_score}, Epsilon={audit_epsilon}")

                final_output, q_used = self.calibrator.predict(logits, 
                                                               martingale_stable=True,
                                                               audit_score=audit_score,
                                                               audit_epsilon=audit_epsilon)
            
        # Explicit clean up to prevent double-graph retention
        del out, logits, features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 4. Phase 2: Adaptation (Optional)
        # Adapt only if not collapsed and optimizer is set
        if adapt and self.optimizer and not is_collapsed:
            self.model.train()
            self.optimizer.zero_grad()
            
            # Re-forward for gradients
            out_train = self.model(x_t)
            
            loss = 0.0
            if isinstance(out_train, tuple):
                logits_seg = None
                logits_ef = None
                
                for item in out_train:
                    if isinstance(item, torch.Tensor):
                        if item.ndim >= 4:
                            logits_seg = item
                        elif item.ndim == 2:
                            logits_ef = item
                
                # Fallback if identification failed but tuple exists (legacy R2Plus1D assumption)
                if logits_seg is None and logits_ef is None:
                     logits_ef = out_train[0]
                     logits_seg = out_train[1]
                
                # 1. EF Entropy
                if logits_ef is not None:
                    p_ef = logits_ef
                    if p_ef.shape[1] == 1:
                        p = torch.sigmoid(p_ef)
                        ent_ef = -(p * torch.log(p + 1e-10) + (1-p) * torch.log(1-p + 1e-10)).mean()
                        loss += ent_ef

                # 2. Seg Entropy
                if logits_seg is not None:
                    probs_seg = F.softmax(logits_seg, dim=1) if logits_seg.shape[1] > 1 else torch.sigmoid(logits_seg)
                    if logits_seg.shape[1] > 1:
                         ent_seg = -torch.sum(probs_seg * torch.log(probs_seg + 1e-10), dim=1).mean()
                    else:
                         # Binary Seg
                         p = probs_seg
                         ent_seg = -(p * torch.log(p + 1e-10) + (1-p) * torch.log(1-p + 1e-10)).mean()
                    
                    loss += ent_seg
            else:
                logits_train = out_train
                # Handle Binary Case (B, 1) - explicit entropy calculation
                if logits_train.shape[1] == 1:
                    p = logits_train
                    loss = -torch.sum(p * torch.log(p + 1e-10) + (1-p) * torch.log(1-p + 1e-10), dim=1).mean()
                else:
                    probs = F.softmax(logits_train, dim=1)
                    loss = -torch.sum(probs * torch.log(probs + 1e-10), dim=1).mean()
            
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) # Safety clip
            self.optimizer.step()
            self.model.eval()

        return final_output, q_used, is_collapsed, getattr(self.auditor, 'current_trace', {
            'martingale': [self.auditor.martingale],
            'p_value': [self.auditor.p_value]
        })
