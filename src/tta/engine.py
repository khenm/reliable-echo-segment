import copy
import logging
from typing import Optional, Tuple, Any, Dict, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .augmentations import VideoAugmentor
from src.tta.auditor import SelfAuditor
from src.tta.conformal import ConformalCalibrator

logger = logging.getLogger(__name__)


class TTAEngine(nn.Module):
    """
    Test-Time Adaptation Engine using TENT (Test-time Entropy Minimization).
    
    Adapts the model on each test video by minimizing prediction entropy
    before generating the final prediction.
    """
    def __init__(self, 
                 model: nn.Module, 
                 lr: float = 1e-4, 
                 n_augments: int = 4, 
                 steps: int = 1, 
                 optimizer_name: str = "SGD"):
        super().__init__()
        self.model = model
        self.initial_state = copy.deepcopy(model.state_dict())
        
        self.lr = lr
        self.n_augments = n_augments
        self.steps = steps
        self.optimizer_name = optimizer_name
        
        self.augmentor = VideoAugmentor(hflip=True, shift_limit=0.1)
        
        self.configure_model_for_tent()
        self.reset_optimizer()

    def configure_model_for_tent(self) -> None:
        """
        Freezes all weights except BatchNorm/LayerNorm/GroupNorm affine parameters.
        Sets the model to train mode to update BN running stats.
        """
        self.model.train()
        self.model.requires_grad_(False)
        
        self.trainable_params = []
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm)):
                if m.weight is not None: 
                    m.weight.requires_grad = True
                    self.trainable_params.append(m.weight)
                if m.bias is not None: 
                    m.bias.requires_grad = True
                    self.trainable_params.append(m.bias)

    def reset_optimizer(self) -> None:
        """Re-initializes the optimizer."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.optimizer_name == "Adam":
            self.optimizer = optim.Adam(params, lr=self.lr)
        else:
            self.optimizer = optim.SGD(params, lr=self.lr, momentum=0.9)

    def reset(self) -> None:
        """Restore original weights after a video is processed."""
        self.model.load_state_dict(self.initial_state)
        self.reset_optimizer()
        self.configure_model_for_tent()

    def get_augmented_batch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create N augmented versions of input x.
        
        Args:
            x: Input tensor (1, C, T, H, W)
            
        Returns:
            Augmented batch (N, C, T, H, W)
        """
        x_repeated = x.repeat(self.n_augments, 1, 1, 1, 1)
        return self.augmentor(x_repeated)

    def forward_and_adapt(self, x_stream: torch.Tensor) -> torch.Tensor:
        """
        Adapt on x_stream and return final prediction.
        
        Args:
            x_stream: Input tensor (1, C, T, H, W)
        """
        for step in range(self.steps):
            aug_inputs = self.get_augmented_batch(x_stream)
            
            self.optimizer.zero_grad()
            outputs = self.model(aug_inputs)
            
            # Variance Consistency Loss for regression
            variance = torch.var(outputs)
            
            # Safety Check
            if step == 0 and variance > 100.0:
                logger.warning(f"Skipping TTA update: High initial variance {variance.item():.2f}")
                break

            variance.backward()
            nn.utils.clip_grad_norm_(self.trainable_params, max_norm=1.0)
            self.optimizer.step()
        
        with torch.no_grad():
            self.model.eval()
            final_pred = self.model(x_stream)
            self.model.train()
            
        return final_pred


class SafeTTAEngine:
    """
    Implements Safe-TTA:
    1. Audited Forward Pass (Drift/Entropy Check)
    2. Conformal Prediction (Prediction Sets)
    3. Conditional Adaptation
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
        self.optimizer_name = optimizer_name
        
        self.initial_state = copy.deepcopy(model.state_dict())
        self.optimizer = None
        self.configure_optimization()

    def configure_optimization(self) -> None:
        """Sets up optimizer for the adaptation phase."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            return

        if self.optimizer_name == "Adam":
            self.optimizer = optim.Adam(params, lr=self.lr)
        else:
            self.optimizer = optim.SGD(params, lr=self.lr, momentum=0.9)

    def reset(self) -> None:
        """Resets model to initial state and resets auditor."""
        self.model.load_state_dict(self.initial_state)
        self.auditor.reset_audit()
        self.configure_optimization()

    def run_calibration(self, val_loader: torch.utils.data.DataLoader, device: torch.device) -> None:
        """Runs the offline calibration for Auditor and Conformal Calibrator."""
        logger.info("Running Safe-TTA Calibration Phase...")
        self.auditor.calibrate(val_loader, self.model, device)
        self.calibrator.calibrate(val_loader, self.model, device)

    def predict_step(self, x_t: torch.Tensor, adapt: bool = True) -> Tuple[Any, float, bool, Dict]:
        """
        Processes a single test batch x_t.
        
        Returns:
            prediction_sets: List of sets
            q_val: Quantile used
            collapsed: Boolean indicating if model collapsed
            audit_stats: Trace stats
        """
        # 1. Forward Pass & Audit
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(x_t)
            calib_logits, audit_logits, features = self._extract_model_outputs(outputs)
            
            is_collapsed = self.auditor.update(audit_logits, features)
            
            if is_collapsed:
                logger.warning("SafeTTA: Collapse detected. Outputting max uncertainty.")
                self.reset()
                return None, 1.0, True, self._get_audit_stats()
            
            # 2. Conformal Output
            audit_score = self.auditor.scores[-1] if self.auditor.scores else None
            final_output, q_used = self.calibrator.predict(
                calib_logits, 
                martingale_stable=True,
                audit_score=audit_score,
                audit_epsilon=self.auditor.epsilon
            )
            
        # 3. Adaptation
        if adapt and self.optimizer and not is_collapsed:
            self._adapt_model(x_t)

        return final_output, q_used, is_collapsed, self._get_audit_stats()

    def _extract_model_outputs(self, out: Any) -> Tuple[Any, Any, Any]:
        """Extracts (calib_logits, audit_logits, features) from varied model outputs."""
        calib_logits = out
        audit_logits = out
        features = None
        
        if isinstance(out, dict):
            audit_logits = out.get('mask_logits', out)
            # Feature extraction from 5D logits logic
            if isinstance(audit_logits, torch.Tensor) and audit_logits.ndim == 5:
                B, C, T, H, W = audit_logits.shape
                flat = audit_logits.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
                features = flat.mean(dim=(2, 3))
            elif isinstance(audit_logits, torch.Tensor):
                features = audit_logits.mean(dim=tuple(range(2, audit_logits.ndim)))
                
        elif isinstance(out, tuple):
            audit_logits = out
            if len(out) == 3:
                # (val1, val2, features) pattern
                val1, val2, feat = out
                features = feat
                audit_logits = (val2, val1) if val1.ndim >= 4 else (val1, val2)
            elif len(out) == 2:
                # (logits, features) pattern
                audit_logits, features = out
            else:
                audit_logits = out[0]
                features = out[-1]
                
        else:
            # Fallback to model.features if stored
            features = getattr(self.model, 'features', getattr(self.model, 'last_features', None))
            if features is None:
                 features = torch.zeros_like(out) if isinstance(out, torch.Tensor) else None

        return calib_logits, audit_logits, features

    def _adapt_model(self, x_t: torch.Tensor) -> None:
        """Performs one step of adaptation on the test input."""
        self.model.train()
        self.optimizer.zero_grad()
        
        out_train = self.model(x_t)
        logits_seg, logits_ef = self._parse_train_outputs(out_train)
        
        loss = torch.tensor(0.0, device=x_t.device, requires_grad=True)
        # Re-assigning to use zero-tensor as base to ensure graph connectivity if needed, though usually backward works on components.
        # But wait, if I init loss=0.0 (float), I can't backward.
        
        current_loss = 0.0

        if logits_ef is not None:
             current_loss = current_loss + self._compute_binary_entropy(logits_ef)
            
        if logits_seg is not None:
             if logits_seg.shape[1] > 1:
                 current_loss = current_loss + self._compute_categorical_entropy(logits_seg)
             else:
                 current_loss = current_loss + self._compute_binary_entropy(logits_seg)
        
        if isinstance(current_loss, torch.Tensor) and current_loss.item() > 0: # Check if tensor has value
            current_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
        self.model.eval()

    def _parse_train_outputs(self, out: Any) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Parses model output to find segmentation and EF logits."""
        logits_seg = None
        logits_ef = None

        if isinstance(out, dict):
            logits_seg = out.get('mask_logits')
            logits_ef = out.get('pred_ef')
        elif isinstance(out, tuple):
            for item in out:
                if isinstance(item, torch.Tensor):
                    if item.ndim >= 4:
                        logits_seg = item
                    elif item.ndim == 2:
                        logits_ef = item
            # Fallback
            if logits_seg is None and logits_ef is None and len(out) >= 2:
                logits_ef = out[0]
                logits_seg = out[1]
        else:
            # Single tensor assumption
            logits_seg = out
            
        return logits_seg, logits_ef

    def _compute_binary_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Computes mean entropy for binary logits."""
        p = torch.sigmoid(logits) if logits.shape[1] == 1 else logits
        # Ensure p is in range to avoid nan
        p = torch.clamp(p, 1e-7, 1.0 - 1e-7)
        return -(p * torch.log(p) + (1 - p) * torch.log(1 - p)).mean()

    def _compute_categorical_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Computes mean entropy for categorical logits."""
        probs = F.softmax(logits, dim=1)
        probs = torch.clamp(probs, 1e-7, 1.0 - 1e-7)
        return -torch.sum(probs * torch.log(probs), dim=1).mean()

    def _get_audit_stats(self) -> Dict[str, Any]:
        """Returns current auditor statistics."""
        return getattr(self.auditor, 'current_trace', {
            'martingale': [self.auditor.martingale],
            'p_value': [self.auditor.p_value]
        })
