import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss

from src.registry import register_loss
from src.utils.logging import get_logger

logger = get_logger()

class PolarFocalVolumeLoss(nn.Module):
    """
    A difficulty-aware orthogonal loss function. 
    Projects EDV/ESV into polar space (Magnitude/Angle) and applies independent 
    focal difficulty weighting to both the scale and the physiological ratio.
    """
    def __init__(
        self, 
        gamma: float = 2.0,
        scale_weight: float = 1.0, 
        ratio_weight: float = 10.0, 
        clip_threshold: float = 0.5,
        eps: float = 1e-7
    ):
        super().__init__()
        self.gamma = gamma
        self.scale_weight = scale_weight
        self.ratio_weight = ratio_weight
        self.clip_threshold = clip_threshold
        self.eps = eps

    def forward(
        self, 
        pred_edv: torch.Tensor, 
        pred_esv: torch.Tensor, 
        target_edv: torch.Tensor, 
        target_esv: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        
        # 1. Flatten inputs and filter for valid labeled frames
        p_edv, p_esv = pred_edv.view(-1), pred_esv.view(-1)
        t_edv, t_esv = target_edv.view(-1), target_esv.view(-1)
        
        valid = (t_edv >= 0) & (t_esv >= 0)
        
        if not valid.any():
            dummy_loss = 0.0 * p_edv.sum()
            return dummy_loss, {"scale_loss": dummy_loss.detach(), "ratio_loss": dummy_loss.detach()}

        # 2. Project into 2D Polar Space
        pred_vec = torch.stack([p_edv[valid], p_esv[valid]], dim=-1)
        target_vec = torch.stack([t_edv[valid], t_esv[valid]], dim=-1)

        # --- SCALE (MAGNITUDE) CALCULATION ---
        pred_mag = torch.norm(pred_vec, p=2, dim=-1)
        target_mag = torch.norm(target_vec, p=2, dim=-1)
        
        # Calculate base scale loss (Huber)
        base_loss_scale = F.huber_loss(pred_mag, target_mag, reduction='none', delta=self.clip_threshold)
        
        # Calculate dynamic focal weight for scale (Relative Error)
        error_scale = torch.abs(pred_mag - target_mag)
        relative_error_scale = error_scale / (target_mag + self.eps)
        p_scale = torch.clamp(relative_error_scale, min=0.0, max=1.0)
        weight_scale = (1.0 + torch.pow(p_scale, self.gamma)).detach() # Detach is critical!
        
        # Apply weight
        focal_loss_scale = (weight_scale * base_loss_scale).mean()


        # --- RATIO (ANGLE) CALCULATION ---
        cos_sim = F.cosine_similarity(pred_vec, target_vec, dim=-1, eps=self.eps)
        
        # Calculate base ratio loss (1 - Cosine Similarity)
        # Bounded between 0.0 (perfect) and 2.0 (opposite)
        base_loss_ratio = 1.0 - cos_sim
        
        # Calculate dynamic focal weight for ratio
        # We clamp at 1.0 to prevent the multiplier from exceeding (1 + 1^gamma) = 2.0
        p_ratio = torch.clamp(base_loss_ratio, min=0.0, max=1.0)
        weight_ratio = (1.0 + torch.pow(p_ratio, self.gamma)).detach()
        
        # Apply weight
        focal_loss_ratio = (weight_ratio * base_loss_ratio).mean()


        # --- COMBINE ORTHOGONAL LOSSES ---
        total_loss = (self.scale_weight * focal_loss_scale) + (self.ratio_weight * focal_loss_ratio)

        return total_loss, {
            "scale_loss": focal_loss_scale.detach(),
            "ratio_loss": focal_loss_ratio.detach()
        }

@register_loss("TemporalWeakSegLoss")
class TemporalWeakSegLoss(nn.Module):
    """
    Computes spatial and volumetric regression losses for echocardiography video segments.
    Focuses strictly on mask-guided spatial grounding and direct volume regression.
    """
    def __init__(
        self,
        dice_weight: float = 1.0,
        volume_weight: float = 1.0,
        phase_weight: float = 0.5,
        gamma: float = 2.0,
        focal_clip_threshold: float = 0.5,
        focal_scale_weight: float = 1.0,
        focal_ratio_weight: float = 10.0,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.volume_weight = volume_weight
        self.phase_weight = phase_weight
        
        self.dice_func = DiceCELoss(sigmoid=True, reduction='mean')
        
        self.vol_loss_func = PolarFocalVolumeLoss(
            gamma=gamma, 
            scale_weight=focal_scale_weight,
            ratio_weight=focal_ratio_weight,
            clip_threshold=focal_clip_threshold
        )

    def forward(
        self,
        pred_logits: torch.Tensor,
        target_masks: torch.Tensor,
        frame_mask: torch.Tensor,
        target_edv: torch.Tensor,
        target_esv: torch.Tensor,
        pred_edv: torch.Tensor,
        pred_esv: torch.Tensor,
        **kwargs
    ):
        """
        Accepts **kwargs to gracefully handle legacy pipeline arguments (e.g., pred_vol_curve)
        during the architectural transition without breaking the forward pass.
        """
        loss_dice = self._compute_dice_loss(pred_logits, target_masks, frame_mask)
        
        pred_vol_curve = kwargs.get("pred_vol_curve", None)
        pred_phase_logits = kwargs.get("pred_phase_logits", None)
        
        loss_phase = torch.tensor(0.0, device=pred_logits.device)
        
        # Always use the state-gated volumes directly
        loss_vol, vol_loss_dict = self.vol_loss_func(
            pred_edv, pred_esv, target_edv, target_esv
        )
            
        if pred_phase_logits is not None:
            loss_phase = self._compute_phase_loss(pred_phase_logits, frame_mask)

        total_loss = (self.dice_weight * loss_dice) + (self.volume_weight * loss_vol) + (self.phase_weight * loss_phase)

        loss_dict = {
            "dice_loss": loss_dice,
            "volume_loss": loss_vol,
            "phase_loss": loss_phase,
        }
        
        # Add volume sub-components to loss dict for logging
        if vol_loss_dict:
            loss_dict.update(vol_loss_dict)

        return total_loss, loss_dict

    def _compute_dice_loss(
        self,
        pred_logits: torch.Tensor,
        target_masks: torch.Tensor,
        frame_mask: torch.Tensor
    ) -> torch.Tensor:
        """Vectorized Dice+CE on valid labeled frames."""
        if pred_logits.shape[1] == 1 and pred_logits.shape[2] > 1:
            pred_logits = pred_logits.permute(0, 2, 1, 3, 4)

        if target_masks.shape[1] == 1 and target_masks.shape[2] > 1:
            target_masks = target_masks.permute(0, 2, 1, 3, 4)

        batch_size, seq_len, channels, height, width = pred_logits.shape

        pred_flat = pred_logits.reshape(-1, channels, height, width)
        target_flat = target_masks.reshape(-1, channels, height, width)
        mask_flat = frame_mask.reshape(-1)

        valid_indices = mask_flat > 0.5

        if valid_indices.sum() == 0:
            return 0.0 * pred_logits.sum()

        return self.dice_func(
            pred_flat[valid_indices],
            target_flat[valid_indices]
        )

    def _compute_phase_loss(self, pred_phase_logits: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        """
        Computes Cross-Entropy loss for predicted cyclic phase (0=Systole, 1=Diastole)
        by inferring ground truth implicitly from ED/ES frames.
        """
        # pred_phase_logits: (B, T, 2)
        B, T, _ = pred_phase_logits.shape
        loss_fn = nn.CrossEntropyLoss()
        total_phase_loss = 0.0
        valid_batches = 0
        
        for b in range(B):
            ed_idx = torch.where(frame_mask[b] == 2.0)[0]
            es_idx = torch.where(frame_mask[b] == 1.0)[0]
            
            if len(ed_idx) > 0 and len(es_idx) > 0:
                ed = ed_idx[0].item()
                es = es_idx[0].item()
                
                target_phases = torch.zeros(T, dtype=torch.long, device=pred_phase_logits.device)
                
                if ed < es:
                    target_phases[:ed+1] = 1
                    target_phases[ed+1:es+1] = 0
                    target_phases[es+1:] = 1
                else:
                    target_phases[:es+1] = 0
                    target_phases[es+1:ed+1] = 1
                    target_phases[ed+1:] = 0
                    
                total_phase_loss += loss_fn(pred_phase_logits[b], target_phases)
                valid_batches += 1
                
        if valid_batches > 0:
            return total_phase_loss / valid_batches
        return torch.tensor(0.0, device=pred_phase_logits.device, requires_grad=True)
