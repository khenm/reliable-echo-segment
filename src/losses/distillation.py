import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.registry import register_loss
from src.utils.logging import get_logger

logger = get_logger()


def _isolate_and_load(hub_repo: str, model_name: str, **kwargs) -> nn.Module:
    """Load from torch.hub with namespace isolation."""
    # 1. Capture current state
    cwd = os.getcwd()
    original_sys_path = list(sys.path)
    
    # 2. Aggressively clean sys.path
    # Remove CWD, '', and the script directory if it matches CWD
    sys.path = [p for p in sys.path if p != '' and os.path.abspath(p) != cwd]
    
    # 3. Purge conflicting modules from sys.modules
    cached_src_modules = {}
    for key in list(sys.modules.keys()):
        if key == 'src' or key.startswith('src.'):
            cached_src_modules[key] = sys.modules.pop(key)
            
    try:
        # 4. Load from Hub (this will add the repo to sys.path)
        # verbose=False to reduce noise, assuming user trusts the repo
        model = torch.hub.load(hub_repo, model_name, force_reload=False, verbose=False, **kwargs)
        return model
    except Exception as e:
        logger.error(f"Failed to load {model_name} from {hub_repo}: {e}")
        raise e
    finally:
        # 5. Restore State
        # Restore sys.modules
        sys.modules.update(cached_src_modules)
        
        # Restore sys.path
        sys.path[:] = original_sys_path


@register_loss("PanEchoDistillation")
class PanEchoDistillationLoss(nn.Module):
    """
    Dual-head knowledge distillation from PanEcho.
    
    Architecture:
        - Frame-level: PanEcho image encoder (768-dim per frame) → segmentation supervision
        - Video-level: PanEcho full backbone (768-dim per video) → temporal supervision
        
    Loss = alpha * frame_loss + (1 - alpha) * video_loss
    """

    def __init__(
        self,
        student_dim: int = 256,
        teacher_dim: int = 768,
        temperature: float = 1.0,
        alpha: float = 0.1,
        clip_len: int = 16
    ):
        super().__init__()
        self.alpha = alpha
        self.clip_len = clip_len
        self.teacher_dim = teacher_dim
        # temperature is unused but kept for config compatibility

        # Use CosineEmbeddingLoss for feature alignment
        self.criterion = nn.CosineEmbeddingLoss()

        self.frame_projection = nn.Sequential(
            nn.Linear(student_dim, teacher_dim),
            nn.ReLU(),
            nn.Linear(teacher_dim, teacher_dim)
        )

        self.video_projection = nn.Sequential(
            nn.Linear(student_dim, teacher_dim),
            nn.ReLU(),
            nn.Linear(teacher_dim, teacher_dim)
        )

        self.image_encoder = None
        self.video_encoder = None
        self._device = None

        logger.info(
            f"PanEchoDistillationLoss initialized: "
            f"student_dim={student_dim}, alpha={alpha} (frame vs video), loss=CosineEmbedding"
        )

    def _ensure_teachers(self, device: torch.device):
        """Lazy load both teacher models."""
        if self.image_encoder is None:
            logger.info("Loading PanEcho image encoder (frame-level)...")
            self.image_encoder = _isolate_and_load(
                'CarDS-Yale/PanEcho', 'PanEcho', image_encoder_only=True
            )
            self.image_encoder.eval()
            for p in self.image_encoder.parameters():
                p.requires_grad = False

            logger.info("Loading PanEcho video backbone (video-level)...")
            self.video_encoder = _isolate_and_load(
                'CarDS-Yale/PanEcho', 'PanEcho', backbone_only=True, clip_len=self.clip_len
            )
            self.video_encoder.eval()
            for p in self.video_encoder.parameters():
                p.requires_grad = False

            self._device = device
            self.image_encoder.to(device)
            self.video_encoder.to(device)
            self.frame_projection.to(device)
            self.video_projection.to(device)
            logger.info("PanEcho teachers loaded and frozen")

        elif self._device != device:
            self.image_encoder.to(device)
            self.video_encoder.to(device)
            self.frame_projection.to(device)
            self.video_projection.to(device)
            self._device = device

    def forward(
        self,
        student_features: torch.Tensor,
        input_video: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute dual-head distillation loss.
        
        Args:
            student_features: (B, T, D_student) hidden states from student
            input_video: (B, C, T, H, W) original video
            
        Returns:
            loss: Combined frame + video distillation loss
        """
        device = student_features.device
        self._ensure_teachers(device)

        B, T, D = student_features.shape

        teacher_input = self._prepare_input(input_video)

        frame_loss = self._compute_frame_loss(student_features, teacher_input)

        video_loss = self._compute_video_loss(student_features, teacher_input)

        loss = self.alpha * frame_loss + (1 - self.alpha) * video_loss

        return loss

    def _prepare_input(self, video: torch.Tensor) -> torch.Tensor:
        """Resize to 224x224 and ensure 3 channels."""
        B, C, T, H, W = video.shape

        if C == 1:
            video = video.repeat(1, 3, 1, 1, 1)

        if H != 224 or W != 224:
            video_flat = video.permute(0, 2, 1, 3, 4).reshape(B * T, 3, H, W)
            video_resized = F.interpolate(
                video_flat, size=(224, 224), mode='bilinear', align_corners=False
            )
            video = video_resized.view(B, T, 3, 224, 224).permute(0, 2, 1, 3, 4)

        return video

    def _compute_frame_loss(
        self,
        student_features: torch.Tensor,
        teacher_input: torch.Tensor
    ) -> torch.Tensor:
        """Frame-level distillation using image encoder."""
        B, T, D = student_features.shape
        _, _, T_vid, H, W = teacher_input.shape

        frames = teacher_input.permute(0, 2, 1, 3, 4).reshape(B * T_vid, 3, H, W)

        with torch.no_grad():
            teacher_frame_features = self.image_encoder(frames)
            teacher_frame_features = teacher_frame_features.view(B, T_vid, -1)

        if T_vid != T:
            teacher_frame_features = F.interpolate(
                teacher_frame_features.permute(0, 2, 1),
                size=T,
                mode='linear',
                align_corners=False
            ).permute(0, 2, 1)

        student_proj = self.frame_projection(student_features)

        # Flatten for CosineEmbeddingLoss: (B*T, D)
        student_flat = student_proj.reshape(-1, self.teacher_dim)
        teacher_flat = teacher_frame_features.reshape(-1, self.teacher_dim)
        
        # Target is 1.0 (maximize similarity)
        target = torch.ones(student_flat.shape[0], device=student_flat.device)

        return self.criterion(student_flat, teacher_flat, target)

    def _compute_video_loss(
        self,
        student_features: torch.Tensor,
        teacher_input: torch.Tensor
    ) -> torch.Tensor:
        """Video-level distillation using full backbone."""
        B, T, D = student_features.shape
        _, C, T_vid, H, W = teacher_input.shape

        with torch.no_grad():
            if T_vid < self.clip_len:
                padded = F.pad(teacher_input, (0, 0, 0, 0, 0, self.clip_len - T_vid), mode='replicate')
            elif T_vid > self.clip_len:
                padded = teacher_input[:, :, :self.clip_len, :, :]
            else:
                padded = teacher_input

            teacher_video_embedding = self.video_encoder(padded)

        student_pooled = student_features.mean(dim=1)
        student_proj = self.video_projection(student_pooled)

        # Target is 1.0 (maximize similarity)
        target = torch.ones(B, device=student_proj.device)

        return self.criterion(student_proj, teacher_video_embedding, target)
