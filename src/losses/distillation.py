import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.registry import register_loss
from src.utils.logging import get_logger

logger = get_logger()


def _load_panecho_teacher(clip_len: int = 16) -> nn.Module:
    """
    Load PanEcho backbone with proper namespace isolation.
    
    Handles collision between local 'src' package and PanEcho's 'src' package
    by temporarily clearing src.* modules from sys.modules cache.
    """
    cwd = os.getcwd()

    if cwd in sys.path:
        sys.path.remove(cwd)

    cached_src_modules = {
        key: mod for key, mod in sys.modules.items()
        if key == 'src' or key.startswith('src.')
    }
    for key in cached_src_modules:
        del sys.modules[key]

    try:
        model = torch.hub.load(
            'CarDS-Yale/PanEcho', 'PanEcho',
            force_reload=False,
            backbone_only=True,
            clip_len=clip_len
        )
    finally:
        sys.modules.update(cached_src_modules)
        if cwd not in sys.path:
            sys.path.insert(0, cwd)

    return model


@register_loss("PanEchoDistillation")
class PanEchoDistillationLoss(nn.Module):
    """
    Frame-wise knowledge distillation from PanEcho teacher.
    
    Architecture:
        - Teacher: PanEcho backbone (frozen) producing 768-dim embeddings per frame
        - Student: Projects hidden states (256-dim) to match teacher space
        - Loss: MSE on L2-normalized features for each frame
    """

    def __init__(
        self,
        student_dim: int = 256,
        teacher_dim: int = 768,
        temperature: float = 1.0,
        clip_len: int = 16
    ):
        super().__init__()
        self.temperature = temperature
        self.clip_len = clip_len
        self.teacher_dim = teacher_dim

        self.projection = nn.Sequential(
            nn.Linear(student_dim, teacher_dim),
            nn.ReLU(),
            nn.Linear(teacher_dim, teacher_dim)
        )

        self.teacher = None
        self._teacher_device = None

        logger.info(
            f"PanEchoDistillationLoss initialized: "
            f"student_dim={student_dim}, teacher_dim={teacher_dim}"
        )

    def _ensure_teacher(self, device: torch.device):
        """Lazy load teacher on first forward pass."""
        if self.teacher is None:
            logger.info("Loading PanEcho teacher model...")
            self.teacher = _load_panecho_teacher(clip_len=self.clip_len)
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False
            self._teacher_device = device
            self.teacher.to(device)
            logger.info("PanEcho teacher loaded and frozen")
        elif self._teacher_device != device:
            self.teacher.to(device)
            self._teacher_device = device

    def forward(
        self,
        student_features: torch.Tensor,
        input_video: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute frame-wise distillation loss.
        
        Args:
            student_features: (B, T, D_student) hidden states from student GRU
            input_video: (B, C, T, H, W) original video input
            
        Returns:
            loss: Scalar distillation loss
        """
        device = student_features.device
        self._ensure_teacher(device)

        B, T_student, D = student_features.shape
        B_vid, C, T_vid, H, W = input_video.shape

        teacher_input = self._prepare_teacher_input(input_video)
        teacher_features = self._get_frame_features(teacher_input)

        if teacher_features.shape[1] != T_student:
            teacher_features = F.interpolate(
                teacher_features.permute(0, 2, 1),
                size=T_student,
                mode='linear',
                align_corners=False
            ).permute(0, 2, 1)

        student_proj = self.projection(student_features)

        student_norm = F.normalize(student_proj, dim=-1)
        teacher_norm = F.normalize(teacher_features, dim=-1)

        loss = F.mse_loss(student_norm / self.temperature, teacher_norm / self.temperature)

        return loss

    def _prepare_teacher_input(self, video: torch.Tensor) -> torch.Tensor:
        """Resize video from student size (112) to teacher size (224)."""
        B, C, T, H, W = video.shape

        if C == 1:
            video = video.repeat(1, 3, 1, 1, 1)

        if H != 224 or W != 224:
            video_flat = video.permute(0, 2, 1, 3, 4).reshape(B * T, 3, H, W)
            video_resized = F.interpolate(video_flat, size=(224, 224), mode='bilinear', align_corners=False)
            video = video_resized.view(B, T, 3, 224, 224).permute(0, 2, 1, 3, 4)

        return video

    def _get_frame_features(self, video: torch.Tensor) -> torch.Tensor:
        """
        Extract per-frame features from PanEcho teacher.
        
        PanEcho backbone outputs (B, 768) for the entire clip.
        We extract features for each frame by sliding window.
        """
        B, C, T, H, W = video.shape

        with torch.no_grad():
            if T <= self.clip_len:
                padded = F.pad(video, (0, 0, 0, 0, 0, self.clip_len - T), mode='replicate')
                embedding = self.teacher(padded)
                frame_features = embedding.unsqueeze(1).expand(B, T, -1)
            else:
                frame_features_list = []
                for t in range(T):
                    start = max(0, t - self.clip_len // 2)
                    end = min(T, start + self.clip_len)
                    if end - start < self.clip_len:
                        start = max(0, end - self.clip_len)

                    clip = video[:, :, start:end, :, :]
                    if clip.shape[2] < self.clip_len:
                        clip = F.pad(clip, (0, 0, 0, 0, 0, self.clip_len - clip.shape[2]), mode='replicate')

                    embedding = self.teacher(clip)
                    frame_features_list.append(embedding)

                frame_features = torch.stack(frame_features_list, dim=1)

        return frame_features
