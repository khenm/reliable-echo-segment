import torch
import torch.nn.functional as F
import logging
import numpy as np
from src.tta.augmentations import GeometryAligner, SpectralNormalizer

logger = logging.getLogger(__name__)

class AdaptiveGatekeeper:
    """
    Manages the 'State' of the patient input stream.
    Acts as the Gatekeeper that normalizes Geometry and Texture based on:
    1. Offline 'Golden' Reference (Initial State)
    2. Online 'Cloud' Feedback (Updated State)
    """
    def __init__(self, golden_landmarks=None, golden_amplitude=None, momentum=0.1):
        """
        Args:
            golden_landmarks (torch.Tensor): (N, 2) average keypoints.
            golden_amplitude (torch.Tensor): (C, H, W) average amplitude.
            momentum (float): Update momentum (0.0 = Replace, 1.0 = Keep old). 
                              User specified 0.1 as smoothing factor 'm'.
                              Formula: current = (1-m)*current + m*new
        """
        self.golden_landmarks = golden_landmarks
        self.golden_amplitude = golden_amplitude
        self.momentum = momentum
        
        # State Variables
        self.current_landmarks = None # Initialized to None as per plan
        if golden_amplitude is not None:
             self.current_amplitude = golden_amplitude.clone()
        else:
             self.current_amplitude = None
        
        # Tools
        self.geo_aligner = GeometryAligner()
        self.spec_normalizer = SpectralNormalizer()
        
    def process_input(self, frame):
        """
        Adapts the raw input frame using current state.
        Args:
            frame: (1, C, T, H, W) or (1, C, H, W)
        Returns:
            aligned_frame: Adapted tensor.
        """
        if self.current_amplitude is None and self.golden_amplitude is not None:
             self.current_amplitude = self.golden_amplitude.clone()

        # 1. Geometry Alignment
        # Only align if we have determined 'current_landmarks' (i.e. we know where the patient IS)
        # And we have golden landmarks (where they SHOULD BE).
        if self.current_landmarks is not None and self.golden_landmarks is not None:
            # We want to warp Frame (at Current) -> Golden
            # So dst = Golden, src = Current
            # M maps Current -> Golden
            
            B = frame.shape[0]
            # Ensure proper device/dtype
            curr_land = self.current_landmarks.to(frame.device).type(frame.dtype)
            gold_land = self.golden_landmarks.to(frame.device).type(frame.dtype)
            
            src_pts = curr_land.unsqueeze(0).repeat(B, 1, 1) # (B, N, 2)
            dst_pts = gold_land.unsqueeze(0).repeat(B, 1, 1) # (B, N, 2)
            
            M = self.geo_aligner.estimate_affine_matrix(src_pts, dst_pts)
            frame = self.geo_aligner.warp(frame, M)

        # 2. Spectral Normalization (Texture)
        if self.current_amplitude is not None:
            target_amp = self.current_amplitude.to(frame.device)
            
            # Create proper shape for broadcasting
            if frame.ndim == 5:
                # Target: (1, C, 1, H, W)
                # target_amp is (C, H, W) -> (1, C, 1, H, W)
                target_amp = target_amp.unsqueeze(0).unsqueeze(2)
            elif frame.ndim == 4:
                 target_amp = target_amp.unsqueeze(0)
                 
            frame = self.spec_normalizer.inject_style(frame, target_amp)
            
        return frame

    def update_state(self, cloud_feedback):
        """
        Updates state based on Ground Truth from Cloud.
        Args:
            cloud_feedback (dict): Contains 'keypoints', 'video'/'image' (for amplitude extraction).
        """
        # 1. Update Landmarks
        if 'keypoints' in cloud_feedback:
             cloud_kps = cloud_feedback['keypoints'] # (B, N, 2) or (N, 2)
             if cloud_kps.ndim == 3: 
                 cloud_kps = cloud_kps.mean(dim=0) # Average over batch
             
             # Store separately or update moving average?
             # User says: "Updates current_landmarks = Cloud Landmarks" (Direct replacement)
             self.current_landmarks = cloud_kps.detach().cpu() # Keep on CPU until needed? Or GPU?
             # Let's keep on CPU or move to device in process_input
             
        # 2. Update Amplitude
        # User says: "Updates current_amplitude = (1-m)*current + m*(Cloud_Frame_Amplitude)"
        img = None
        if 'video' in cloud_feedback:
             img = cloud_feedback['video']
        elif 'image' in cloud_feedback:
             img = cloud_feedback['image']
             
        if img is not None:
            # Extract Cloud Frame Amplitude
            cloud_amp = self.spec_normalizer.extract_amplitude(img)
            
            # Aggregate to (C, H, W)
            if cloud_amp.ndim == 5:
                # Mean over Batch and Time
                cloud_amp = cloud_amp.mean(dim=(0, 2))
            elif cloud_amp.ndim == 4:
                cloud_amp = cloud_amp.mean(dim=0)
                
            if self.current_amplitude is None:
                self.current_amplitude = cloud_amp.detach().clone()
            else:
                cloud_amp = cloud_amp.to(self.current_amplitude.device)
                m = self.momentum
                self.current_amplitude = (1 - m) * self.current_amplitude + m * cloud_amp

    def reset(self):
         """Resets state to initial Golden references."""
         self.current_landmarks = None
         if self.golden_amplitude is not None:
              self.current_amplitude = self.golden_amplitude.clone()
         else:
              self.current_amplitude = None
