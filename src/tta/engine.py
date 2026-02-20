import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import copy
from typing import Optional, Tuple, Any, Dict, List

logger = logging.getLogger(__name__)

class CloudAdaptiveEngine:
    """
    Execution Engine for Cloud-Guided Adaptation.
    Replaces optimization-based TTA with a 'Gatekeeper' approach.
    NO Backpropagation.
    """
    def __init__(self, model, gatekeeper, uncertainty_threshold=0.5):
        """
        Args:
            model (nn.Module): The frozen model.
            gatekeeper (AdaptiveGatekeeper): The state machine for input adaptation.
            uncertainty_threshold (float): Threshold to trigger Cloud Oracle.
        """
        self.model = model
        self.gatekeeper = gatekeeper
        self.uncertainty_threshold = uncertainty_threshold
        
        # Ensure model is frozen
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
            
    def predict_step(self, video, batch_data=None, simulate_cloud=False):
        """
        Runs the Adapt -> Infer -> Audit -> (Trigger) -> Infer loop.
        Args:
            video: Input tensor.
            batch_data (dict): Full batch data (for Cloud Simulation).
            simulate_cloud (bool): Whether to use batch_data as Oracle.
        Returns:
            output: Model output.
            meta: Dict with 'cloud_triggered', 'uncertainty', etc.
        """
        meta = {
            'uncertainty': 0.0,
            'cloud_triggered': False,
            'adaptation_step': 0,
            'aligned_video': None # Optional: return aligned video for debugging
        }
        
        # Step 1: Adapt (Gatekeeper)
        # Apply current state (Landmarks/Amplitude)
        aligned_video = self.gatekeeper.process_input(video)
        
        # Step 2: Infer
        with torch.no_grad():
            output = self.model(aligned_video)
            
        # Step 3: Audit (Calculate Uncertainty)
        # Using Entropy as a proxy for "Conformal Uncertainty" in this implementation
        # Higher Entropy = Higher Uncertainty
        uncertainty = self._compute_entropy_metric(output)
        meta['uncertainty'] = uncertainty
        
        # Step 4: Trigger Logic
        # Only simulate cloud if enabled and batch data is available
        if uncertainty > self.uncertainty_threshold:
            if simulate_cloud and batch_data is not None:
                logger.info(f"Uncertainty {uncertainty:.4f} > {self.uncertainty_threshold}. Triggering Cloud Oracle...")
                meta['cloud_triggered'] = True
                
                # Retrieve Ground Truth
                # We need Landmarks ('keypoints') and Style ('video' or 'image')
                cloud_feedback = {}
                if 'keypoints' in batch_data:
                    cloud_feedback['keypoints'] = batch_data['keypoints']
                
                # For style update
                cloud_feedback['video'] = video 
                if 'image' in batch_data and 'video' not in batch_data:
                    cloud_feedback['image'] = batch_data['image']
                
                # Update Gatekeeper
                self.gatekeeper.update_state(cloud_feedback)
                
                # Re-Infer with updated state
                aligned_video_new = self.gatekeeper.process_input(video)
                with torch.no_grad():
                    output = self.model(aligned_video_new)
                
                meta['adaptation_step'] = 1
                meta['aligned_video'] = aligned_video_new
        
        return output, meta

    def compute_conformal_score(self, output):
        """
        Calculates Conformal Uncertainty Score.
        Placeholder for actual conformal set size or non-conformity score.
        Here we use Entropy.
        """
        return self._compute_entropy_metric(output)

    def _compute_entropy_metric(self, output):
        """
        Computes an uncertainty score (Entropy).
        Handles Dict, Tuple, or Tensor outputs.
        """
        logits = output
        
        # Extract Logits
        if isinstance(output, dict):
             logits = output.get('mask_logits', output)
        elif isinstance(output, tuple):
             # Assume first item is relevant logits
             logits = output[0]
             
        if not isinstance(logits, torch.Tensor):
            return 0.0 # Fallback
            
        # Calculate Entropy
        # If Seg (B, C, T, H, W) -> Binary or Multiclass
        if logits.ndim >= 4:
            if logits.shape[1] == 1:
                # Binary
                probs = torch.sigmoid(logits)
                ent = -(probs * torch.log(probs + 1e-10) + (1-probs) * torch.log(1-probs + 1e-10))
                return ent.mean().item()
            else:
                # Multiclass
                probs = F.softmax(logits, dim=1)
                ent = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
                return ent.mean().item()

        return 0.0
