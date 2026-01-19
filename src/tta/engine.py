import torch
import torch.nn as nn
import torch.optim as optim
import copy
import logging
from .augmentations import VideoAugmentor

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
