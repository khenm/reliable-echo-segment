
import torch
import numpy as np
import os
import joblib

class LatentProfiler:
    """
    Profiles the latent space of a trained VAE model to establish a "Normal" anatomy baseline.
    Uses the Mahalanobis distance to measure deviation from the training distribution.
    """
    def __init__(self, latent_dim=256):
        """
        Initializes the LatentProfiler.
        
        Args:
            latent_dim (int): Dimensionality of the latent space.
        """
        self.latent_dim = latent_dim
        self.mu_train = None
        self.cov_inv = None
        
    def fit(self, model, loader, device):
        """
        Fits the profiler to the training data.
        
        Args:
            model (torch.nn.Module): Trainded VAE-U-Net model.
            loader (DataLoader): DataLoader for the training set.
            device (torch.device): Device to run computation on.
        """
        model.eval()
        mus = []
        
        print("Profiling latent space on training set...")
        with torch.no_grad():
            for batch in loader:
                imgs = batch["image"].to(device)
                # Unpack tuple: logits, mu, log_var
                _, mu, _ = model(imgs)
                mus.append(mu.cpu().numpy())
                
        all_mus = np.concatenate(mus, axis=0)
        
        # Calculate Empirical Mean and Covariance
        self.mu_train = np.mean(all_mus, axis=0)
        cov = np.cov(all_mus, rowvar=False)
        
        # Calculate Inverse Covariance (Precision Matrix)
        # Add epsilon for numerical stability if needed, though usually latent space is regularized
        try:
            self.cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            print("Warning: Singular covariance matrix. Using pseudo-inverse.")
            self.cov_inv = np.linalg.pinv(cov)
            
        print(f"Profiling complete. Processed {len(all_mus)} samples.")
        
    def get_mahalanobis_distance(self, mu_vector):
        """
        Calculates the Mahalanobis distance for a given latent vector.
        
        Args:
            mu_vector (np.array): Latent mean vector of shape (latent_dim,).
            
        Returns:
            float: Mahalanobis distance.
        """
        if self.mu_train is None or self.cov_inv is None:
            raise RuntimeError("Profiler is not fitted. Call fit() or load() first.")
            
        diff = mu_vector - self.mu_train
        # D_M = sqrt( (x-mu)T * S^-1 * (x-mu) )
        dm_sq = np.dot(np.dot(diff, self.cov_inv), diff.T)
        return np.sqrt(dm_sq)

    def save(self, path):
        """
        Saves the profiler state to a file.
        
        Args:
            path (str): Path to save the file.
        """
        state = {
            'mu_train': self.mu_train,
            'cov_inv': self.cov_inv,
            'latent_dim': self.latent_dim
        }
        joblib.dump(state, path)
        print(f"Profiler saved to {path}")
        
    def load(self, path):
        """
        Loads the profiler state from a file.
        
        Args:
            path (str): Path to load from.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Profiler file not found at {path}")
            
        state = joblib.load(path)
        self.mu_train = state['mu_train']
        self.cov_inv = state['cov_inv']
        self.latent_dim = state['latent_dim']
        print(f"Profiler loaded from {path}")
