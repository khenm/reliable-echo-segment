import os
import joblib
import numpy as np

class UncertaintyEstimator:
    """
    Estimates volume uncertainty using Conformal Prediction and Latent Profiling.
    
    Delta V = Q * s_hat(d) * Volume
    """
    def __init__(self, profiler, scaler, Q):
        """
        Args:
            profiler (LatentProfiler): Fitted latent profiler.
            scaler (AdaptiveScaler): Fitted adaptive scaler.
            Q (float): Conformal quantile.
        """
        self.profiler = profiler
        self.scaler = scaler
        self.Q = Q
        
    @classmethod
    def load(cls, calibration_state_path, profiler_path=None):
        """
        Loads the estimator from saved calibration state.
        
        Args:
            calibration_state_path (str): Path to calibration_state.joblib
            profiler_path (str, optional): Path to latent_profile.joblib. 
                                           If None, tries to find it in same dir.
        """
        if not os.path.exists(calibration_state_path):
            raise FileNotFoundError(f"Calibration state not found at {calibration_state_path}")
            
        state = joblib.load(calibration_state_path)
        scaler = state['scaler']
        Q = state['Q']
        
        if profiler_path is None:
            # Assume profiler is in same dir as calibration state
            run_dir = os.path.dirname(calibration_state_path)
            candidate_path = os.path.join(run_dir, "latent_profile.joblib")
            if os.path.exists(candidate_path):
                profiler_path = candidate_path
            else:
                 from src.utils.util_ import find_latest_latent_profile
                 print(f"Latent profile not found at {candidate_path}. Searching for latest in runs/...")
                 profiler_path = find_latest_latent_profile("runs")
                 if profiler_path:
                      print(f"Found latest latent profile: {profiler_path}")
                 else:
                      # Let it fail in LatentProfiler.load if still None/invalid, or raise here
                      pass
            
        from src.analysis.latent_profile import LatentProfiler
        profiler = LatentProfiler()
        profiler.load(profiler_path)
        
        return cls(profiler, scaler, Q)
        
    def estimate_uncertainty(self, mu_vector, volume):
        """
        Estimates the uncertainty radius (Delta V) for a given volume calculation.
        
        Args:
            mu_vector (np.array): Latent mean vector of shape (latent_dim,).
            volume (float): Calculated volume (or area).
            
        Returns:
            float: Uncertainty radius Delta V.
        """
        # 1. Compute Mahalanobis Distance
        dist = self.profiler.get_mahalanobis_distance(mu_vector)
        
        # 2. Predict Error Score (s_hat)
        # scaler.predict expects array
        s_hat = self.scaler.predict(np.array([dist]))[0]
        
        # Ensure non-negative (though scaler fit should handle this, modest clip)
        s_hat = max(s_hat, 1e-6)
        
        # 3. Compute Delta V = Q * s_hat * V
        delta_v = self.Q * s_hat * volume
        
        return delta_v
