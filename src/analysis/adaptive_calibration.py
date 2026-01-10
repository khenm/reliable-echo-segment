
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class AdaptiveScaler:
    """
    Learns a linear relationship between latent distance (d) and segmentation error (s).
    s_hat(d) = alpha * d + beta
    """
    def __init__(self):
        self.model = LinearRegression()
        self.is_fitted = False
        
    def fit(self, distances, errors):
        """
        Fits the linear model to the provided distances and errors.
        
        Args:
            distances (np.array): Array of Mahalanobis distances (N,).
            errors (np.array): Array of True Errors (e.g. 1-Dice) (N,).
        """
        # Reshape for sklearn (N, 1)
        X = distances.reshape(-1, 1)
        y = errors
        
        self.model.fit(X, y)
        self.is_fitted = True
        
        print(f"Adaptive Scaler Fitted: s = {self.model.coef_[0]:.4f} * d + {self.model.intercept_:.4f}")
        
    def predict(self, distances):
        """
        Predicts the expected error for given distances.
        
        Args:
            distances (np.array): Array of distances.
            
        Returns:
            np.array: Predicted error scores s_hat.
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted.")
            
        X = distances.reshape(-1, 1)
        return self.model.predict(X)
    
    def plot_calibration(self, distances, errors, save_path=None):
        """
        Plots the scatter plot of Distance vs Error and the regression line.
        """
        if not self.is_fitted:
            print("Scaler not fitted, cannot plot regression line.")
            return

        plt.figure(figsize=(8, 6))
        plt.scatter(distances, errors, alpha=0.5, label='Calibration Data')
        
        d_range = np.linspace(distances.min(), distances.max(), 100)
        s_pred = self.predict(d_range)
        
        plt.plot(d_range, s_pred, color='red', linewidth=2, label='Linear Fit')
        
        plt.xlabel("Mahalanobis Distance (d)")
        plt.ylabel("True Error (1 - Dice)")
        plt.title("Adaptive Calibration: Distance vs Error")
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Calibration plot saved to {save_path}")
        else:
            plt.show()
