import torch
import torch.nn.functional as F
import math

class VideoAugmentor:
    """
    Provides consistent augmentations for video tensors of shape (B, C, T, H, W).
    Ensures that spatial transformations are applied consistently across ALL frames (T).
    """
    def __init__(self, hflip=True, shift_limit=0.1):
        self.hflip = hflip
        self.shift_limit = shift_limit

    def __call__(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T, H, W).
        Returns:
            torch.Tensor: Augmented tensor of same shape.
        """
        return self.augment(x)

    def augment(self, x):
        B, C, T, H, W = x.shape
        
        # 1. Random Horizontal Flip
        if self.hflip and torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[-1]) # Flip W dimension

        # 2. Random Translation (Shift)
        # We apply the SAME shift to all frames in the video to maintain temporal consistency
        if self.shift_limit > 0:
            dx = (torch.rand(1).item() * 2 - 1) * self.shift_limit * W
            dy = (torch.rand(1).item() * 2 - 1) * self.shift_limit * H
            
            # Create affine grid
            # Theta: (B, 2, 3)
            # Translation in grid_sample uses normalized coordinates [-1, 1]
            # tx refers to horizontal shift, ty to vertical
            tx = -dx / (W / 2.0)
            ty = -dy / (H / 2.0)
            
            theta = torch.tensor([[1, 0, tx], [0, 1, ty]], dtype=x.dtype, device=x.device)
            theta = theta.repeat(B * T, 1, 1) # Apply to each frame efficiently
            
            grid = F.affine_grid(theta, torch.Size((B * T, C, H, W)), align_corners=False)
            
            # Reshape x to (B*T, C, H, W) for grid_sample
            x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            x_shifted = F.grid_sample(x_reshaped, grid, mode='bilinear', padding_mode='border', align_corners=False)
            
            # Reshape back to (B, C, T, H, W) -> (B, T, C, H, W) -> permute
            x = x_shifted.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)

        return x

class GeometryAligner:
    """
    Aligns input frames to a 'Golden' reference geometry using Affine Transformation.
    Also handles inverse warping of masks back to patient coordinates.
    """
    def __init__(self, output_size=(112, 112)):
        self.output_size = output_size

    def estimate_affine_matrix(self, src_pts, dst_pts):
        """
        Estimates affine matrix M (2x3) such that dst = M * src.
        Uses Least Squares.
        Args:
            src_pts: (B, N, 2) [x, y]
            dst_pts: (B, N, 2) [x, y]
        Returns:
            M: (B, 2, 3)
        """
        B, N, _ = src_pts.shape
        
        # Construct system of linear equations
        # We want to solve for M in: M * [x, y, 1]^T = [x', y']^T
        # Transpose to solve: [x, y, 1] * M^T = [x', y']
        
        ones = torch.ones(B, N, 1, device=src_pts.device, dtype=src_pts.dtype)
        X = torch.cat([src_pts, ones], dim=2)  # (B, N, 3)
        Y = dst_pts  # (B, N, 2)

        # Solve X * M^T = Y
        # M^T = (X^T * X)^-1 * X^T * Y
        # torch.linalg.lstsq is better for stability
        
        # Result is (B, 3, 2), need (B, 2, 3)
        try:
            M_T = torch.linalg.lstsq(X, Y).solution
            M = M_T.permute(0, 2, 1)  # (B, 2, 3)
        except RuntimeError:
            # Fallback to identity if singular
             M = torch.tensor([[1, 0, 0], [0, 1, 0]], 
                             device=src_pts.device, dtype=src_pts.dtype).unsqueeze(0).repeat(B, 1, 1)
        
        return M

    def warp(self, img, matrix, mode='bilinear'):
        """
        Warps image using affine matrix.
        Args:
            img: (B, C, H, W) or (B, C, T, H, W)
            matrix: (B, 2, 3)
        """
        is_video = (img.ndim == 5)
        if is_video:
            B, C, T, H, W = img.shape
            # Apply same matrix to all frames usually, or different if matrix is (B, T, 2, 3)
            # Assuming matrix is (B, 2, 3) -> applied globally
            img = img.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            matrix_rep = matrix.repeat_interleave(T, dim=0)
        else:
            B, C, H, W = img.shape
            matrix_rep = matrix

        grid = F.affine_grid(matrix_rep, img.size(), align_corners=False)
        warped = F.grid_sample(img, grid, mode=mode, padding_mode='border', align_corners=False)

        if is_video:
            warped = warped.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)
            
        return warped

    def inverse_warp(self, img, matrix, mode='nearest'):
        """
        Warps image/mask back to original coordinates using Inverse Affine.
        """
        # Invert the 2x3 matrix (augment to 3x3)
        B = matrix.shape[0]
        row = torch.tensor([0, 0, 1], device=matrix.device, dtype=matrix.dtype).view(1, 1, 3).repeat(B, 1, 1)
        M_3x3 = torch.cat([matrix, row], dim=1) # (B, 3, 3)
        
        M_inv = torch.linalg.inv(M_3x3)[:, :2, :] # (B, 2, 3)
        
        return self.warp(img, M_inv, mode=mode)


class SpectralNormalizer:
    """
    Aligns input texture to a 'Golden' reference texture using Fourier Domain Adaptation (FDA).
    """
    def __init__(self, beta=0.01):
        """
        Args:
            beta (float): Fraction of low-frequency spectrum to swap (0.0 to 1.0).
                          Original FDA paper suggests 0.01 - 0.1 depending on domain gap.
        """
        self.beta = beta

    def extract_amplitude(self, img):
        """
        Compute FFT amplitude of the image.
        Args:
            img: (B, C, H, W) or (B, C, T, H, W)
        Returns:
             amplitude: same shape
        """
        # FFT expects last 2 dims to be H, W
        fft = torch.fft.rfft2(img, norm='backward')
        return torch.abs(fft)

    def inject_style(self, img, ref_amplitude):
        """
        Swaps the low-frequency amplitude of img with ref_amplitude.
        Args:
            img: Source image
            ref_amplitude: Target amplitude (Golden Ref)
        """
        # 1. FFT
        fft_src = torch.fft.rfft2(img, norm='backward')
        amp_src, pha_src = torch.abs(fft_src), torch.angle(fft_src)
        
        # 2. Spectral Swap (Low-Freq)
        batch = img.shape[0]
        h, w = fft_src.shape[-2:]
        b = int(h * self.beta) # Beta window size
        
        if b > 0:
            center_h, center_w = int(h/2), int(w/2) # Not used for RFFT2, RFFT puts DC at (0,0)
            # RFFT structure: 
            # H dim: [0, 1, ... H/2, -H/2+1 ... -1] (Standard) or just 0..H-1?
            # torch.fft.rfft2 returns standard layout (freq 0 at index 0)
            
            # Create a mask for low frequencies
            # For RFFT, top-left corner is low freq.
            
            # Simple square mask at (0,0)
            amp_src[..., 0:b, 0:b] = ref_amplitude[..., 0:b, 0:b]

            # Note: Depending on layout, we might need to handle corners, but standard FDA often
            # just takes the center after fftshift. 
            # With unshifted RFFT, low freqs are at (0,0).
            
        # 3. Reconstruct
        fft_new = amp_src * torch.exp(1j * pha_src)
        img_new = torch.fft.irfft2(fft_new, s=img.shape[-2:], norm='backward')
        
        return img_new
