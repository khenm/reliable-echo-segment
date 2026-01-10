import sys
import os
import pandas as pd
import numpy as np
import cv2

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset_echonet import EchoNetDataset

def verify_mask_generation():
    print("Verifying EchoNet mask generation logic...")

    # 1. Create Synthetic Tracing Data
    # Simulating a simple 10x10 square in the center of a 100x100 image
    # The 'points' argument in _generate_mask is expected to be a DataFrame-like slice
    # with columns X1, Y1, X2, Y2.
    
    # Points defining a square from y=45 to y=55
    # Left side (X1, Y1): (45, 45) -> (45, 55)
    # Right side (X2, Y2): (55, 45) -> (55, 55)
    
    data = {
        "X1": [45, 45],
        "Y1": [45, 55],
        "X2": [55, 55],
        "Y2": [45, 55]
    }
    points_df = pd.DataFrame(data)
    
    H, W = 100, 100
    
    # 2. Call the method
    # Since _generate_mask doesn't use 'self', we can call it by passing None for self,
    # or referencing it from the class if it were static, but python unbound method call:
    print("Testing EchoNetDataset._generate_mask...")
    mask = EchoNetDataset._generate_mask(None, points_df, H, W)
    
    # 3. Verify Output
    print(f"Output Mask Shape: {mask.shape}")
    print(f"Output Mask Unique Values: {np.unique(mask)}")
    
    # Check if we have the expected filled area
    # Square 10x10 (approx) should have area around 100
    area = mask.sum()
    print(f"Filled Area (pixels): {area}")
    
    if area > 0 and area < H*W:
        print("SUCCESS: Mask contains a filled region.")
    else:
        print("FAILURE: Mask is empty or full image.")
        
    # Check center pixel
    center_val = mask[50, 50]
    print(f"Center pixel (50,50) value: {center_val}")
    
    if center_val == 1:
        print("SUCCESS: Center pixel is part of the mask.")
    else:
        print("FAILURE: Center pixel is NOT part of the mask.")
        
if __name__ == "__main__":
    verify_mask_generation()
