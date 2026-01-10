import os
import cv2
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from monai.transforms import (
    Compose, ScaleIntensityRangePercentiles, Resize, EnsureChannelFirst
)
from src.utils.logging import get_logger

logger = get_logger()

class EchoNetDataset(Dataset):
    """
    Dataset class for EchoNet-Dynamic.
    Loads video frames and generates masks from coordinate tracings.
    """
    def __init__(self, root_dir, split="TRAIN", img_size=(256, 256), transform=None):
        """
        Args:
            root_dir (str): Path to EchoNet-Dynamic dataset root (containing FileList.csv, VolumeTracings.csv, Videos/).
            split (str): One of "TRAIN", "VAL", "TEST".
            img_size (tuple): Target size (H, W).
            transform (callable): MONAI transforms.
        """
        self.root_dir = root_dir
        self.split = split.upper()
        self.img_size = img_size
        self.transform = transform
        
        self.file_list_path = os.path.join(root_dir, "FileList.csv")
        self.tracings_path = os.path.join(root_dir, "VolumeTracings.csv")
        self.videos_dir = os.path.join(root_dir, "Videos")
        
        if not os.path.exists(self.file_list_path):
            raise FileNotFoundError(f"FileList.csv not found at {self.file_list_path}")
        
        # Load File List
        df = pd.read_csv(self.file_list_path)
        
        # Filter by split
        if "Split" in df.columns:
            df = df[df["Split"].str.upper() == self.split]
        else:
            logger.warning("No 'Split' column in FileList.csv. Using all data.")
            
        self.file_list = df
        
        # Load Tracings
        if os.path.exists(self.tracings_path):
            self.tracings = pd.read_csv(self.tracings_path)
            # Filter tracings for relevant files
            self.tracings = self.tracings[self.tracings["FileName"].isin(self.file_list["FileName"])]
        else:
            logger.warning(f"VolumeTracings.csv not found at {self.tracings_path}. No masks will be generated.")
            self.tracings = None
            
        # Index valid samples (File, Frame) that have tracings
        self.samples = []
        if self.tracings is not None:
            # Group by FileName and Frame
            # Only keep frames that have tracings
            grouped = self.tracings.groupby(["FileName", "Frame"])
            for (fname, frame), _ in grouped:
                self.samples.append((fname, frame))
        else:
            # Skip frames without tracings
             pass
             
        logger.info(f"EchoNet ({self.split}): Found {len(self.samples)} valid labeled frames from {len(df)} videos.")

    def __len__(self):
        return len(self.samples)

    def _load_video_frame(self, video_path, frame_idx):
        """
        Loads a specific frame from a video file.

        Args:
            video_path (str): Path to the video file.
            frame_idx (int): Index of the frame to load.

        Returns:
            np.ndarray: The loaded frame as a grayscale numpy array, or None if loading fails.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None
            
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def _generate_mask(self, points, height, width):
        """
        Generates a binary mask from coordinate points.

        Args:
            points (pd.DataFrame): DataFrame containing 'X1', 'Y1', 'X2', 'Y2' coordinates.
            height (int): Height of the mask.
            width (int): Width of the mask.

        Returns:
            np.ndarray: Binary mask with the polygon filled.
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        
        x1 = points["X1"].values
        y1 = points["Y1"].values
        x2 = points["X2"].values
        y2 = points["Y2"].values
        
        xs = np.concatenate([x1, x2[::-1]])
        ys = np.concatenate([y1, y2[::-1]])
        
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
        
        return mask

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: Dictionary containing 'image', 'label', 'case', 'view', and 'phase'.
        """
        fname, frame_idx = self.samples[idx]
        
        video_path = os.path.join(self.videos_dir, fname)
        if not fname.lower().endswith(".avi"):
             video_path += ".avi"
             
        # Load Image
        img_arr = self._load_video_frame(video_path, frame_idx)
        if img_arr is None:
            logger.error(f"Error loading {fname} frame {frame_idx}")
            return self.__getitem__((idx+1)%len(self))
            
        H, W = img_arr.shape
        
        # Load/Gen Mask
        if self.tracings is not None:
            # Get points
            t_subset = self.tracings[(self.tracings["FileName"] == fname) & (self.tracings["Frame"] == frame_idx)]
            mask_arr = self._generate_mask(t_subset, H, W)
        else:
            mask_arr = np.zeros_like(img_arr)

        # Prepare for Transforms
        img_arr = img_arr[None, ...].astype(np.float32) # (1, H, W)
        mask_arr = mask_arr[None, ...].astype(np.float32) # (1, H, W)
        
        data = {"image": img_arr, "label": mask_arr}
        
        # Apply Transforms
        if self.transform:
            data = self.transform(data)
            
        # Metadata
        data["case"] = fname
        data["view"] = "A4C"
        
        # Identify Phase (ED/ES) for EF calculation
        row = self.file_list[self.file_list["FileName"] == fname]
        phase = f"F{frame_idx}"
        
        if not row.empty:
            ed_frame = row.iloc[0].get("EDFrame", -1)
            es_frame = row.iloc[0].get("ESFrame", -1)
            
            # EchoNet Frame indices are integers
            if frame_idx == ed_frame:
                phase = "ED"
            elif frame_idx == es_frame:
                phase = "ES"
                
        data["phase"] = phase
        
        return data

def get_echonet_dataloaders(cfg):
    """
    Returns (train, val, test) dataloaders for EchoNet.
    """
    root_dir = cfg['data']['root_dir']
    img_size = tuple(cfg['data']['img_size'])
    batch_size = cfg['training']['batch_size_train']
    num_workers = cfg['training'].get('num_workers', 4)
    
    # Define Transforms
    from monai.transforms import ScaleIntensityRangePercentilesd, ResizeWithPadOrCropd, RandFlipd, RandRotate90d, RandAffined, ToTensord
    
    tf_tr = Compose([
        ScaleIntensityRangePercentilesd("image", 1, 99, 0, 1, clip=True),
        ResizeWithPadOrCropd(("image", "label"), img_size),
        RandFlipd(("image", "label"), prob=0.5, spatial_axis=1),
        RandRotate90d(("image", "label"), prob=0.5, max_k=3),
        RandAffined(("image", "label"), prob=0.3, rotate_range=math.pi/18, mode=("bilinear", "nearest")),
        ToTensord(("image", "label"))
    ])
    
    tf_val = Compose([
        ScaleIntensityRangePercentilesd("image", 1, 99, 0, 1, clip=True),
        ResizeWithPadOrCropd(("image", "label"), img_size),
        ToTensord(("image", "label"))
    ])
    
    ds_tr = EchoNetDataset(root_dir, split="TRAIN", img_size=img_size, transform=tf_tr)
    ds_va = EchoNetDataset(root_dir, split="VAL", img_size=img_size, transform=tf_val)
    ds_ts = EchoNetDataset(root_dir, split="TEST", img_size=img_size, transform=tf_val)
    
    # Collate: default list_data_collate from Monai works well for dicts
    from monai.data import list_data_collate, DataLoader
    
    ld_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=list_data_collate)
    ld_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=list_data_collate)
    ld_ts = DataLoader(ds_ts, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=list_data_collate)
    
    return ld_tr, ld_va, ld_ts
