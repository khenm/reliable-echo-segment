import math
import os

import cv2
import numpy as np
import pandas as pd
import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    Lambda,
    RandFlip,
    RandRotate90,
    Resize,
    ScaleIntensityRangePercentiles,
    ToTensor,
)
from torch.utils.data import Dataset, DataLoader

from src.utils.logging import get_logger
from src.registry import register_dataset

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
            max_retries (int): Maximum number of retries for loading samples.
        """
        self.root_dir = root_dir
        self.split = split.upper()
        self.img_size = img_size
        self.transform = transform
        self.max_retries = 5
        
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
            
        # Normalize FileName: remove .avi extension if present to ensure consistency
        df["FileName"] = df["FileName"].astype(str).apply(lambda x: x[:-4] if x.lower().endswith('.avi') else x)
        self.file_list = df
        
        # Load Tracings
        if os.path.exists(self.tracings_path):
            self.tracings = pd.read_csv(self.tracings_path)
            
            # Normalize Tracings FileName as well
            self.tracings["FileName"] = self.tracings["FileName"].astype(str).apply(lambda x: x[:-4] if x.lower().endswith('.avi') else x)
            
            # Filter tracings for relevant files
            # Check intersection
            valid_files = set(self.file_list["FileName"])
            original_tracing_count = len(self.tracings)
            self.tracings = self.tracings[self.tracings["FileName"].isin(valid_files)]
            
            if len(self.tracings) == 0 and original_tracing_count > 0:
                logger.warning(f"No tracings matched file list for split {self.split}. "
                               f"Sample FileList: {list(valid_files)[:5]}. "
                               f"Sample Tracings: {list(pd.read_csv(self.tracings_path)['FileName'].unique())[:5]}")

        else:
            logger.warning(f"VolumeTracings.csv not found at {self.tracings_path}. No masks will be generated.")
            self.tracings = None
            
        # Index valid samples (File, Frame) that have tracings
        self.samples = []
        if self.tracings is not None:
            # Group by FileName and Frame
            grouped = self.tracings.groupby(["FileName", "Frame"])
            for (fname, frame), _ in grouped:
                self.samples.append((fname, frame))
             
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
    
        # Drop the first row (the longitudinal axis)
        points = points.iloc[1:] 

        x1 = points["X1"].values
        y1 = points["Y1"].values
        x2 = points["X2"].values
        y2 = points["Y2"].values
        
        # Traverse down X1/Y1, then UP X2/Y2
        xs = np.concatenate([x1, x2[::-1]])
        ys = np.concatenate([y1, y2[::-1]])
        
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)

        return mask

    def __getitem__(self, idx, _retry_count=0):
        """
        Retrieves a sample from the dataset.

        Args:
            idx (int): Index of the sample.
            _retry_count (int): Internal counter for retry attempts.

        Returns:
            dict: Dictionary containing 'image', 'label', 'case', 'view', and 'phase'.
        """
        max_retries = self.max_retries

        fname, frame_idx = self.samples[idx]
        
        video_path = os.path.join(self.videos_dir, fname)
        if not fname.lower().endswith(".avi"):
             video_path += ".avi"
             
        # Load Image
        img_arr = self._load_video_frame(video_path, frame_idx)
        if img_arr is None:
            logger.error(f"Error loading {fname} frame {frame_idx}")
            if _retry_count >= max_retries:
                raise RuntimeError(f"Failed to load {max_retries} consecutive samples starting from idx {idx}")
            return self.__getitem__((idx+1)%len(self), _retry_count + 1)
            
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

class EchoNetVideoDataset(Dataset):
    """
    Dataset class for EchoNet-Dynamic Video Classification/Regression.
    Loads video clips for R(2+1)D model.
    """
    def __init__(self, root_dir, split="TRAIN", clip_len=32, sampling_rate=1, transform=None, return_keypoints=False):
        """
        Args:
            root_dir (str): Path to EchoNet-Dynamic dataset root.
            split (str): One of "TRAIN", "VAL", "TEST".
            clip_len (int): Number of frames to sample.
            sampling_rate (int): Step size between frames.
            transform (callable): Spatial transforms.
            max_retries (int): Maximum number of retries for video loading.
        """
        self.root_dir = root_dir
        self.split = split.upper()
        self.clip_len = clip_len
        self.sampling_rate = sampling_rate
        self.transform = transform
        self.return_keypoints = return_keypoints
        self.max_retries = 5
        
        # Use filename with frames metadata by default if available, otherwise fallback or config
        # Ideally passed via config, but hardcoding priority for now based on plan
        fname_w_frames = "FileListwFrames112.csv"
        self.file_list_path = os.path.join(root_dir, fname_w_frames)
        if not os.path.exists(self.file_list_path):
             # Fallback
             self.file_list_path = os.path.join(root_dir, "FileList.csv")
             logger.warning(f"{fname_w_frames} not found, falling back to FileList.csv. Cycle sampling may fail if columns missing.")

        self.videos_dir = os.path.join(root_dir, "Videos")
        
        if not os.path.exists(self.file_list_path):
            raise FileNotFoundError(f"FileList.csv not found at {self.file_list_path}")
        
        # Load File List
        df = pd.read_csv(self.file_list_path)
        
        # Filter by split
        if "Split" in df.columns:
            df = df[df["Split"].str.upper() == self.split]
            
        # Normalize FileName
        df["FileName"] = df["FileName"].astype(str).apply(lambda x: x[:-4] if x.lower().endswith('.avi') else x)
        
        # Filter by existence on disk (Robustness for partial datasets)
        if os.path.exists(self.videos_dir):
            available_files = set(os.listdir(self.videos_dir))
            # Create set of available basenames (assuming .avi)
            available_bases = {f[:-4] if f.lower().endswith('.avi') else f for f in available_files}
            
            # Keep only files that exist
            original_len = len(df)
            df = df[df["FileName"].isin(available_bases)]
            
            if len(df) < original_len:
                logger.warning(f"Filtered {original_len - len(df)} missing videos from FileList. Remaining: {len(df)}")
        
        self.file_list = df
        self.samples = df["FileName"].values
        self.targets = df["EF"].values if "EF" in df.columns else None
        
        # Create Metadata Lookup for Cycle Sampling
        # Ensure EDFrame/ESFrame exist
        self.meta_lookup = {}
        if "EDFrame" in df.columns and "ESFrame" in df.columns:
            # Create dict: fname -> {'ED': int, 'ES': int}
            # Handle potential float/int issues
            temp_df = df.set_index("FileName")[["EDFrame", "ESFrame"]]
            self.meta_lookup = temp_df.to_dict('index')
        else:
            logger.warning("EDFrame/ESFrame columns missing. Beat-centric sampling disabled.")

        self.tracings_path = os.path.join(root_dir, "VolumeTracings.csv")
        self.tracings = None
        if os.path.exists(self.tracings_path):
            self.tracings = pd.read_csv(self.tracings_path)
            self.tracings["FileName"] = self.tracings["FileName"].astype(str).apply(lambda x: x[:-4] if x.lower().endswith('.avi') else x)
            # Filter tracings for relevant files
            valid_files = set(self.samples)
            self.tracings = self.tracings[self.tracings["FileName"].isin(valid_files)]
            
            # Cache annotated frames for smarter sampling
            self.file_to_frames = self.tracings.groupby("FileName")["Frame"].apply(list).to_dict()
        else:
            logger.warning(f"VolumeTracings.csv not found at {self.tracings_path}. No masks will be generated.")
            self.file_to_frames = {}

    def __len__(self):
        return len(self.samples)

    def _generate_mask(self, points, height, width):
        """
        Generates a binary mask from coordinate points.
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        
        points = points.iloc[1:] 
        
        x1 = points["X1"].values
        y1 = points["Y1"].values
        x2 = points["X2"].values
        y2 = points["Y2"].values
        
        xs = np.concatenate([x1, x2[::-1]])
        ys = np.concatenate([y1, y2[::-1]])
        
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
        
        return mask

    def _load_video_clip(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fname = os.path.basename(video_path).split('.')[0]
        
        # Beat-Centric Sampling Logic
        start_frame = 0
        cycle_found = False
        
        # Define constraints
        # We want to capture [min(ED, ES), max(ED, ES)]
        # We pad to self.clip_len (treated as MAX_LEN now)
        MAX_LEN = self.clip_len
        
        if fname in self.meta_lookup:
            meta = self.meta_lookup[fname]
            ed = int(meta["EDFrame"])
            es = int(meta["ESFrame"])
            
            # 1.5x Cycle Logic
            cycle_dist = abs(ed - es)
            
            # User request: "length = 1.5 ed and es"
            # We interpret this as 1.5 * cycle_dist centered around midpoint
            target_len = int(1.5 * cycle_dist)
            
            # Clamp target_len to MAX_LEN (clip_len) to avoid buffer overflow
            # And clamp to at least some minimum (e.g. cycle_dist)
            target_len = min(MAX_LEN, max(target_len, cycle_dist))
            
            # Determine Window Center
            midpoint = (ed + es) / 2
            
            # Calculate Start Frame
            start_frame = int(midpoint - target_len / 2)
            
            # Ensure valid bounds
            # 1. Start >= 0
            # 2. End (start + target_len) <= total_frames
            
            # Bias shifts if out of bounds
            if start_frame < 0:
                start_frame = 0
            elif start_frame + target_len > total_frames:
                start_frame = max(0, total_frames - target_len)
                
            # Frames to read is target_len (unless video is too short)
            frames_to_read = target_len
            
        else:
            # Fallback to old logic or full clip
            frames_to_read = MAX_LEN
            if total_frames > MAX_LEN * self.sampling_rate:
                if self.split == "TRAIN":
                    start_frame = np.random.randint(0, total_frames - MAX_LEN * self.sampling_rate)
                else:
                    start_frame = (total_frames - MAX_LEN * self.sampling_rate) // 2
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        # Load up to `frames_to_read` frames
        count = 0
        while count < frames_to_read:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            count += 1
            
            # Skip frames for sampling rate
            for _ in range(self.sampling_rate - 1):
                cap.read()
                
        cap.release()
        
        if len(frames) == 0:
            return None
            
        actual_len = len(frames)
        
        # Pad with Zeros up to MAX_LEN (112)
        # This is CRITICAL: tensor size must be fixed (MAX_LEN), but actual_len tells model where to stop.
        if actual_len < MAX_LEN:
            pad_needed = MAX_LEN - actual_len
            # Pad with zeros
            if actual_len > 0:
                padding = [np.zeros_like(frames[0])] * pad_needed
            frames.extend(padding)
            
        # Stack -> (MAX_LEN, H, W, C)
        video = np.stack(frames, axis=0)
        return video, start_frame, actual_len

    def __getitem__(self, idx, _retry_count=0):
        max_retries = self.max_retries
        fname = self.samples[idx]
        target = self.targets[idx] if self.targets is not None else 0.0
        
        video_path = os.path.join(self.videos_dir, fname)
        if not fname.lower().endswith(".avi"):
             video_path += ".avi"
             
        result = self._load_video_clip(video_path)
        if result is None:
            logger.warning(f"Failed to load video clip: {video_path}")
            if _retry_count >= max_retries:
                raise RuntimeError(f"Failed to load {max_retries} consecutive video samples starting from idx {idx}")
            return self.__getitem__((idx+1)%len(self), _retry_count + 1)
        
        video, start_frame, actual_len = result
        T_clip, H, W, C = video.shape

        # Generate Masks for the clip
        # Output shape: (T, H, W) -> will become (1, T, H, W) after transform usually or we handle it manually
        mask_clip = np.zeros((T_clip, H, W), dtype=np.uint8)
        frame_mask = np.zeros((T_clip,), dtype=np.float32)
        
        # Keypoints: (T, 42, 2) - Normalized [0, 1]
        # We fill with -1 or similar for missing frames, effectively 0 with frame_mask=0
        keypoints_clip = np.zeros((T_clip, 42, 2), dtype=np.float32)

        if self.tracings is not None:
             # Find tracings for this file
             file_tracings = self.tracings[self.tracings["FileName"] == fname]
             if not file_tracings.empty:
                 # Reconstruction of frame indices to match clip
                 current_frame_idx = start_frame
                 for t in range(T_clip):
                     if t >= actual_len:
                         break
                         
                     # Tracing frame is integer
                     t_subset = file_tracings[file_tracings["Frame"] == current_frame_idx]
                     if not t_subset.empty:
                          mask_clip[t] = self._generate_mask(t_subset, H, W)
                          frame_mask[t] = 1.0
                          
                          if self.return_keypoints:
                              # Extract keypoints
                              pts_df = t_subset.iloc[1:] # Drop axis
                              x1 = pts_df["X1"].values
                              y1 = pts_df["Y1"].values
                              x2 = pts_df["X2"].values
                              y2 = pts_df["Y2"].values
                              
                              # Concatenate x1, x2[::-1] as per mask generation
                              # Shape should be 21 + 21 = 42
                              # If length matches 21 rows -> 42 points.
                              # EchoNet tracings usually have 21 interpolation points.
                              
                              xs = np.concatenate([x1, x2[::-1]])
                              ys = np.concatenate([y1, y2[::-1]])
                              
                              # Stack (42, 2)
                              kps = np.stack([xs, ys], axis=1).astype(np.float32)
                              
                              # Fix number of points to 42 using interpolation if necessary?
                              # Usually EchoNet is consistent. Let's assume consistent for now or pad/trim.
                              if len(kps) != 42:
                                   # Simple resampling or padding if mismatched (rare in cleaned EchoNet)
                                   # logic to resample kps to 42 if needed could go here
                                   # For now, let's just create a fixed size placeholder if mismatch
                                   if len(kps) > 42:
                                       kps = kps[:42]
                                   else:
                                       # pad with last point
                                       pad = np.tile(kps[-1:], (42 - len(kps), 1))
                                       kps = np.concatenate([kps, pad], axis=0)

                              # Normalize
                              kps[:, 0] /= W
                              kps[:, 1] /= H
                              
                              keypoints_clip[t] = kps
                     
                     current_frame_idx += self.sampling_rate

        # (T, H, W, C) -> (C, T, H, W) for PyTorch/Monai
        video = video.transpose(3, 0, 1, 2) # (C, T, H, W)
        video = video.astype(np.float32) / 255.0
        
        # Mask: (T, H, W) -> (1, T, H, W)
        mask_clip = mask_clip[None, ...].astype(np.float32)

        if self.transform:
            # Monai expects dictionary
            # For 3D transforms, 'video' is (C, T, H, W), 'label' should be (C, T, H, W)
            data = {"video": video, "label": mask_clip} 
            data = self.transform(data)
            video = data["video"]
            mask_clip = data.get("label", mask_clip)
            
        # Normalization: EF is 0-100, we want 0-1 for loss balancing with Dice
        if self.targets is not None:
            target = target / 100.0
            
        output = {
            "video": video, 
            "target": torch.tensor(target, dtype=torch.float32), 
            "label": mask_clip, 
            "case": fname,
            "frame_mask": torch.tensor(frame_mask, dtype=torch.float32),
            "lengths": torch.tensor(actual_len, dtype=torch.long)
        }
        
        if self.return_keypoints:
            output["keypoints"] = torch.tensor(keypoints_clip, dtype=torch.float32)
            
        return output

class ResizeVideoLabel:
    def __init__(self, size, clip_len):
        self.size = size
        self.clip_len = clip_len

    def __call__(self, data):
        # data is dict
        vid = torch.as_tensor(data["video"]).unsqueeze(0) # (1, C, T, H, W)
        tgt_size = (self.clip_len, self.size[0], self.size[1])
        
        vid = torch.nn.functional.interpolate(vid, size=tgt_size, mode='trilinear', align_corners=False).squeeze(0)
        data["video"] = vid
        
        if "label" in data:
            lab = torch.as_tensor(data["label"]).unsqueeze(0) # (1, C, T, H, W)
            lab = torch.nn.functional.interpolate(lab, size=tgt_size, mode='nearest').squeeze(0)
            data["label"] = lab
        return data

@register_dataset("ECHONET")
class EchoNet:
    @staticmethod
    def get_dataloaders(cfg):
        """
        Returns (train, val, test) dataloaders for EchoNet.
        """
        root_dir = cfg['data']['root_dir']
        img_size = tuple(cfg['data']['img_size'])
        batch_size = cfg['training']['batch_size_train']
        num_workers = cfg['training'].get('num_workers', 4)
        model_name = cfg['model'].get('name', 'VAEUNet') # Default to VAEUNet
        
        if model_name.lower() in ["r2plus1d", "unet_tcm", "skeletal_tracker"]:
            # Video Configuration
            clip_len = cfg['model'].get('clip_length', 32)
            
            # Force keypoints for skeletal_tracker
            if model_name.lower() == "skeletal_tracker":
                return_kps = True
            else:
                return_kps = cfg['model'].get('return_keypoints', False)
            
            resize_op = ResizeVideoLabel(img_size, clip_len)

            train_transforms = Compose([
                Lambda(func=resize_op),
                # Add augmentations here later
            ])
            
            val_transforms = Compose([
                Lambda(func=resize_op),
            ])

            max_retries = cfg['data'].get('max_retries', 5)
            ds_tr = EchoNetVideoDataset(root_dir, split="TRAIN", clip_len=clip_len, transform=train_transforms, return_keypoints=return_kps)
            ds_tr.max_retries = max_retries
            ds_va = EchoNetVideoDataset(root_dir, split="VAL", clip_len=clip_len, transform=val_transforms, return_keypoints=return_kps)
            ds_va.max_retries = max_retries
            ds_ts = EchoNetVideoDataset(root_dir, split="TEST", clip_len=clip_len, transform=val_transforms, return_keypoints=return_kps)
            ds_ts.max_retries = max_retries


            if cfg['data'].get('subset_size'):
                from torch.utils.data import Subset
                subset_size = int(cfg['data']['subset_size'])
                logger.info(f"Using subset of size {subset_size} for all splits")
                ds_tr = Subset(ds_tr, range(min(len(ds_tr), subset_size)))
                ds_va = Subset(ds_va, range(min(len(ds_va), subset_size)))
                ds_ts = Subset(ds_ts, range(min(len(ds_ts), subset_size)))

            ld_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            ld_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            ld_ts = DataLoader(ds_ts, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            
            return ld_tr, ld_va, ld_ts
        else:
            raise NotImplementedError(f"EchoNet dataloader not implemented for model '{model_name}'")
