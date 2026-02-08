import os
import cv2
import numpy as np
import pandas as pd
import torch
from monai.transforms import Compose, Lambda
from torch.utils.data import Dataset, DataLoader, Subset

from src.utils.logging import get_logger
from src.registry import register_dataset

logger = get_logger()

class EchoNetVideoDataset(Dataset):
    """
    Dataset class for EchoNet-Dynamic Video Classification/Regression.
    Loads video clips for R(2+1)D model.
    """
    def __init__(self, root_dir, split="TRAIN", clip_len=32, sampling_rate=1, transform=None, return_keypoints=False):
        self.root_dir = root_dir
        self.split = split.upper()
        self.clip_len = clip_len
        self.sampling_rate = sampling_rate
        self.transform = transform
        self.return_keypoints = return_keypoints
        self.max_retries = 5
        self.videos_dir = os.path.join(root_dir, "Videos")

        self.file_list = self._load_file_list()
        self.samples = self.file_list["FileName"].values
        
        # Targets
        self.ef_targets = self._get_normalized_column("EF", 100.0)
        self.edv_targets = self._get_normalized_column("EDV", 300.0)
        self.esv_targets = self._get_normalized_column("ESV", 300.0)

        # Meta Lookup for Beat-Centric Sampling
        self.meta_lookup = self._create_meta_lookup()

        # Tracings
        self.tracings = self._load_tracings()

    def _load_file_list(self):
        fname_w_frames = "FileListwFrames112.csv"
        path = os.path.join(self.root_dir, fname_w_frames)
        if not os.path.exists(path):
            path = os.path.join(self.root_dir, "FileList.csv")
            logger.warning(f"{fname_w_frames} not found, falling back to FileList.csv.")

        if not os.path.exists(path):
            raise FileNotFoundError(f"FileList.csv not found at {path}")

        df = pd.read_csv(path)
        if "Split" in df.columns:
            df = df[df["Split"].str.upper() == self.split]

        # Normalize FileName
        df["FileName"] = df["FileName"].astype(str).apply(lambda x: x[:-4] if x.lower().endswith('.avi') else x)

        # Robustness: Filter by existence on disk
        if os.path.exists(self.videos_dir):
            available_files = {f[:-4] if f.lower().endswith('.avi') else f for f in os.listdir(self.videos_dir)}
            original_len = len(df)
            df = df[df["FileName"].isin(available_files)]
            if len(df) < original_len:
                logger.warning(f"Filtered {original_len - len(df)} missing videos. Remaining: {len(df)}")
        
        return df

    def _get_normalized_column(self, col_name, scale):
        if col_name in self.file_list.columns:
            return self.file_list[col_name].values / scale
        return np.full(len(self.file_list), -1.0 if "V" in col_name else 0.0)

    def _create_meta_lookup(self):
        if "EDFrame" in self.file_list.columns and "ESFrame" in self.file_list.columns:
            return self.file_list.set_index("FileName")[["EDFrame", "ESFrame"]].to_dict('index')
        logger.warning("EDFrame/ESFrame columns missing. Beat-centric sampling disabled.")
        return {}

    def _load_tracings(self):
        path = os.path.join(self.root_dir, "VolumeTracings.csv")
        if not os.path.exists(path):
            logger.warning(f"VolumeTracings.csv not found at {path}. No masks will be generated.")
            return None
            
        df = pd.read_csv(path)
        df["FileName"] = df["FileName"].astype(str).apply(lambda x: x[:-4] if x.lower().endswith('.avi') else x)
        return df[df["FileName"].isin(set(self.samples))]

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _generate_mask(points, height, width):
        mask = np.zeros((height, width), dtype=np.uint8)
        points = points.iloc[1:] 
        pts = np.stack([
            np.concatenate([points["X1"].values, points["X2"].values[::-1]]),
            np.concatenate([points["Y1"].values, points["Y2"].values[::-1]])
        ], axis=1).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
        return mask

    def _get_start_frame(self, fname, total_frames):
        max_len = self.clip_len
        
        if fname in self.meta_lookup:
            meta = self.meta_lookup[fname]
            ed, es = int(meta["EDFrame"]), int(meta["ESFrame"])
            cycle_dist = abs(ed - es)
            target_len = min(max_len, max(int(1.5 * cycle_dist), cycle_dist))
            midpoint = (ed + es) / 2
            start_frame = int(midpoint - target_len / 2)
            
            # Bounds check
            if start_frame < 0:
                start_frame = 0
            elif start_frame + target_len > total_frames:
                start_frame = max(0, total_frames - target_len)
            
            return start_frame, target_len
        
        # Fallback
        frames_to_read = max_len
        if total_frames > max_len * self.sampling_rate:
            if self.split == "TRAIN":
                start_frame = np.random.randint(0, total_frames - max_len * self.sampling_rate)
            else:
                start_frame = (total_frames - max_len * self.sampling_rate) // 2
        else:
            start_frame = 0
            
        return start_frame, frames_to_read

    def _load_video_clip(self, video_path, fname):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame, frames_to_read = self._get_start_frame(fname, total_frames)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        count = 0
        while count < frames_to_read:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            count += 1
            for _ in range(self.sampling_rate - 1):
                cap.read()
        cap.release()
        
        if not frames:
            return None
            
        actual_len = len(frames)
        if actual_len < self.clip_len:
            frames.extend([np.zeros_like(frames[0])] * (self.clip_len - actual_len))
            
        return np.stack(frames, axis=0), start_frame, actual_len

    def _get_keypoints(self, t_subset, H, W):
        pts_df = t_subset.iloc[1:]
        kps = np.stack([
            np.concatenate([pts_df["X1"].values, pts_df["X2"].values[::-1]]),
            np.concatenate([pts_df["Y1"].values, pts_df["Y2"].values[::-1]])
        ], axis=1).astype(np.float32)

        # Placeholder logic: Ensure 42 points
        if len(kps) != 42:
            if len(kps) > 42:
                kps = kps[:42]
            else:
                pad = np.tile(kps[-1:], (42 - len(kps), 1))
                kps = np.concatenate([kps, pad], axis=0)

        # Normalize
        kps[:, 0] /= W
        kps[:, 1] /= H
        return kps

    def __getitem__(self, idx, _retry_count=0):
        fname = self.samples[idx]
        video_path = os.path.join(self.videos_dir, fname + (".avi" if not fname.lower().endswith(".avi") else ""))
        
        result = self._load_video_clip(video_path, fname)
        if result is None:
            logger.warning(f"Failed to load video clip: {video_path}")
            if _retry_count >= self.max_retries:
                raise RuntimeError(f"Failed to load {self.max_retries} consecutive samples from {idx}")
            return self.__getitem__((idx+1)%len(self), _retry_count + 1)
        
        video, start_frame, actual_len = result
        T_clip, H, W, _ = video.shape

        mask_clip = np.zeros((T_clip, H, W), dtype=np.uint8)
        frame_mask = np.zeros((T_clip,), dtype=np.float32)
        keypoints_clip = np.zeros((T_clip, 42, 2), dtype=np.float32)

        if self.tracings is not None:
             file_tracings = self.tracings[self.tracings["FileName"] == fname]
             if not file_tracings.empty:
                 current_frame_idx = start_frame
                 for t in range(min(T_clip, actual_len)):
                     t_subset = file_tracings[file_tracings["Frame"] == current_frame_idx]
                     if not t_subset.empty:
                          mask_clip[t] = self._generate_mask(t_subset, H, W)
                          frame_mask[t] = 1.0
                          if self.return_keypoints:
                              keypoints_clip[t] = self._get_keypoints(t_subset, H, W)
                     current_frame_idx += self.sampling_rate

        video = video.transpose(3, 0, 1, 2).astype(np.float32) / 255.0 # (C, T, H, W)
        mask_clip = mask_clip[None, ...].astype(np.float32) # (1, T, H, W)

        if self.transform:
            data = self.transform({"video": video, "label": mask_clip})
            video, mask_clip = data["video"], data.get("label", mask_clip)
            
        output = {
            "video": video, 
            "label": mask_clip, 
            "case": fname,
            "frame_mask": torch.tensor(frame_mask, dtype=torch.float32),
            "lengths": torch.tensor(actual_len, dtype=torch.long),
            "target_ef": torch.tensor(self.ef_targets[idx], dtype=torch.float32),
            "target_edv": torch.tensor(self.edv_targets[idx], dtype=torch.float32),
            "target_esv": torch.tensor(self.esv_targets[idx], dtype=torch.float32)
        }
        
        if self.return_keypoints:
            output["keypoints"] = torch.tensor(keypoints_clip, dtype=torch.float32)
            
        return output

class ResizeVideoLabel:
    def __init__(self, size, clip_len):
        self.size = size
        self.clip_len = clip_len

    def __call__(self, data):
        vid = torch.as_tensor(data["video"]).unsqueeze(0)
        tgt_size = (self.clip_len, self.size[0], self.size[1])
        vid = torch.nn.functional.interpolate(vid, size=tgt_size, mode='trilinear', align_corners=False).squeeze(0)
        data["video"] = vid
        
        if "label" in data:
            lab = torch.as_tensor(data["label"]).unsqueeze(0)
            lab = torch.nn.functional.interpolate(lab, size=tgt_size, mode='nearest').squeeze(0)
            data["label"] = lab
        return data

@register_dataset("ECHONET")
class EchoNet:
    @staticmethod
    def get_dataloaders(cfg):
        root_dir = cfg['data']['root_dir']
        img_size = tuple(cfg['data']['img_size'])
        batch_size = cfg['training'].get('batch_size', 8)
        num_workers = cfg['training'].get('num_workers', 4)
        model_name = cfg['model'].get('name', 'VAEUNet')
        
        if model_name.lower() not in ["r2plus1d", "unet_tcm", "skeletal_tracker", "segment_tracker", "temporal_segment_tracker"]:
             raise NotImplementedError(f"EchoNet dataloader not implemented for model '{model_name}'")

        clip_len = cfg['model'].get('clip_length', 32)
        return_kps = cfg['model'].get('return_keypoints', False)
        if model_name.lower() == "skeletal_tracker": return_kps = True
        elif model_name.lower() in ["segment_tracker", "temporal_segment_tracker"]: return_kps = False
        
        resize_op = ResizeVideoLabel(img_size, clip_len)
        train_transforms = Compose([Lambda(func=resize_op)])
        val_transforms = Compose([Lambda(func=resize_op)])

        ds_tr = EchoNetVideoDataset(root_dir, "TRAIN", clip_len, transform=train_transforms, return_keypoints=return_kps)
        ds_va = EchoNetVideoDataset(root_dir, "VAL", clip_len, transform=val_transforms, return_keypoints=return_kps)
        ds_ts = EchoNetVideoDataset(root_dir, "TEST", clip_len, transform=val_transforms, return_keypoints=return_kps)

        if cfg['data'].get('subset_size'):
            subset_size = int(cfg['data']['subset_size'])
            logger.info(f"Using subset of size {subset_size} for all splits")
            ds_tr = Subset(ds_tr, range(min(len(ds_tr), subset_size)))
            ds_va = Subset(ds_va, range(min(len(ds_va), subset_size)))
            ds_ts = Subset(ds_ts, range(min(len(ds_ts), subset_size)))

        return (
            DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers),
            DataLoader(ds_ts, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        )
