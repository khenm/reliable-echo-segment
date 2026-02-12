import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from src.utils.logging import get_logger
from src.registry import register_dataset

logger = get_logger()

class EchoNetVideoDataset(Dataset):
    """
    Dataset class for EchoNet-Dynamic Video Classification/Regression.
    Loads video clips for R(2+1)D model.
    """
    def __init__(self, root_dir, split="TRAIN", max_clip_len=64, img_size=(112, 112), sampling_rate=1, transform=None, return_keypoints=False):
        self.root_dir = root_dir
        self.split = split.upper()
        self.max_clip_len = max_clip_len
        self.img_size = img_size
        self.sampling_rate = sampling_rate
        self.transform = transform
        self.return_keypoints = return_keypoints
        self.max_retries = 5
        self.videos_dir = os.path.join(root_dir, "Videos")

        self.file_list = self._load_file_list()
        self.samples = self.file_list["FileName"].values
        
        self.ef_targets = self._get_normalized_column("EF", 100.0)
        self.edv_targets = self._get_normalized_column("EDV", 300.0)
        self.esv_targets = self._get_normalized_column("ESV", 300.0)

        self.meta_lookup = self._create_meta_lookup()
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

        df["FileName"] = df["FileName"].astype(str).apply(lambda x: x[:-4] if x.lower().endswith('.avi') else x)

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

    def _get_clip_window(self, fname, total_frames):
        if fname in self.meta_lookup:
            meta = self.meta_lookup[fname]
            center_idx = (int(meta["EDFrame"]) + int(meta["ESFrame"])) // 2
        else:
            center_idx = np.random.randint(0, total_frames) if self.split == "TRAIN" else total_frames // 2

        half_len = self.max_clip_len // 2
        start_idx = center_idx - half_len
        end_idx = start_idx + self.max_clip_len
        return start_idx, end_idx

    def _pad_video_tensor(self, video_chunk, start_idx, end_idx, valid_start, valid_end):
        pad_left = max(0, valid_start - start_idx)
        pad_right = max(0, end_idx - valid_end)
        
        if pad_left > 0 or pad_right > 0:
            video_chunk = np.pad(
                video_chunk, 
                ((pad_left, pad_right), (0,0), (0,0), (0,0)), 
                mode='constant'
            )
        
        # Ensure exact length
        if video_chunk.shape[0] != self.max_clip_len:
             current_len = video_chunk.shape[0]
             if current_len < self.max_clip_len:
                 video_chunk = np.pad(video_chunk, ((0, self.max_clip_len - current_len), (0,0), (0,0), (0,0)), mode='constant')
             else:
                 video_chunk = video_chunk[:self.max_clip_len]
        return video_chunk

    def _load_video_clip(self, video_path, fname):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return None

        start_idx, end_idx = self._get_clip_window(fname, total_frames)
        valid_start = max(0, start_idx)
        valid_end = min(total_frames, end_idx)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, valid_start)
        
        frames = []
        for _ in range(valid_end - valid_start):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if (frame.shape[0], frame.shape[1]) != self.img_size:
                frame = cv2.resize(frame, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR)
            frames.append(frame)
        cap.release()
        
        if not frames:
            return np.zeros((self.max_clip_len, *self.img_size, 3), dtype=np.uint8), 0, total_frames
            
        video_chunk = np.stack(frames, axis=0)
        video_chunk = self._pad_video_tensor(video_chunk, start_idx, end_idx, valid_start, valid_end)

        return video_chunk, start_idx, total_frames

    def _get_keypoints(self, t_subset, H, W):
        pts_df = t_subset.iloc[1:]
        kps = np.stack([
            np.concatenate([pts_df["X1"].values, pts_df["X2"].values[::-1]]),
            np.concatenate([pts_df["Y1"].values, pts_df["Y2"].values[::-1]])
        ], axis=1).astype(np.float32)

        if len(kps) != 42:
            if len(kps) > 42:
                kps = kps[:42]
            else:
                pad = np.tile(kps[-1:], (42 - len(kps), 1))
                kps = np.concatenate([kps, pad], axis=0)

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
        
        video, start_idx, total_frames = result
        T_clip, H, W, _ = video.shape

        mask_clip = np.zeros((T_clip, H, W), dtype=np.uint8)
        frame_mask = np.zeros((T_clip,), dtype=np.float32)
        keypoints_clip = np.zeros((T_clip, 42, 2), dtype=np.float32)

        if self.tracings is not None:
             file_tracings = self.tracings[self.tracings["FileName"] == fname]
             if not file_tracings.empty:
                 # Map output time t to original frame index
                 for t in range(T_clip):
                     original_frame_idx = start_idx + t
                     if 0 <= original_frame_idx < total_frames:
                         t_subset = file_tracings[file_tracings["Frame"] == original_frame_idx]
                         if not t_subset.empty:
                              mask_clip[t] = self._generate_mask(t_subset, H, W)
                              frame_mask[t] = 1.0
                              if self.return_keypoints:
                                   keypoints_clip[t] = self._get_keypoints(t_subset, H, W)

        video = video.transpose(3, 0, 1, 2).astype(np.float32) / 255.0  # (C, T, H, W)
        mask_clip = mask_clip[None, ...].astype(np.float32)  # (1, T, H, W)

        video = torch.from_numpy(video)
        mask_clip = torch.from_numpy(mask_clip)

        if self.transform:
            data = self.transform({"video": video, "label": mask_clip})
            video, mask_clip = data["video"], data.get("label", mask_clip)

        output = {
            "video": video,
            "label": mask_clip, 
            "case": fname,
            "frame_mask": torch.tensor(frame_mask, dtype=torch.float32),
            "lengths": torch.tensor(T_clip, dtype=torch.long),
            "target_ef": torch.tensor(self.ef_targets[idx], dtype=torch.float32),
            "target_edv": torch.tensor(self.edv_targets[idx], dtype=torch.float32),
            "target_esv": torch.tensor(self.esv_targets[idx], dtype=torch.float32)
        }
        
        if self.return_keypoints:
            output["keypoints"] = torch.tensor(keypoints_clip, dtype=torch.float32)
            
        return output

@register_dataset("ECHONET")
class EchoNet:
    @staticmethod
    def get_dataloaders(cfg):
        root_dir = cfg['data']['root_dir']
        batch_size = cfg['training'].get('batch_size', 8)
        num_workers = cfg['training'].get('num_workers', 4)
        model_name = cfg['model'].get('name', 'VAEUNet')

        if model_name.lower() not in ["r2plus1d", "unet_tcm", "skeletal_tracker", "segment_tracker", "temporal_segment_tracker"]:
            raise NotImplementedError(f"EchoNet dataloader not implemented for model '{model_name}'")

        max_clip_len = cfg['model'].get('max_clip_len', 250)
        img_size = tuple(cfg['data'].get('img_size', [112, 112]))
        return_kps = cfg['model'].get('return_keypoints', False)
        
        if model_name.lower() == "skeletal_tracker":
            return_kps = True
        elif model_name.lower() in ["segment_tracker", "temporal_segment_tracker"]:
            return_kps = False

        ds_tr = EchoNetVideoDataset(root_dir, "TRAIN", max_clip_len=max_clip_len, img_size=img_size, return_keypoints=return_kps)
        ds_va = EchoNetVideoDataset(root_dir, "VAL", max_clip_len=max_clip_len, img_size=img_size, return_keypoints=return_kps)
        ds_ts = EchoNetVideoDataset(root_dir, "TEST", max_clip_len=max_clip_len, img_size=img_size, return_keypoints=return_kps)

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
