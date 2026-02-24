import time
import os
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..utils.config_loader import load_config
from ..utils.logging import get_logger
from ..registry import get_model_class
import src.models

from .inference_onnx import (
    NumpySelfAuditor,
    compute_volume_from_mask,
    compute_ef_from_volumes,
    create_volume_plot_frame,
    save_final_plot,
    load_clinical_data,
    process_video,
    run_benchmark,
    run_calibration
)

logger = get_logger()

class PyTorchSegmentationInference:
    """
    PyTorch inference engine for echo segmentation.
    Supports BATCH mode inference natively with PyTorch.
    """
    def __init__(self, config_path: str, checkpoint_path: str, use_gpu: bool = True):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        logger.info(f"Loading PyTorch model on {self.device}...")
        
        cfg = load_config(config_path)
        model_name = cfg.get("model", {}).get("name", "segment_tracker")
        model_cls = get_model_class(model_name)
        self.model = model_cls.from_config(cfg)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, frames: np.ndarray) -> torch.Tensor:
        """
        Preprocess input frames. Expected input: (T, H, W) or (T, C, H, W)
        Returns: (1, 3, T, H, W) as PyTorch tensor
        """
        if frames.ndim == 3:
            frames = frames[:, np.newaxis, :, :]
        
        if frames.shape[1] == 1:
            frames = np.repeat(frames, 3, axis=1)
        elif frames.shape[1] > 3:
            frames = frames[:, :3, :, :]
        
        frames = frames.astype(np.float32)
        if frames.max() > 1.0:
            frames = frames / 255.0
            
        frames = np.expand_dims(frames, axis=0) # (1, T, C, H, W)
        frames = frames.transpose(0, 2, 1, 3, 4) # (1, C, T, H, W)
        return torch.from_numpy(frames).to(self.device)

    def predict(self, input_data: np.ndarray):
        """
        Run inference.
        Returns Numpy arrays matching the ONNX output.
        """
        input_tensor = torch.from_numpy(input_data).to(self.device) if isinstance(input_data, np.ndarray) else input_data
        
        with torch.no_grad():
            output = self.model(input_tensor)
            
            if isinstance(output, dict):
                mask_logits = output["mask_logits"]
                if "hidden_features" in output:
                    features = output["hidden_features"]
                elif "features" in output:
                    features = output["features"]
                else:
                    features = torch.zeros((input_tensor.shape[0], 1, input_tensor.shape[2]), device=input_tensor.device)
                
                pred_vol_curve = output.get("pred_vol_curve")
                pred_phase_vel = output.get("pred_phase_vel")
            else:
                mask_logits = output[0]
                features = output[-1]
                pred_vol_curve = None
                pred_phase_vel = None
                
            mask_prob = torch.sigmoid(mask_logits)
            
        return (
            mask_prob.cpu().numpy(), 
            features.cpu().numpy(),
            pred_vol_curve.cpu().numpy() if pred_vol_curve is not None else None,
            pred_phase_vel.cpu().numpy() if pred_phase_vel is not None else None
        )

    def segment_video(self, frames: np.ndarray, threshold: float = 0.5):
        """
        Extract masks, probs, and features from a batch of frames.
        """
        input_data = self.preprocess(frames)
        mask_prob, features, pred_vol, pred_phase_vel = self.predict(input_data)
        
        mask_prob_squeezed = mask_prob.squeeze(0).squeeze(0)
        binary_mask = (mask_prob_squeezed > threshold).astype(np.uint8)
        
        if features is not None:
            if features.ndim <= 1 or (features.ndim == 2 and features.shape[-1] == 1):
                features = None
            else:
                 if features.ndim == 3 and features.shape[0] == 1:
                     features = features.squeeze(0)
                 
                 T = frames.shape[0]
                 feat = features
                 
                 if feat.ndim == 2:
                     if feat.shape[1] == T:
                         features = feat.T 
                     elif feat.shape[0] == T:
                         features = feat
                     else:
                         features = None
                 else:
                     features = None
                     
        volumes_list = []
        if pred_vol is not None:
            volumes_list = (pred_vol.squeeze() * 300.0).tolist()
            if not isinstance(volumes_list, list): volumes_list = [volumes_list]
            
        phase_vel_list = []
        if pred_phase_vel is not None:
            pv_sq = pred_phase_vel.squeeze()
            if pv_sq.ndim == 0: phase_vel_list = [pv_sq.item()]
            else: phase_vel_list = pv_sq.tolist()
                     
        return binary_mask, mask_prob_squeezed, features, volumes_list, phase_vel_list


class StreamingPyTorchInference:
    """
    PyTorch inference engine for STREAMING mode echo segmentation.
    Retains hidden states sequentially.
    """
    def __init__(self, config_path: str, checkpoint_path: str, use_gpu: bool = True):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        logger.info(f"Loading PyTorch STREAMING model on {self.device}...")
        
        cfg = load_config(config_path)
        model_name = cfg.get("model", {}).get("name", "segment_tracker")
        model_cls = get_model_class(model_name)
        self.model = model_cls.from_config(cfg)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.to(self.device)
        self.model.eval()
        self.hidden_state = None

    def reset_state(self):
        self.hidden_state = None

    def preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess single frame.
        Takes (H, W) or (C, H, W).
        Returns (1, 3, H, W) tensor
        """
        if frame.ndim == 2:
            frame = frame[np.newaxis, :, :] 
            frame = np.repeat(frame, 3, axis=0) 
        elif frame.ndim == 3:
            if frame.shape[0] == 1:
                frame = np.repeat(frame, 3, axis=0)
            elif frame.shape[0] > 3:
                frame = frame[:3, :, :]
            
        frame = frame.astype(np.float32)
        if frame.max() > 1.0:
            frame = frame / 255.0
            
        return torch.from_numpy(frame[np.newaxis, :, :, :]).to(self.device)

    def predict_step(self, frame_raw: np.ndarray):
        input_tensor = self.preprocess(frame_raw)
        
        with torch.no_grad():
            if hasattr(self.model, "forward_step"):
                outputs = self.model.forward_step(input_tensor, self.hidden_state)
                self.hidden_state = outputs["hidden_state"]
            elif hasattr(self.model, "step"):
                outputs, self.hidden_state = self.model.step(input_tensor, self.hidden_state)
            else:
                raise AttributeError(f"Model {type(self.model).__name__} lacks 'forward_step' or 'step' method.")
            
            mask_logits = outputs.get("mask_logits")
            pred_vol = outputs.get("pred_vol")
            pred_phase_vel = outputs.get("pred_phase_vel", None)
            features = outputs.get("hidden_features", None)
            
            mask_prob = torch.sigmoid(mask_logits) if mask_logits is not None else None
            
        result = {}
        if mask_prob is not None:
            result["mask_prob"] = mask_prob.cpu().numpy()
        if pred_vol is not None:
            result["pred_vol"] = pred_vol.cpu().numpy()
        if pred_phase_vel is not None:
            result["pred_phase_vel"] = pred_phase_vel.cpu().numpy()
        if features is not None:
            result["features"] = features.cpu().numpy()
            
        if self.hidden_state and isinstance(self.hidden_state, dict):
            coeff_ema = self.hidden_state.get("coeff_ema")
            if coeff_ema is not None:
                result["coeff_ema"] = coeff_ema.cpu().numpy()
            
        return result

    def segment_video(self, frames: np.ndarray, threshold: float = 0.5):
        self.reset_state()
        
        masks_list = []
        features_list = []
        volumes_list = []
        phase_vel_list = []
        coeff_ema_list = []
        
        T = frames.shape[0]
        
        for t in range(T):
            frame = frames[t]
            outputs = self.predict_step(frame)
            
            if "mask_prob" in outputs:
                masks_list.append(outputs["mask_prob"].squeeze()) 
            if "features" in outputs:
                features_list.append(outputs["features"].squeeze())
            if "pred_vol" in outputs:
                volumes_list.append(float(outputs["pred_vol"].squeeze()) * 300.0)
            if "pred_phase_vel" in outputs:
                phase_vel_list.append(outputs["pred_phase_vel"].squeeze())
            if "coeff_ema" in outputs:
                coeff_ema_list.append(outputs["coeff_ema"].squeeze())
                
        mask_probs = np.stack(masks_list, axis=0) if masks_list else None
        binary_mask = (mask_probs > threshold).astype(np.uint8) if mask_probs is not None else None
        
        if features_list:
            features = np.stack(features_list, axis=0) 
        else:
            features = None
            
        return binary_mask, mask_probs, features, volumes_list, phase_vel_list

    def predict(self, input_data: np.ndarray):
        """
        Compatibility method for benchmark.
        input_data: (1, 3, T, H, W)
        """
        frames_t = input_data[0].transpose(1, 0, 2, 3) # (T, 3, H, W)
        _, mask_probs, features, volumes, phase_vels = self.segment_video(frames_t)
        mask_prob_out = mask_probs[np.newaxis, np.newaxis, :, :, :]
        features_out = features[np.newaxis, :, :] if features is not None else None
        return mask_prob_out, features_out, volumes, phase_vels

def run_inference_pytorch(
    config,
    checkpoint,
    video=None,
    output="results/inference_output.mp4",
    benchmark=False,
    num_frames=32,
    img_size=112,
    warmup=5,
    iterations=20,
    csv="datasets/echonet-dynamic/FileListwFrames112.csv",
    calibrate=False,
    audit_stats="audit_stats.json",
    calibration_samples=50,
    streaming=False,
    use_gpu=True
):
    if not Path(checkpoint).exists():
        logger.error(f"Checkpoint not found: {checkpoint}")
        return
        
    if streaming:
        engine = StreamingPyTorchInference(config, checkpoint, use_gpu=use_gpu)
    else:
        engine = PyTorchSegmentationInference(config, checkpoint, use_gpu=use_gpu)
        
    if calibrate:
        if not csv:
            logger.error("Calibration requires --csv")
            return
        run_calibration(engine, csv, audit_stats, max_samples=calibration_samples)
        return

    if benchmark:
        run_benchmark(engine, num_frames=num_frames, img_size=img_size, warmup_iters=warmup, benchmark_iters=iterations)
    
    if video:
        clinical_data = None
        if csv and Path(csv).exists():
            video_filename = Path(video).stem
            clinical_data = load_clinical_data(csv, video_filename)
            
        process_video(
            engine, 
            video, 
            output, 
            clinical_data=clinical_data,
            audit_stats_path=audit_stats
        )
