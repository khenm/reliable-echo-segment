"""
ONNX Runtime Inference Script for Echo Segmentation with SelfAuditor Wealth Plotting.

Runs optimized CPU inference using ONNX Runtime for real-time segmentation.
Supports SelfAuditor for Out-of-Distribution detection via Martingale Wealth.

Usage:
    # 1. Calibration (Optional but recommended)
    python3 inference_onnx.py --model echo_segmentation.onnx --calibrate --csv val_dataset.csv

    # 2. Inference with Wealth Plot
    python3 inference_onnx.py --model echo_segmentation.onnx \
                              --video path/to/video.avi \
                              --output output_mask.mp4 \
                              --audit-stats audit_stats.json
"""

import argparse
import os
import time
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from scipy.special import expit, softmax
from ..utils.logging import get_logger

logger = get_logger()

def parse_args():
    parser = argparse.ArgumentParser(description="ONNX Runtime inference for echo segmentation")
    parser.add_argument(
        "--model",
        type=str,
        default="echo_segmentation.onnx",
        help="Path to ONNX model"
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Path to input video (optional, for video inference)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/inference_output.mp4",
        help="Path to output video with overlay"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark to measure inference speed"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=32,
        help="Number of frames for benchmark"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=112,
        help="Input image size"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup iterations for benchmark"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="datasets/echonet-dynamic/FileListwFrames112.csv",
        help="Path to FileListwFrames112.csv"
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run calibration on the dataset defined by --csv"
    )
    parser.add_argument(
        "--audit-stats",
        type=str,
        default="audit_stats.json",
        help="Path to save/load calibration statistics (mu_source, epsilon)"
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=50,
        help="Number of samples to use for calibration (-1 for all)"
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming inference mode (requires streaming ONNX model)"
    )
    return parser.parse_args()


class NumpySelfAuditor:
    """
    Numpy implementation of SelfAuditor for ONNX Inference.
    Calculates Martingale Wealth based on Entropy and Feature Drift.
    """
    
    def __init__(self, feature_dim=None, alpha=0.5, delta=0.05, lambda_val=0.5):
        self.alpha = alpha
        self.delta = delta
        self.lambda_val = lambda_val
        self.threshold = 1.0 / delta
        
        self.martingale = 1.0
        self.collapsed = False
        self.wealth_history = []
        
        # Calibration Stats
        self.mu_source = None
        self.epsilon = 0.5  # Default conservative value if not calibrated
        self.max_ent = np.log(2) # Default binary entropy max
        
    def load_stats(self, stats_path: str):
        """Load calibration statistics from JSON."""
        if os.path.exists(stats_path):
            try:
                with open(stats_path, 'r') as f:
                    stats = json.load(f)
                if stats['mu_source'] is not None:
                    self.mu_source = np.array(stats['mu_source'])
                else:
                    self.mu_source = None
                self.epsilon = stats['epsilon']
                self.max_ent = stats.get('max_ent', np.log(2))
                logger.info(f"Loaded auditor stats from {stats_path}")
                logger.info(f"Epsilon: {self.epsilon:.4f}, MaxEnt: {self.max_ent:.4f}")
            except Exception as e:
                logger.warning(f"Failed to load stats: {e}. Using defaults.")
        else:
            logger.warning(f"Stats file {stats_path} not found. Using defaults (Wealth may be inaccurate).")

    def _compute_entropy(self, inputs: np.ndarray, is_logits=True):
        """
        Compute entropy of logits or probabilities (Numpy).
        Args:
            inputs: (C, H, W) or (1, H, W) - Logits or Probabilities
            is_logits: If True, applies sigmoid/softmax. If False, assumes valid probs.
        Returns:
            entropy: scalar mean entropy
        """
        if is_logits:
            if inputs.ndim == 3 and inputs.shape[0] > 1:
                probs = softmax(inputs, axis=0)
                max_ent = np.log(inputs.shape[0])
            else:
                probs = expit(inputs)
                max_ent = np.log(2)
        else:
            probs = inputs
            if inputs.ndim == 3 and inputs.shape[0] > 1:
                max_ent = np.log(inputs.shape[0])
            else:
                max_ent = np.log(2)

        # Compute Entropy
        if inputs.ndim == 3 and inputs.shape[0] > 1:
             ent = -np.sum(probs * np.log(probs + 1e-10), axis=0)
        else:
             ent = -(probs * np.log(probs + 1e-10) + (1-probs) * np.log(1-probs + 1e-10))
             
        return np.mean(ent), max_ent

    def update(self, inputs: np.ndarray, features: np.ndarray, is_logits=True):
        """
        Update Martingale Wealth.
        Args:
            inputs: (C, H, W) for current frame (Logits or Probs)
            features: (D,) or (D, 1) feature vector for current frame
            is_logits: Whether inputs are logits
        Returns:
            wealth: Current martingale value
        """
        # 1. Entropy
        entropy, max_ent = self._compute_entropy(inputs, is_logits=is_logits)
        norm_entropy = entropy / max_ent if max_ent > 0 else 0
        
        # 2. Drift
        drift = 0.0
        if self.mu_source is not None and features is not None:
            feat_flat = features.flatten()
            if self.mu_source.shape == feat_flat.shape:
                # Normalize
                feat_norm = feat_flat / (np.linalg.norm(feat_flat) + 1e-10)
                # Cosine distance (1 - similarity)
                drift = 1.0 - np.dot(feat_norm, self.mu_source)
            else:
                 # Shape mismatch, ignore drift
                 pass
        
        # 3. Composite Score
        # If no drift available (no calibration), rely on entropy
        if self.mu_source is None:
             score = norm_entropy
        else:
             score = self.alpha * norm_entropy + (1 - self.alpha) * drift
        
        # 4. Update Wealth
        # Bet: 1 + lambda * (score - epsilon)
        bet = 1 + self.lambda_val * (score - self.epsilon)
        bet = max(0.1, bet) # Prevent wealth from truly hitting 0 or becoming negative
        
        self.martingale *= bet
        self.wealth_history.append(self.martingale)
        
        if self.martingale > self.threshold:
            self.collapsed = True
            
        return self.martingale


class ONNXSegmentationInference:
    """
    ONNX Runtime inference engine for echo segmentation.
    
    Provides optimized CPU inference with graph optimizations and
    optional threading configuration.
    """
    
    def __init__(
        self,
        model_path: str,
        num_threads: int = None,
        use_gpu: bool = False
    ):
        """
        Initialize ONNX inference session.
        
        Args:
            model_path: Path to .onnx model file
            num_threads: Number of threads for inference (None = auto)
            use_gpu: Whether to use GPU if available
        """
        self.model_path = model_path
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        if num_threads is not None:
            sess_options.intra_op_num_threads = num_threads
            sess_options.inter_op_num_threads = num_threads
        
        if use_gpu:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        
        logger.info(f"Loading ONNX model from {model_path}")
        self.session = ort.InferenceSession(
            model_path,
            sess_options,
            providers=providers
        )
        
        self.input_name = self.session.get_inputs()[0].name
        
        # Handle multiple outputs (mask, features)
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        input_shape = self.session.get_inputs()[0].shape
        logger.info(f"Model input: {self.input_name} with shape {input_shape}")
        logger.info(f"Model outputs: {self.output_names}")
        logger.info(f"Providers: {self.session.get_providers()}")
    
    def preprocess(self, frames: np.ndarray) -> np.ndarray:
        """
        Preprocess input frames for model inference.
        
        Args:
            frames: Input array of shape (T, H, W) or (T, C, H, W)
        
        Returns:
            Preprocessed array of shape (1, 3, T, H, W)
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
        
        frames = np.expand_dims(frames, axis=0)
        frames = frames.transpose(0, 2, 1, 3, 4)
        
        return frames
    
    def predict(self, input_data: np.ndarray):
        """
        Run inference on preprocessed input.
        
        Args:
            input_data: Preprocessed input of shape (B, C, T, H, W)
        
        Returns:
            mask_prob: Segmentation mask of shape (B, 1, T, H, W)
            features: Features if available, else None
        """
        inputs = {self.input_name: input_data}
        outputs = self.session.run(None, inputs)
        
        mask_prob = outputs[0]
        features = outputs[1] if len(outputs) > 1 else None
        
        return mask_prob, features
    
    def segment_video(self, frames: np.ndarray, threshold: float = 0.5):
        """
        Full pipeline: preprocess, predict, postprocess.
        
        Args:
            frames: Raw input frames (T, H, W) or (T, C, H, W)
            threshold: Binary threshold for mask
        
        Returns:
            Binary segmentation masks (T, H, W)
            Probability masks (T, H, W)
            Features (T, D) or None
        """
        input_data = self.preprocess(frames)
        mask_prob, features = self.predict(input_data)
        
        # Squeeze batch dims
        mask_prob_squeezed = mask_prob.squeeze(0).squeeze(0) # (T, H, W)
        binary_mask = (mask_prob_squeezed > threshold).astype(np.uint8)
        
        # Handle features
        # Handle features
        if features is not None:
            # Model may output (B, D, T), (B, T, D), or scalar (B,)/(1,)
            if features.ndim <= 1 or (features.ndim == 2 and features.shape[-1] == 1):
                # Scalar output (e.g. predicted EF) - not usable as per-frame features
                logger.debug(f"Features is scalar-like with shape {features.shape}, skipping feature processing")
                features = None
            else:
                 # Robust handling of (1, D, T) vs (1, T, D)
                 # Squeeze batch if present
                 if features.ndim == 3 and features.shape[0] == 1:
                     features = features.squeeze(0)
                 
                 # Heuristic: The dimension matching T (Time) is the temporal dim
                 T = frames.shape[0]
                 feat = features
                 
                 if feat.ndim == 2:
                     if feat.shape[1] == T:
                         # Shape is (D, T) -> Transpose to (T, D)
                         features = feat.T 
                     elif feat.shape[0] == T:
                         # Shape is (T, D) -> Keep
                         features = feat
                     else:
                         logger.warning(f"Feature shape {feat.shape} does not match frame count {T}")
                         features = None
                 else:
                     # Unexpected shape
                     features = None
        
        # We need raw logits for entropy, but if model outputs Sigmoid(logits), 
        # we can invert it or just use probabilities for entropy calc (works fine).
        # Wrapper currently returns Sigmoid(logits). 
        # For Auditor _compute_entropy binary, it takes logits or probabilities if carefully handled.
        # My implementation of _compute_entropy above does expit(logits). 
        # If input is already PROBS, we should change that.
        # Let's adjust `process_video` to pass logits if possible, OR
        # Change Auditor to accept probs.
        
        return binary_mask, mask_prob_squeezed, features


class StreamingONNXInference:
    """
    ONNX Runtime inference engine for STREAMING echo segmentation.
    Manages hidden state internally.
    """

    def __init__(
        self,
        model_path: str,
        num_threads: int = None,
        use_gpu: bool = False
    ):
        self.model_path = model_path
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        if num_threads is not None:
            sess_options.intra_op_num_threads = num_threads
            sess_options.inter_op_num_threads = num_threads
        
        if use_gpu:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        
        logger.info(f"Loading STREAMING ONNX model from {model_path}")
        self.session = ort.InferenceSession(
            model_path,
            sess_options,
            providers=providers
        )
        
        # Init state
        self.hidden_state = None
        # Default hidden shape (Layers=2, B=1, Hidden=128) - derived from model def or dummy input
        # We will init it on first run or use what's passed
        self.hidden_shape = (2, 1, 128) 
        
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        logger.info(f"Model inputs: {self.input_names}")
        logger.info(f"Model outputs: {self.output_names}")

    def reset_state(self):
        """Reset hidden state to zeros."""
        self.hidden_state = np.zeros(self.hidden_shape, dtype=np.float32)

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess single frame.
        Args:
            frame: (H, W) or (C, H, W)
        Returns:
            (1, C, H, W)
        """
        if frame.ndim == 2:
            frame = frame[np.newaxis, :, :] # (1, H, W)
            frame = np.repeat(frame, 3, axis=0) # (3, H, W)
        elif frame.ndim == 3:
            if frame.shape[0] == 1:
                frame = np.repeat(frame, 3, axis=0)
            elif frame.shape[0] > 3:
                frame = frame[:3, :, :]
            # Else assumed (3, H, W)
            
        frame = frame.astype(np.float32)
        if frame.max() > 1.0:
            frame = frame / 255.0
            
        return frame[np.newaxis, :, :, :] # (1, 3, H, W)

    def predict_step(self, frame_raw: np.ndarray):
        """
        Run inference for one frame.
        """
        if self.hidden_state is None:
            self.reset_state()
            
        input_tensor = self.preprocess(frame_raw)
        
        inputs = {
            self.input_names[0]: input_tensor,
            self.input_names[1]: self.hidden_state
        }
        
        # Outputs: mask, volume, h_new, features (optional)
        outputs = self.session.run(None, inputs)
        
        mask_prob = outputs[0]     # (1, 1, H, W)
        pred_vol = outputs[1]      # (1, 1)
        self.hidden_state = outputs[2] # (L, 1, H) - Update state
        
        features = None
        if len(outputs) > 3:
            features = outputs[3] # (1, D)
            
        return mask_prob, pred_vol, features

    def segment_video(self, frames: np.ndarray, threshold: float = 0.5):
        """
        Process entire video frame-by-frame (simulated streaming).
        Args:
            frames: (T, H, W) or (T, C, H, W)
        Returns:
            binary_mask: (T, H, W)
            mask_probs: (T, H, W)
            features: (T, D) or None
        """
        self.reset_state()
        
        masks_list = []
        features_list = []
        vols_list = [] # We might need volumes too, but interface asks for masks/feats
        
        # Handle input shape
        if frames.ndim == 3: # (T, H, W)
             pass
        elif frames.ndim == 4: # (T, C, H, W)
             pass # Logic handles it inside loop (we slice [t])
             
        T = frames.shape[0]
        
        for t in range(T):
            frame = frames[t]
            mask_prob, vol, feat = self.predict_step(frame)
            
            masks_list.append(mask_prob.squeeze()) # (H, W)
            if feat is not None:
                features_list.append(feat.squeeze()) # (D,)
                
        # Stack
        mask_probs = np.stack(masks_list, axis=0) # (T, H, W)
        binary_mask = (mask_probs > threshold).astype(np.uint8)
        
        if features_list:
            features = np.stack(features_list, axis=0) # (T, D)
        else:
            features = None
            
        return binary_mask, mask_probs, features

    def predict(self, input_data: np.ndarray):
        """
        Compatibility method for benchmark.
        Args:
            input_data: (1, 3, T, H, W)
        Returns:
            mask_prob: (1, 1, T, H, W)
            features: (1, T, D)
        """
        # (1, 3, T, H, W) -> (T, 3, H, W)
        frames_t = input_data[0].transpose(1, 0, 2, 3) 
        
        _, mask_probs, features = self.segment_video(frames_t)
        
        # mask_probs: (T, H, W) -> (1, 1, T, H, W)
        mask_prob_out = mask_probs[np.newaxis, np.newaxis, :, :, :]
        
        # features: (T, D) -> (1, T, D)
        features_out = None
        if features is not None:
             features_out = features[np.newaxis, :, :]
             
        return mask_prob_out, features_out



def run_benchmark(
    inference_engine: ONNXSegmentationInference,
    num_frames: int = 32,
    img_size: int = 112,
    warmup_iters: int = 5,
    benchmark_iters: int = 20
):
    """
    Run inference benchmark to measure latency and throughput.
    """
    logger.info(f"Running benchmark: {num_frames} frames @ {img_size}x{img_size}")
    logger.info(f"Warmup: {warmup_iters} iterations, Benchmark: {benchmark_iters} iterations")
    
    dummy_input = np.random.rand(1, 3, num_frames, img_size, img_size).astype(np.float32)
    
    logger.info("Warming up...")
    for _ in range(warmup_iters):
        _ = inference_engine.predict(dummy_input)
    
    logger.info("Benchmarking...")
    latencies = []
    
    for i in range(benchmark_iters):
        start = time.perf_counter()
        _ = inference_engine.predict(dummy_input)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
    
    latencies = np.array(latencies)
    
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    
    fps = 1000.0 / mean_latency
    frames_per_second = (num_frames * 1000.0) / mean_latency
    
    logger.info("=" * 50)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 50)
    logger.info(f"Input shape: (1, 1, {num_frames}, {img_size}, {img_size})")
    logger.info(f"Mean latency: {mean_latency:.2f} ms (±{std_latency:.2f})")
    logger.info(f"Min latency:  {min_latency:.2f} ms")
    logger.info(f"Max latency:  {max_latency:.2f} ms")
    logger.info(f"P50 latency:  {p50:.2f} ms")
    logger.info(f"P95 latency:  {p95:.2f} ms")
    logger.info(f"P99 latency:  {p99:.2f} ms")
    logger.info("-" * 50)
    logger.info(f"Batches per second: {fps:.2f}")
    logger.info(f"Frames per second:  {frames_per_second:.2f}")
    logger.info("=" * 50)
    
    return {
        "mean_latency_ms": mean_latency,
        "std_latency_ms": std_latency,
        "min_latency_ms": min_latency,
        "max_latency_ms": max_latency,
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "p99_latency_ms": p99,
        "batches_per_second": fps,
        "frames_per_second": frames_per_second
    }


def compute_volume_from_mask(mask: np.ndarray, pixel_spacing: float = 1.0) -> float:
    """
    Compute estimated volume from binary segmentation mask using Simpson's Method.
    
    Args:
        mask: Binary mask (H, W)
        pixel_spacing: Physical size of pixel in mm (default 1.0 for normalized)
    
    Returns:
        Estimated volume in arbitrary units
    """
    area = np.sum(mask > 0.5)
    
    if area == 0:
        return 0.0
    
    coords = np.where(mask > 0.5)
    if len(coords[0]) == 0:
        return 0.0
    
    major_axis = max(coords[0].max() - coords[0].min(), coords[1].max() - coords[1].min()) + 1
    
    volume = (8.0 * area * area) / (3.0 * np.pi * major_axis + 1e-8)
    
    return volume * (pixel_spacing ** 3)


def compute_ef_from_volumes(volumes: list) -> dict:
    """
    Compute Ejection Fraction from per-frame volume curve.

    Identifies ED (max volume) and ES (min volume) frames,
    then computes EF = (EDV - ESV) / EDV * 100.

    Args:
        volumes: List of per-frame volume values.

    Returns:
        Dict with ed_frame, es_frame, edv, esv, ef (percentage).
        Returns None if volumes is empty or all-zero.
    """
    if not volumes or max(volumes) < 1e-8:
        logger.warning("Cannot compute EF: empty or zero volume curve.")
        return None

    volumes_arr = np.array(volumes)
    ed_frame = int(np.argmax(volumes_arr))
    es_frame = int(np.argmin(volumes_arr))
    edv = float(volumes_arr[ed_frame])
    esv = float(volumes_arr[es_frame])

    ef = (edv - esv) / (edv + 1e-8) * 100.0
    ef = float(np.clip(ef, 0.0, 100.0))

    return {
        "ed_frame": ed_frame,
        "es_frame": es_frame,
        "edv": edv,
        "esv": esv,
        "ef": ef,
    }


def create_volume_plot_frame(
    volumes: list, 
    wealths: list,
    current_idx: int, 
    plot_width: int, 
    plot_height: int,
    ef_info: dict = None
) -> np.ndarray:
    """
    Create a matplotlib plot frame showing volume AND wealth over time.
    
    Args:
        volumes: List of volume values
        wealths: List of wealth values
        current_idx: Current frame index
        plot_width: Width of the plot in pixels
        plot_height: Height of the plot in pixels
    
    Returns:
        RGB numpy array of the plot
    """
    base_size = min(plot_width, plot_height)
    title_fontsize = max(6, int(base_size * 0.08))
    label_fontsize = max(5, int(base_size * 0.06))
    tick_fontsize = max(4, int(base_size * 0.05))
    
    fig, ax1 = plt.subplots(figsize=(plot_width / 100, plot_height / 100), dpi=100)
    
    fig.patch.set_facecolor('#1a1a2e')
    ax1.set_facecolor('#1a1a2e')
    
    x_vals = list(range(len(volumes)))
    
    # Plot Volume (Left Axis)
    color_vol = '#00FF00'
    ax1.plot(x_vals, volumes, color=color_vol, linewidth=1.5, label='Volume')
    ax1.set_xlabel('Frame', fontsize=label_fontsize, color='white')
    ax1.set_ylabel('Volume', fontsize=label_fontsize, color=color_vol)
    ax1.tick_params(axis='y', labelcolor=color_vol, labelsize=tick_fontsize)
    ax1.tick_params(axis='x', colors='white', labelsize=tick_fontsize)
    ax1.grid(True, alpha=0.3, color='gray')
    
    for spine in ax1.spines.values():
        spine.set_color('gray')

    # Plot Wealth (Right Axis)
    ax2 = ax1.twinx()
    color_wealth = '#9467bd' # Purple
    ax2.plot(x_vals, wealths, color=color_wealth, linewidth=1.5, linestyle=':', label='Wealth')
    ax2.set_ylabel('Wealth (OOD)', fontsize=label_fontsize, color=color_wealth)
    ax2.tick_params(axis='y', labelcolor=color_wealth, labelsize=tick_fontsize)
    ax2.spines['right'].set_color(color_wealth)
    
    # Current Position Indicator
    ax1.axvline(x=current_idx, color='#FF4444', linewidth=1, linestyle='--', alpha=0.8)
    
    if len(volumes) > 1:
        ax1.set_xlim(-1, len(volumes) + 5)
        # Dynamic Scalings
        ax1.set_ylim(0, max(volumes) * 1.2 + 1)
        ax2.set_ylim(0, max(max(wealths), 2.0) * 1.2) # Wealth usually starts at 1
    
    ax1.set_title('Volume & Auditor Wealth', fontsize=title_fontsize, color='white', fontweight='bold')
    
    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.2)
    
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    plot_array = np.asarray(buf)
    plot_array = plot_array[:, :, :3]
    
    plt.close(fig)
    
    return plot_array


def save_final_plot(volumes: list, wealths: list, output_path: str, ef_info: dict = None):
    """
    Save the complete volume/wealth time series as a high-quality plot.
    """
    
    fig, ax1 = plt.subplots(figsize=(10, 4), dpi=150)
    
    fig.patch.set_facecolor('#1a1a2e')
    ax1.set_facecolor('#1a1a2e')
    
    x_vals = list(range(len(volumes)))
    
    # Volume
    c1 = '#00FF00'
    ax1.plot(x_vals, volumes, color=c1, linewidth=2, label='Volume')
    ax1.fill_between(x_vals, volumes, alpha=0.2, color=c1)
    ax1.set_xlabel('Frame', fontsize=12, color='white')
    ax1.set_ylabel('Volume (a.u.)', fontsize=12, color=c1)
    ax1.tick_params(axis='y', labelcolor=c1, colors='white')
    ax1.tick_params(axis='x', colors='white')
    
    # Wealth
    ax2 = ax1.twinx()
    c2 = '#9467bd'
    ax2.plot(x_vals, wealths, color=c2, linewidth=2, linestyle='--', label='Wealth')
    ax2.set_ylabel('Wealth (OOD Score)', fontsize=12, color=c2)
    ax2.tick_params(axis='y', labelcolor=c2, colors='white')
    
    if ef_info is not None:
        ax1.axvline(x=ef_info['ed_frame'], color='#FF6B6B', linewidth=1.5,
                    linestyle='-', alpha=0.8, label=f"ED #{ef_info['ed_frame']}")
        ax1.axvline(x=ef_info['es_frame'], color='#4ECDC4', linewidth=1.5,
                    linestyle='-', alpha=0.8, label=f"ES #{ef_info['es_frame']}")
        ax1.legend(loc='upper left', fontsize=8, facecolor='#1a1a2e',
                   edgecolor='gray', labelcolor='white')
        title = f"Predicted Volume & Wealth | EF={ef_info['ef']:.1f}%"
    else:
        title = 'Predicted Volume & Auditor Wealth'

    ax1.set_title(title, fontsize=14, color='white', fontweight='bold')
    ax1.grid(True, alpha=0.3, color='gray')
    
    for spine in ax1.spines.values():
        spine.set_color('gray')
    ax2.spines['right'].set_color(c2)
    
    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    
    logger.info(f"Saved analysis plot to {output_path}")


def load_clinical_data(csv_path: str, video_filename: str) -> dict:
    """
    Load EDFrame and ESFrame for a given video from CSV.
    """
    import pandas as pd
    
    try:
        df = pd.read_csv(csv_path)
        row = df[df['FileName'] == video_filename]
        
        if row.empty:
            logger.warning(f"Video {video_filename} not found in {csv_path}")
            return None
            
        data = row.iloc[0].to_dict()
        return data
        
    except Exception as e:
        logger.error(f"Error loading clinical data: {e}")
        return None


def run_calibration(inference_engine, csv_path, audit_stats_path="audit_stats.json", max_samples=50):
    """
    Run calibration loop to compute mu_source and epsilon.
    """
    logger.info("Starting Calibration...")
    import pandas as pd
    
    if not os.path.exists(csv_path):
        logger.error(f"Dataset CSV not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    # Try to deduce dataset directory from CSV path
    csv_dir = Path(csv_path).parent
    dataset_dir = csv_dir / "Videos"
    if not dataset_dir.exists():
        # Fallback to current dir / Videos or just csv_dir
        if (csv_dir / "Videos").exists():
            dataset_dir = csv_dir / "Videos"
        else:
            dataset_dir = csv_dir # Maybe videos are here?
            
    logger.info(f"Looking for videos in: {dataset_dir}")
    
    features_list = []
    entropies_list = []
    max_ent_val = np.log(2)
    
    auditor = NumpySelfAuditor()
    
    count = 0
    for _, row in df.iterrows():
        if max_samples != -1 and count >= max_samples:
            break
            
        fname = str(row['FileName'])
        if not fname.endswith('.avi'):
             fname += ".avi"
             
        fpath = dataset_dir / fname
        
        if not fpath.exists():
             continue
             
        # Process Video
        cap = cv2.VideoCapture(str(fpath))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            if len(frame.shape) == 3: gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else: gray = frame
            frames.append(cv2.resize(gray, (112, 112)))
        cap.release()
        
        if not frames: continue
        
        frames_arr = np.stack(frames, axis=0)
        
        # Inference
        try:
            _, mask_probs, features = inference_engine.segment_video(frames_arr)
            
            # Features - only usable if (T, D) shaped
            if features is not None and features.ndim == 2 and features.shape[1] > 1:
                norm_feats = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-10)
                features_list.append(norm_feats)
            
            # Entropy
            # Mask probs are (T, H, W).
            # Fix A: Use probabilities directly, no round-trip logit conversion.
            
            for t in range(len(mask_probs)):
                 ent, mx = auditor._compute_entropy(mask_probs[t], is_logits=False)
                 entropies_list.append(ent)
                 max_ent_val = mx
                 
            count += 1
            if count % 5 == 0:
                logger.info(f"Calibrated {count} videos...")
                
        except Exception as e:
            logger.warning(f"Error calibrating {fname}: {e}")
            continue

    if not entropies_list:
        logger.error("No entropy data collected during calibration.")
        return

    all_entropies = np.array(entropies_list)
    norm_entropies = all_entropies / max_ent_val

    if features_list:
        # Full calibration with features + entropy
        all_feats = np.concatenate(features_list, axis=0)  # (TotalFrames, D)
        mu_source = np.mean(all_feats, axis=0)
        mu_source = mu_source / (np.linalg.norm(mu_source) + 1e-10)

        drift = 1.0 - np.dot(all_feats, mu_source)
        scores = 0.5 * norm_entropies + 0.5 * drift

        stats = {
            "mu_source": mu_source.tolist(),
            "epsilon": float(np.mean(scores) + 3 * np.std(scores)),
            "max_ent": float(max_ent_val)
        }
    else:
        # Entropy-only calibration (model outputs scalar features)
        logger.warning("No per-frame features available. Using entropy-only calibration.")
        scores = norm_entropies

        stats = {
            "mu_source": None,
            "epsilon": float(np.mean(scores) + 3 * np.std(scores)),
            "max_ent": float(max_ent_val)
        }

    with open(audit_stats_path, 'w') as f:
        json.dump(stats, f)

    logger.info(f"Calibration saved to {audit_stats_path}")
    logger.info(f"Epsilon: {stats['epsilon']:.4f}")


def process_video(
    inference_engine: ONNXSegmentationInference,
    video_path: str,
    output_path: str = None,
    clinical_data: dict = None,
    audit_stats_path: str = "audit_stats.json"
):
    """
    Process a video file with segmentation + wealth tracking.
    """
    logger.info(f"Processing video: {video_path}")
    
    # Initialize Auditor
    auditor = NumpySelfAuditor(alpha=0.5, delta=0.05)
    auditor.load_stats(audit_stats_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return
    
    frames = []
    original_frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        original_frames.append(frame.copy())
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        resized = cv2.resize(gray, (112, 112))
        frames.append(resized)
    
    cap.release()
    frames_array = np.stack(frames, axis=0)
    
    # Inference
    start = time.perf_counter()
    binary_masks, prob_masks, features = inference_engine.segment_video(frames_array)
    end = time.perf_counter()
    
    fps = len(frames) / (end - start)
    
    # Compute Volumes & Wealth
    volumes = []
    wealths = []
    
    for t in range(len(prob_masks)):
        vol = compute_volume_from_mask(prob_masks[t])
        volumes.append(vol)
        
        feat_t = features[t] if features is not None else None
        w = auditor.update(prob_masks[t], feat_t, is_logits=False)
        wealths.append(w)
    
    # Compute EF from volume curve
    ef_info = compute_ef_from_volumes(volumes)
    if ef_info is not None:
        logger.info("=" * 50)
        logger.info("EJECTION FRACTION PREDICTION")
        logger.info("=" * 50)
        logger.info(f"ED Frame: {ef_info['ed_frame']}  |  EDV: {ef_info['edv']:.2f}")
        logger.info(f"ES Frame: {ef_info['es_frame']}  |  ESV: {ef_info['esv']:.2f}")
        logger.info(f"Predicted EF: {ef_info['ef']:.1f}%")
        if clinical_data is not None and 'EF' in clinical_data:
            gt_ef = float(clinical_data['EF'])
            logger.info(f"Ground Truth EF: {gt_ef:.1f}%")
            logger.info(f"Error: {abs(ef_info['ef'] - gt_ef):.1f}%")
        logger.info("=" * 50)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        orig_h, orig_w = original_frames[0].shape[:2]
        min_output_size = 400
        scale_factor = max(1.0, min_output_size / min(orig_h, orig_w))
        output_w = int(orig_w * scale_factor)
        output_h = int(orig_h * scale_factor)
        
        plot_height = max(150, int(output_h * 0.5))
        plot_width = output_w
        combined_height = output_h + plot_height
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (output_w, combined_height))
        
        COLOR_MASK = np.array([0, 255, 0], dtype=np.uint8)
        COLOR_ED = (107, 107, 255)  # Red (BGR) for ED frame
        COLOR_ES = (196, 205, 78)   # Teal (BGR) for ES frame
        
        for t in range(len(original_frames)):
            frame = original_frames[t].copy()
            if scale_factor > 1.0:
                frame = cv2.resize(frame, (output_w, output_h))
            
            # Overlay
            mask_resized = cv2.resize(prob_masks[t], (output_w, output_h))
            mask_binary = (mask_resized > 0.5).astype(np.uint8)
            overlay = frame.copy()
            overlay[mask_binary == 1] = COLOR_MASK
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
            
            # ED/ES frame border highlight
            if ef_info is not None:
                border_w = max(3, int(output_w * 0.01))
                if t == ef_info['ed_frame']:
                    cv2.rectangle(frame, (0, 0), (output_w - 1, output_h - 1),
                                  COLOR_ED, border_w)
                elif t == ef_info['es_frame']:
                    cv2.rectangle(frame, (0, 0), (output_w - 1, output_h - 1),
                                  COLOR_ES, border_w)
            
            # Text Overlay
            font_scale = max(0.5, output_h / 700.0)
            thickness = max(1, int(font_scale * 2))
            text_x = int(output_w * 0.02)
            text_y_vol = int(output_h * 0.1)
            text_y_fps = int(output_h * 0.18)
            text_y_ef = int(output_h * 0.26)
            
            vol_text = f"Vol: {volumes[t]:.1f}"
            fps_text = f"FPS: {fps:.1f}"
            
            # Shadow + white text for Vol and FPS
            for txt, y_pos in [(vol_text, text_y_vol), (fps_text, text_y_fps)]:
                cv2.putText(frame, txt, (text_x, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2)
                cv2.putText(frame, txt, (text_x, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            
            # EF overlay
            if ef_info is not None:
                ef_text = f"EF: {ef_info['ef']:.1f}%"
                cv2.putText(frame, ef_text, (text_x, text_y_ef),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2)
                cv2.putText(frame, ef_text, (text_x, text_y_ef),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
            
            # ED/ES frame label
            if ef_info is not None:
                label = None
                if t == ef_info['ed_frame']:
                    label = "ED"
                    label_color = COLOR_ED
                elif t == ef_info['es_frame']:
                    label = "ES"
                    label_color = COLOR_ES
                if label is not None:
                    label_scale = font_scale * 1.5
                    label_thickness = max(2, int(label_scale * 2))
                    label_x = output_w - int(output_w * 0.15)
                    label_y = int(output_h * 0.12)
                    cv2.putText(frame, label, (label_x, label_y),
                                cv2.FONT_HERSHEY_SIMPLEX, label_scale,
                                (0, 0, 0), label_thickness + 3)
                    cv2.putText(frame, label, (label_x, label_y),
                                cv2.FONT_HERSHEY_SIMPLEX, label_scale,
                                label_color, label_thickness)
            
            # Plot
            plot_frame = create_volume_plot_frame(
                volumes[:t + 1], wealths[:t + 1], t,
                plot_width, plot_height, ef_info=ef_info
            )
            plot_frame_bgr = cv2.cvtColor(plot_frame, cv2.COLOR_RGB2BGR)
            plot_frame_resized = cv2.resize(plot_frame_bgr, (plot_width, plot_height))
            
            combined = np.vstack([frame, plot_frame_resized])
            out.write(combined)
            
        out.release()
        save_final_plot(
            volumes, wealths,
            output_path.replace(".mp4", "_analysis.png"),
            ef_info=ef_info
        )
    
    return binary_masks, prob_masks, volumes, ef_info


def run_inference(
    model,
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
    streaming=False
):
    if not Path(model).exists():
        logger.error(f"Model not found: {model}")
        return
    
    # Select engine based on mode
    if streaming:
        engine = StreamingONNXInference(model)
    else:
        engine = ONNXSegmentationInference(model)
    
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