#!/usr/bin/env python3
"""
PyTorch Inference Script for Echo Segmentation Models.

Usage:
    python3 scripts/infer.py --config configs/temporal_segmentation.yaml \
                             --checkpoint outputs/best.pt \
                             --video input.avi
"""

import os
os.environ['MPLBACKEND'] = 'Agg'
import argparse
import sys

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference.inference_pytorch import run_inference_pytorch

def parse_args():
    parser = argparse.ArgumentParser(description="Echo Segmentation PyTorch Inference Runner")
    
    # Required Args
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt")
    
    # Inference Args
    parser.add_argument("--video", type=str, help="Path to input video")
    parser.add_argument("--output", type=str, default="results/inference_output.mp4", help="Output path")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--calibrate", action="store_true", help="Run calibration")
    parser.add_argument("--csv", type=str, default="datasets/echonet-dynamic/FileListwFrames112.csv", help="Dataset CSV")
    parser.add_argument("--audit-stats", type=str, default="audit_stats.json", help="Audit stats JSON")
    parser.add_argument("--calibration-samples", type=int, default=50, help="Number of calibration samples")
    
    parser.add_argument("--num-frames", type=int, default=32, help="Number of frames")
    parser.add_argument("--img-size", type=int, default=112, help="Image resolution")
    parser.add_argument("--streaming", action="store_true", help="Enable streaming mode")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=20, help="Benchmark iterations")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference instead of GPU")

    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Starting PyTorch Inference with checkpoint: {args.checkpoint}")
    run_inference_pytorch(
        config=args.config,
        checkpoint=args.checkpoint,
        video=args.video,
        output=args.output,
        benchmark=args.benchmark,
        num_frames=args.num_frames,
        img_size=args.img_size,
        warmup=args.warmup,
        iterations=args.iterations,
        csv=args.csv,
        calibrate=args.calibrate,
        audit_stats=args.audit_stats,
        calibration_samples=args.calibration_samples,
        streaming=args.streaming,
        use_gpu=not args.cpu
    )

if __name__ == "__main__":
    main()
