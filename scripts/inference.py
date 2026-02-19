#!/usr/bin/env python3
"""
Unified script for ONNX Export and Inference.

Usage:
    # Export
    python3 scripts/inference.py --export --config config.yaml --checkpoint model.pt --output model.onnx

    # Inference
    python3 scripts/inference.py --inference --model model.onnx --video input.avi
"""

import argparse
import matplotlib
matplotlib.use('Agg')
import sys
import os

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference.export_onnx import run_export
from src.inference.inference_onnx import run_inference

def parse_args():
    parser = argparse.ArgumentParser(description="Echo Segmentation ONNX Runner")
    
    # Mode selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--export", action="store_true", help="Run in Export mode")
    group.add_argument("--inference", action="store_true", help="Run in Inference mode")

    # Shared / Export Args
    parser.add_argument("--config", type=str, help="Path to config file (Export only)")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint .pt (Export only)")
    parser.add_argument("--output", type=str, default="output.onnx", help="Output path (Export: .onnx, Inference: .mp4)")
    parser.add_argument("--opset", type=int, default=11, help="ONNX Opset version")
    
    # Inference Args
    parser.add_argument("--model", type=str, help="Path to ONNX model (Inference only)")
    parser.add_argument("--video", type=str, help="Path to input video (Inference only)")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--calibrate", action="store_true", help="Run calibration")
    parser.add_argument("--csv", type=str, default="datasets/echonet-dynamic/FileListwFrames112.csv", help="Dataset CSV")
    parser.add_argument("--audit-stats", type=str, default="audit_stats.json", help="Audit stats JSON")
    parser.add_argument("--calibration-samples", type=int, default=50, help="Number of calibration samples")
    
    # Common
    parser.add_argument("--num-frames", type=int, default=32, help="Number of frames")
    parser.add_argument("--img-size", type=int, default=112, help="Image resolution")
    parser.add_argument("--streaming", action="store_true", help="Enable streaming mode")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=20, help="Benchmark iterations")

    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.export:
        if not args.config or not args.checkpoint:
            print("Error: --config and --checkpoint are required for export.")
            return
            
        print(f"Starting Export: {args.checkpoint} -> {args.output}")
        run_export(
            config=args.config,
            checkpoint=args.checkpoint,
            output=args.output,
            opset=args.opset,
            num_frames=args.num_frames,
            img_size=args.img_size,
            streaming=args.streaming
        )
        
    elif args.inference:
        if not args.model:
            print("Error: --model is required for inference.")
            return

        print(f"Starting Inference with model: {args.model}")
        run_inference(
            model=args.model,
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
            streaming=args.streaming
        )

if __name__ == "__main__":
    main()
