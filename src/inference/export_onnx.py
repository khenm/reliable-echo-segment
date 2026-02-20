"""
ONNX Export Script for Echo Segmentation Models.

Exports PyTorch models to ONNX format with optimizations for CPU inference.
Usage:
    python3 export_onnx.py --config configs/temporal_segmentation_debug.yaml \
                           --checkpoint outputs/best.pt \
                           --output echo_segmentation.onnx
"""

import argparse
from pathlib import Path

import torch
import onnx

from ..utils.config_loader import load_config
from ..utils.logging import get_logger
from ..registry import get_model_class
import src.models

logger = get_logger()


def parse_args():
    parser = argparse.ArgumentParser(description="Export model to ONNX format")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="echo_segmentation.onnx",
        help="Output ONNX file path"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=19,
        help="ONNX opset version (default: 19)"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=32,
        help="Number of frames for dummy input (default: 32)"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=112,
        help="Input image size (default: 112)"
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Export in streaming mode (takes one frame + hidden state)"
    )
    return parser.parse_args()


class ONNXWrapper(torch.nn.Module):
    """
    Wrapper to export mask_logits and features for ONNX.
    
    The original model returns (mask_logits, volume, ef), but we need
    features for SelfAuditor wealth calculation.
    """
    
    def __init__(self, model: torch.nn.Module, output_mode: str = "mask_features"):
        super().__init__()
        self.model = model
        self.output_mode = output_mode
    
    def forward(self, x: torch.Tensor):
        output = self.model(x)

        # Tracing-safe output normalization
        if isinstance(output, dict):
            mask_logits = output["mask_logits"]
            if "hidden_features" in output:
                features = output["hidden_features"]
            elif "features" in output:
                features = output["features"]
            else:
                features = torch.zeros((x.shape[0], 1, x.shape[2]), device=x.device)
        else:
            # Assuming 'output' is a tuple here
            mask_logits = output[0]
            # Try to grab features from end if available
            features = output[-1] 
        return torch.sigmoid(mask_logits), features


class StreamingONNXWrapper(torch.nn.Module):
    """
    Wrapper for Streaming Inference.
    Inputs: x (B, C, H, W), h_prev (Layers, B, Hidden)
    Outputs: mask_prob (B, 1, H, W), vol (B, 1), h_new (Layers, B, Hidden)
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor = None):
        # Forward Step
        outputs = self.model.forward_step(x, h_prev)
        
        mask_logits = outputs["mask_logits"]
        pred_vol = outputs["pred_vol"]
        h_new = outputs["hidden_state"]
        features = outputs["hidden_features"]

        return torch.sigmoid(mask_logits), pred_vol, h_new, features


def export_model(
    config_path: str,
    checkpoint_path: str,
    output_path: str,
    opset_version: int = 11,
    num_frames: int = 32,
    img_size: int = 112,
    streaming: bool = False
):
    """
    Export PyTorch model to ONNX format.
    
    Args:
        config_path: Path to model config file
        checkpoint_path: Path to trained weights
        output_path: Output ONNX file path
        opset_version: ONNX opset version
        num_frames: Number of input frames
        img_size: Input image size
        streaming: Export in streaming mode
    """
    logger.info(f"Loading config from {config_path}")
    cfg = load_config(config_path)
    
    model_name = cfg.get("model", {}).get("name", "segment_tracker")
    logger.info(f"Creating model: {model_name}")
    
    model_cls = get_model_class(model_name)
    model = model_cls.from_config(cfg)
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    model.to("cpu")
    
    if streaming:
        logger.info("Exporting in STREAMING mode.")
        wrapped_model = StreamingONNXWrapper(model)
        wrapped_model.eval()
        
        # Single frame input: (1, 3, H, W)
        input_names = ["input", "h_prev"]
        output_names = ["mask", "volume", "h_new", "features"]
        
        # Dynamic axes for batch size (time is implicit loop)
        dynamic_axes = {
            "input": {0: "batch_size"},
            "h_prev": {1: "batch_size"},
            "mask": {0: "batch_size"},
            "volume": {0: "batch_size"},
            "h_new": {1: "batch_size"},
            "features": {0: "batch_size"}
        }
        
        logger.info(f"Dummy inputs: x={dummy_input_x.shape}, h={dummy_input_h.shape}")
        
    else:
        logger.info("Exporting in BATCH mode.")
        wrapped_model = ONNXWrapper(model, output_mode="mask_features")
        wrapped_model.eval()
        
        dummy_inputs = torch.randn(1, 3, num_frames, img_size, img_size)
        input_names = ["input"]
        output_names = ["mask", "features"]
        
        dynamic_axes = {
            "input": {0: "batch_size", 2: "num_frames"},
            "mask": {0: "batch_size", 2: "num_frames"},
            "features": {0: "batch_size", 2: "num_frames"}
        }
        logger.info(f"Dummy input shape: {dummy_inputs.shape} (B, C, T, H, W)")

    
    logger.info(f"Exporting to ONNX (opset {opset_version})...")
    
    dynamic_shapes = {k: v for k, v in dynamic_axes.items() if k in input_names}
    export_options = torch.onnx.ExportOptions(
        opset_version=opset_version,
        do_constant_folding=True,
        dynamic_shapes=dynamic_shapes,
        dynamo=True
    )
    
    torch.onnx.export(
        wrapped_model,
        dummy_inputs,
        output_path,
        export_params=True,
        input_names=input_names,
        output_names=output_names,
        export_options=export_options
    )
    
    logger.info(f"✅ Exported to {output_path}")
    
    logger.info("Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    logger.info("✅ ONNX model verification passed")
    
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    logger.info(f"Model size: {file_size_mb:.2f} MB")
    
    return output_path


def run_export(
    config,
    checkpoint,
    output="echo_segmentation.onnx",
    opset=11,
    num_frames=32,
    img_size=112,
    streaming=False
):
    """
    Wrapper function to be called from scripts.
    """
    return export_model(
        config_path=config,
        checkpoint_path=checkpoint,
        output_path=output,
        opset_version=opset,
        num_frames=num_frames,
        img_size=img_size,
        streaming=streaming
    )