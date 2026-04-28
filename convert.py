"""
Model format conversion utilities for edge AI runtime evaluation.
Converts PyTorch models to ONNX and TFLite formats for cross-runtime benchmarking.
"""
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


@dataclass
class ConversionConfig:
    input_shape: Tuple[int, ...] = (1, 3, 320, 320)  # NCHW for PyTorch
    opset_version: int = 12
    dynamic_axes: bool = True
    output_dir: str = "converted_models"


class PyTorchToONNX:
    """
    Exports a PyTorch model to ONNX format with optional dynamic batch axes.
    Validates the exported model using onnxruntime.
    """

    def __init__(self, model, config: ConversionConfig):
        self.model = model
        self.config = config
        os.makedirs(config.output_dir, exist_ok=True)

    def export(self, output_filename: str = "model.onnx") -> str:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available.")
        if not ONNX_AVAILABLE:
            raise RuntimeError("onnx and onnxruntime required for ONNX export.")
        output_path = os.path.join(self.config.output_dir, output_filename)
        dummy_input = torch.randn(*self.config.input_shape)
        dynamic_axes = None
        if self.config.dynamic_axes:
            dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}
        self.model.eval()
        with torch.no_grad():
            torch.onnx.export(
                self.model,
                dummy_input,
                output_path,
                opset_version=self.config.opset_version,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=dynamic_axes,
                verbose=False,
            )
        logger.info("Exported ONNX model to %s", output_path)
        return output_path

    def validate(self, onnx_path: str) -> Dict:
        """Run a forward pass with onnxruntime and return shape info."""
        if not ONNX_AVAILABLE:
            return {"error": "onnxruntime not available"}
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        dummy = np.random.rand(*self.config.input_shape).astype(np.float32)
        outputs = session.run(None, {"input": dummy})
        return {
            "valid": True,
            "input_shape": list(self.config.input_shape),
            "output_shapes": [list(o.shape) for o in outputs],
        }


class ONNXToTFLite:
    """
    Converts an ONNX model to TFLite via the onnx-tf bridge.
    Requires onnx-tf and tensorflow to be installed.
    """

    def __init__(self, onnx_path: str, output_dir: str = "converted_models"):
        self.onnx_path = onnx_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def convert(self, output_filename: str = "model_from_onnx.tflite") -> Optional[str]:
        try:
            import onnx_tf
        except ImportError:
            logger.error("onnx-tf not installed. Cannot convert ONNX to TFLite.")
            return None
        if not TF_AVAILABLE:
            logger.error("TensorFlow not available.")
            return None
        try:
            import onnx as onnx_lib
            model = onnx_lib.load(self.onnx_path)
            tf_rep = onnx_tf.backend.prepare(model)
            saved_model_dir = os.path.join(self.output_dir, "temp_saved_model")
            tf_rep.export_graph(saved_model_dir)
            converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
            tflite_model = converter.convert()
            out_path = os.path.join(self.output_dir, output_filename)
            with open(out_path, "wb") as f:
                f.write(tflite_model)
            logger.info("TFLite model saved to %s", out_path)
            return out_path
        except Exception as exc:
            logger.error("ONNX to TFLite conversion failed: %s", exc)
            return None


class SimpleConvNet(nn.Module if TORCH_AVAILABLE else object):
    """Minimal convolutional network for format conversion testing."""

    def __init__(self, num_classes: int = 10):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available.")
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def model_size_mb(path: str) -> float:
    if os.path.exists(path):
        return round(os.path.getsize(path) / 1e6, 2)
    return 0.0


if __name__ == "__main__":
    config = ConversionConfig(
        input_shape=(1, 3, 320, 320),
        opset_version=12,
        dynamic_axes=True,
        output_dir="converted_models",
    )
    if TORCH_AVAILABLE and ONNX_AVAILABLE:
        model = SimpleConvNet(num_classes=10)
        exporter = PyTorchToONNX(model=model, config=config)
        onnx_path = exporter.export("simple_conv.onnx")
        print(f"ONNX model exported: {onnx_path} ({model_size_mb(onnx_path)} MB)")
        validation = exporter.validate(onnx_path)
        print("Validation:", validation)
    else:
        print("PyTorch or ONNX not available. Install with:")
        print("  pip install torch onnx onnxruntime")
        print("\nConversionConfig example:")
        print(f"  input_shape: {config.input_shape}")
        print(f"  opset_version: {config.opset_version}")
        print(f"  output_dir: {config.output_dir}")
