"""
Edge AI runtime benchmarking tool.
Measures inference latency, throughput, and memory usage across TFLite, ONNX, and OpenVINO runtimes.
"""
import gc
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False

try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    try:
        import tensorflow.lite as tflite
        TFLITE_AVAILABLE = True
    except ImportError:
        tflite = None
        TFLITE_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class BenchmarkConfig:
    num_warmup: int = 5
    num_runs: int = 100
    input_shape: Tuple[int, ...] = (1, 320, 320, 3)
    input_dtype: str = "float32"
    device: str = "cpu"


@dataclass
class BenchmarkResult:
    runtime: str
    model_path: str
    num_runs: int
    mean_ms: float
    std_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    fps: float
    peak_memory_mb: float
    input_shape: Tuple[int, ...]
    device: str

    def to_dict(self) -> Dict:
        return {
            "runtime": self.runtime,
            "model": os.path.basename(self.model_path),
            "num_runs": self.num_runs,
            "mean_ms": self.mean_ms,
            "std_ms": self.std_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "fps": self.fps,
            "peak_memory_mb": self.peak_memory_mb,
            "input_shape": list(self.input_shape),
            "device": self.device,
        }


def _get_memory_mb() -> float:
    if PSUTIL_AVAILABLE:
        return psutil.Process(os.getpid()).memory_info().rss / 1e6
    return 0.0


class TFLiteBenchmark:
    """Benchmarks TFLite models using the TFLite runtime interpreter."""

    def __init__(self, model_path: str, num_threads: int = 4):
        self.model_path = model_path
        self.num_threads = num_threads

    def run(self, config: BenchmarkConfig) -> Optional[BenchmarkResult]:
        if not TFLITE_AVAILABLE:
            logger.error("TFLite runtime not available.")
            return None
        if not os.path.exists(self.model_path):
            logger.error("Model not found: %s", self.model_path)
            return None
        interpreter = tflite.Interpreter(
            model_path=self.model_path,
            num_threads=self.num_threads,
        )
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        input_idx = input_details[0]["index"]
        dummy = np.random.rand(*config.input_shape).astype(config.input_dtype)
        for _ in range(config.num_warmup):
            interpreter.set_tensor(input_idx, dummy)
            interpreter.invoke()
        mem_before = _get_memory_mb()
        latencies = []
        for _ in range(config.num_runs):
            t0 = time.perf_counter()
            interpreter.set_tensor(input_idx, dummy)
            interpreter.invoke()
            latencies.append((time.perf_counter() - t0) * 1000)
        peak_mem = _get_memory_mb() - mem_before
        lat = np.array(latencies)
        return BenchmarkResult(
            runtime="TFLite",
            model_path=self.model_path,
            num_runs=config.num_runs,
            mean_ms=round(float(lat.mean()), 3),
            std_ms=round(float(lat.std()), 3),
            p50_ms=round(float(np.percentile(lat, 50)), 3),
            p95_ms=round(float(np.percentile(lat, 95)), 3),
            p99_ms=round(float(np.percentile(lat, 99)), 3),
            fps=round(1000.0 / float(lat.mean()), 1),
            peak_memory_mb=round(peak_mem, 2),
            input_shape=config.input_shape,
            device=config.device,
        )


class ONNXBenchmark:
    """Benchmarks ONNX models using ONNX Runtime."""

    def __init__(self, model_path: str, providers: Optional[List[str]] = None):
        self.model_path = model_path
        self.providers = providers or ["CPUExecutionProvider"]

    def run(self, config: BenchmarkConfig) -> Optional[BenchmarkResult]:
        if not ORT_AVAILABLE:
            logger.error("onnxruntime not available.")
            return None
        if not os.path.exists(self.model_path):
            logger.error("Model not found: %s", self.model_path)
            return None
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = os.cpu_count() or 4
        session = ort.InferenceSession(self.model_path, sess_options=sess_opts,
                                       providers=self.providers)
        input_name = session.get_inputs()[0].name
        dummy = np.random.rand(*config.input_shape).astype(config.input_dtype)
        for _ in range(config.num_warmup):
            session.run(None, {input_name: dummy})
        mem_before = _get_memory_mb()
        latencies = []
        for _ in range(config.num_runs):
            t0 = time.perf_counter()
            session.run(None, {input_name: dummy})
            latencies.append((time.perf_counter() - t0) * 1000)
        peak_mem = _get_memory_mb() - mem_before
        lat = np.array(latencies)
        return BenchmarkResult(
            runtime="ONNX Runtime",
            model_path=self.model_path,
            num_runs=config.num_runs,
            mean_ms=round(float(lat.mean()), 3),
            std_ms=round(float(lat.std()), 3),
            p50_ms=round(float(np.percentile(lat, 50)), 3),
            p95_ms=round(float(np.percentile(lat, 95)), 3),
            p99_ms=round(float(np.percentile(lat, 99)), 3),
            fps=round(1000.0 / float(lat.mean()), 1),
            peak_memory_mb=round(peak_mem, 2),
            input_shape=config.input_shape,
            device="GPU" if "CUDAExecutionProvider" in self.providers else "CPU",
        )


class RuntimeBenchmarkSuite:
    """Orchestrates benchmarks across multiple runtimes and formats a comparison report."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[BenchmarkResult] = []

    def add_tflite(self, model_path: str, num_threads: int = 4) -> None:
        bench = TFLiteBenchmark(model_path=model_path, num_threads=num_threads)
        result = bench.run(self.config)
        if result:
            self.results.append(result)

    def add_onnx(self, model_path: str, use_gpu: bool = False) -> None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        bench = ONNXBenchmark(model_path=model_path, providers=providers)
        result = bench.run(self.config)
        if result:
            self.results.append(result)

    def print_report(self) -> None:
        if not self.results:
            print("No benchmark results to display.")
            return
        header = f"{'Runtime':<18} {'Model':<25} {'Mean ms':<10} {'P95 ms':<10} {'FPS':<8} {'Mem MB':<10}"
        print(header)
        print("-" * len(header))
        for r in sorted(self.results, key=lambda x: x.mean_ms):
            model_name = os.path.basename(r.model_path)[:24]
            print(f"{r.runtime:<18} {model_name:<25} {r.mean_ms:<10} {r.p95_ms:<10} {r.fps:<8} {r.peak_memory_mb:<10}")

    def to_dataframe(self):
        try:
            import pandas as pd
            return pd.DataFrame([r.to_dict() for r in self.results])
        except ImportError:
            return [r.to_dict() for r in self.results]


if __name__ == "__main__":
    config = BenchmarkConfig(
        num_warmup=3,
        num_runs=20,
        input_shape=(1, 320, 320, 3),
        input_dtype="float32",
    )
    suite = RuntimeBenchmarkSuite(config)
    print("Edge AI Runtime Benchmark Suite")
    print(f"Config: {config.num_runs} runs, input shape {config.input_shape}")
    print("To run: call suite.add_tflite('model.tflite') or suite.add_onnx('model.onnx')")
    print("Then call suite.print_report() to see the comparison table.")
    print("\nExample (no real models):")
    print(f"{'Runtime':<18} {'Model':<25} {'Mean ms':<10} {'P95 ms':<10} {'FPS':<8}")
    print("-" * 70)
    dummy_results = [
        ("TFLite INT8", "model_int8.tflite", 12.4, 15.2, 80.6),
        ("TFLite FP16", "model_fp16.tflite", 18.1, 22.0, 55.2),
        ("ONNX Runtime", "model.onnx", 22.5, 28.3, 44.4),
    ]
    for runtime, model, mean_ms, p95, fps in dummy_results:
        print(f"{runtime:<18} {model:<25} {mean_ms:<10} {p95:<10} {fps:<8}")
