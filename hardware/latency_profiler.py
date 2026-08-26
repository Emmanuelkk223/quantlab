"""
quantlab/hardware/latency_profiler.py

Micro-benchmarking profiler for neural network inference latency and throughput.
"""

import time
import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional
from quantlab.hardware.memory_tracker import MemoryTracker


class LatencyProfiler:
    """
    Executes isolated benchmark passes over PyTorch modules to extract
    mean, std, P50, P95, and P99 latency percentiles alongside throughput.
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.is_cuda = self.device.type == "cuda"
        self.memory_tracker = MemoryTracker(device=self.device)

    def profile_module(
        self,
        model: torch.nn.Module,
        sample_inputs: Dict[str, torch.Tensor],
        warmup_steps: int = 30,
        active_steps: int = 100,
    ) -> Dict[str, Any]:
        """
        Profiles a model's forward pass under fixed inputs.

        Args:
            model: PyTorch model evaluated in eval mode.
            sample_inputs: Dictionary of input tensors.
            warmup_steps: Iterations to bring GPU to full P-state.
            active_steps: Measured execution iterations.

        Returns:
            Dictionary containing benchmark statistics.
        """
        model.to(self.device)
        model.eval()

        inputs = {k: v.to(self.device) for k, v in sample_inputs.items()}

        # Determine batch size from the first tensor dimension
        first_tensor = next(iter(inputs.values()))
        batch_size = first_tensor.shape[0]

        # 1. Warm-up Iterations
        with torch.no_grad():
            for _ in range(warmup_steps):
                _ = model(**inputs)
                if self.is_cuda:
                    torch.cuda.synchronize(self.device)

        # 2. Reset Memory Tracking
        self.memory_tracker.reset()
        latencies_ms = []

        # 3. Micro-benchmarking Loop
        with torch.no_grad():
            for _ in range(active_steps):
                if self.is_cuda:
                    start_evt = torch.cuda.Event(enable_timing=True)
                    end_evt = torch.cuda.Event(enable_timing=True)

                    start_evt.record()
                    _ = model(**inputs)
                    end_evt.record()

                    torch.cuda.synchronize(self.device)
                    latencies_ms.append(start_evt.elapsed_time(end_evt))
                else:
                    t0 = time.perf_counter()
                    _ = model(**inputs)
                    t1 = time.perf_counter()
                    latencies_ms.append((t1 - t0) * 1000.0)

        # 4. Fetch Peak VRAM Snapshot
        mem_stats = self.memory_tracker.get_stats()

        # 5. Compute Statistical Metrics
        lat_arr = np.array(latencies_ms)
        total_time_sec = np.sum(lat_arr) / 1000.0
        total_samples = batch_size * active_steps

        return {
            "batch_size": batch_size,
            "warmup_steps": warmup_steps,
            "active_steps": active_steps,
            "mean_latency_ms": float(np.mean(lat_arr)),
            "std_latency_ms": float(np.std(lat_arr)),
            "p50_latency_ms": float(np.percentile(lat_arr, 50)),
            "p95_latency_ms": float(np.percentile(lat_arr, 95)),
            "p99_latency_ms": float(np.percentile(lat_arr, 99)),
            "throughput_samples_per_sec": float(total_samples / total_time_sec),
            "peak_vram_mb": mem_stats["peak_allocated_mb"],
            "reserved_vram_mb": mem_stats["reserved_mb"],
            "physical_used_vram_mb": mem_stats["physical_used_mb"],
        }
