"""
quantlab/hardware/profiler.py

Hardware profiling utilities for measuring exact GPU latency, memory footprint,
and throughput for Transformer models under PyTorch execution.
"""

import time
import torch
import numpy as np
from typing import Dict, Any, Callable, Optional, Tuple


class HardwareProfiler:
    """
    Scientific profiler for PyTorch models targeting GPU execution.
    Handles CUDA warmup, event synchronization, peak VRAM measurement,
    and statistical latency calculations.
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.is_cuda = self.device.type == "cuda"

        if not self.is_cuda:
            print(
                "[WARNING] Running profiler on CPU. VRAM tracking and CUDA sync are disabled."
            )

    def reset_memory_stats() -> None:
        """Resets PyTorch CUDA memory allocator peak statistics."""
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()

    def get_memory_stats(self) -> Dict[str, float]:
        """
        Retrieves allocated and peak VRAM memory statistics in Megabytes (MB).

        Returns:
            Dict containing allocated, reserved, and peak memory in MB.
        """
        if not self.is_cuda:
            return {"allocated_mb": 0.0, "peak_mb": 0.0, "reserved_mb": 0.0}

        bytes_in_mb = 1024.0 * 1024.0
        return {
            "allocated_mb": torch.cuda.memory_allocated(self.device) / bytes_in_mb,
            "peak_mb": torch.cuda.max_memory_allocated(self.device) / bytes_in_mb,
            "reserved_mb": torch.cuda.memory_reserved(self.device) / bytes_in_mb,
        }

    def benchmark_inference(
        self,
        model: torch.nn.Module,
        dummy_input: Dict[str, torch.Tensor],
        warmup_steps: int = 30,
        active_steps: int = 100,
    ) -> Dict[str, Any]:
        """
        Runs an isolated, synchronized benchmarking loop over a PyTorch model.

        Args:
            model: PyTorch model instance (eval mode assumed).
            dummy_input: Dictionary of input tensors (e.g., input_ids, attention_mask).
            warmup_steps: Number of unmeasured iterations to warm up GPU context/clocks.
            active_steps: Number of measured iterations for statistical stability.

        Returns:
            Dictionary of memory footprint, throughput, and latency percentiles.
        """
        model.to(self.device)
        model.eval()

        # Move input tensors to target hardware device
        inputs = {k: v.to(self.device) for k, v in dummy_input.items()}

        # 1. Warm-up Phase: Forces GPU to max P-state and handles JIT warm-ups
        with torch.no_grad():
            for _ in range(warmup_steps):
                _ = model(**inputs)
                if self.is_cuda:
                    torch.cuda.synchronize(self.device)

        # 2. Reset Memory Counters right before active measurement
        self.reset_memory_stats()

        latencies_ms = []

        # 3. Active Benchmarking Phase
        with torch.no_grad():
            for _ in range(active_steps):
                if self.is_cuda:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)

                    start_event.record()
                    _ = model(**inputs)
                    end_event.record()

                    # Force CPU to wait until the GPU reaches end_event
                    torch.cuda.synchronize(self.device)
                    elapsed_time = start_event.elapsed_time(end_event)
                    latencies_ms.append(elapsed_time)
                else:
                    start_time = time.perf_counter()
                    _ = model(**inputs)
                    end_time = time.perf_counter()
                    latencies_ms.append((end_time - start_time) * 1000.0)

        # 4. Fetch memory statistics peak during active execution
        mem_stats = self.get_memory_stats()

        # 5. Compute latency distributions
        latencies = np.array(latencies_ms)
        batch_size = next(iter(dummy_input.values())).shape[0]
        total_samples = batch_size * active_steps
        total_time_sec = np.sum(latencies) / 1000.0

        return {
            "batch_size": batch_size,
            "mean_latency_ms": float(np.mean(latencies)),
            "std_latency_ms": float(np.std(latencies)),
            "p50_latency_ms": float(np.percentile(latencies, 50)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "throughput_samples_per_sec": float(total_samples / total_time_sec),
            "peak_vram_mb": mem_stats["peak_mb"],
            "allocated_vram_mb": mem_stats["allocated_mb"],
        }
