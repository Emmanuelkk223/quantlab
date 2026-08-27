"""
quantlab/hardware/latency_profiler.py
"""

import torch
import numpy as np
from typing import Dict, Any


class LatencyProfiler:
    def __init__(self, device: torch.device):
        self.device = device
        self.start_evt = torch.cuda.Event(enable_timing=True)
        self.end_evt = torch.cuda.Event(enable_timing=True)

    def profile_module(
        self,
        model: torch.nn.Module,
        sample_inputs: Dict[str, torch.Tensor],
        warmup_steps: int = 30,
        active_steps: int = 100,
    ) -> Dict[str, Any]:
        """
        Executes isolated profiling with warm-up and comprehensive statistical reporting.
        """
        model.eval()

        # 1. Warm-up Phase (P0 state stabilization)
        with torch.no_grad():
            for _ in range(warmup_steps):
                _ = model(**sample_inputs)

        torch.cuda.synchronize()

        # 2. Active Measurement Phase
        latencies_ms = []
        with torch.no_grad():
            for _ in range(active_steps):
                self.start_evt.record()
                _ = model(**sample_inputs)
                self.end_evt.record()
                torch.cuda.synchronize()

                latencies_ms.append(self.start_evt.elapsed_time(self.end_evt))

        # 3. Statistical Aggregation (Issue 15 Fix)
        lat_array = np.array(latencies_ms)
        stats = {
            "mean_ms": float(np.mean(lat_array)),
            "std_ms": float(np.std(lat_array)),
            "p50_ms": float(np.median(lat_array)),
            "p95_ms": float(np.percentile(lat_array, 95)),
            "p99_ms": float(np.percentile(lat_array, 99)),
            "iterations": active_steps,
            "warmup_iterations": warmup_steps,
        }

        return stats
