"""
quantlab/hardware/memory_tracker.py

Tracks exact allocated, reserved, and board-level VRAM metrics
for PyTorch models on CUDA hardware.
"""

import torch
import pynvml
from typing import Dict, Optional


class MemoryTracker:
    """
    Monitors VRAM footprints across PyTorch dynamic dynamic allocations
    and physical NVIDIA GPU memory.
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.is_cuda = self.device.type == "cuda"
        self._nvml_initialized = False

        if self.is_cuda:
            try:
                pynvml.nvmlInit()
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(
                    self.device.index or 0
                )
                self._nvml_initialized = True
            except Exception as err:
                print(f"[WARNING] NVML direct hardware tracking disabled: {err}")

    def reset(self) -> None:
        """Resets peak memory allocation trackers and empties unused cache."""
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()

    def get_stats(self) -> Dict[str, float]:
        """
        Gathers current and peak memory usage in Megabytes (MB).

        Returns:
            Dict containing allocated, reserved, peak allocated, and physical board VRAM.
        """
        if not self.is_cuda:
            return {
                "allocated_mb": 0.0,
                "peak_allocated_mb": 0.0,
                "reserved_mb": 0.0,
                "physical_used_mb": 0.0,
            }

        bytes_to_mb = 1024.0 * 1024.0
        stats = {
            "allocated_mb": torch.cuda.memory_allocated(self.device) / bytes_to_mb,
            "peak_allocated_mb": torch.cuda.max_memory_allocated(self.device)
            / bytes_to_mb,
            "reserved_mb": torch.cuda.memory_reserved(self.device) / bytes_to_mb,
            "physical_used_mb": 0.0,
        }

        if self._nvml_initialized:
            try:
                info = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
                stats["physical_used_mb"] = info.used / bytes_to_mb
            except Exception:
                stats["physical_used_mb"] = 0.0

        return stats

    def __del__(self):
        if self._nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
