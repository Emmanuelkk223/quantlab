"""
quantlab/scripts/benchmark_baseline.py

Executes FP32 and FP16 benchmark baselines for DistilBERT on target hardware.
"""

from pathlib import Path
import sys

# Dynamically resolve project root directory regardless of working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from quantlab.models.base_model import BaseModelWrapper
from quantlab.hardware.latency_profiler import LatencyProfiler


def run_milestone1_benchmark():
    model_name = "distilbert-base-uncased"
    batch_size = 16
    seq_len = 128
    warmup_steps = 30
    active_steps = 100

    print("=" * 65)
    print(f"       QUANTLAB BENCHMARK: {model_name.upper()}       ")
    print("=" * 65)

    profiler = LatencyProfiler()
    print(f"[+] Execution Device : {profiler.device}")

    # --- FP32 Execution ---
    print("\n[1/2] Benchmarking FP32 Baseline...")
    model_fp32 = BaseModelWrapper(model_name, torch_dtype=torch.float32)
    dummy_inputs = model_fp32.generate_dummy_inputs(
        batch_size=batch_size, seq_len=seq_len
    )

    fp32_params = model_fp32.get_parameter_count()
    fp32_size = model_fp32.get_model_size_mb()

    fp32_bench = profiler.profile_module(
        model=model_fp32,
        sample_inputs=dummy_inputs,
        warmup_steps=warmup_steps,
        active_steps=active_steps,
    )

    # Cleanup memory before next run
    del model_fp32
    torch.cuda.empty_cache()

    # --- FP16 Execution ---
    print("[2/2] Benchmarking FP16 Baseline...")
    model_fp16 = BaseModelWrapper(model_name, torch_dtype=torch.float16)

    fp16_bench = profiler.profile_module(
        model=model_fp16,
        sample_inputs=dummy_inputs,
        warmup_steps=warmup_steps,
        active_steps=active_steps,
    )

    del model_fp16
    torch.cuda.empty_cache()

    # --- Scientific Report Generation ---
    vram_saved_pct = (
        1.0 - (fp16_bench["peak_vram_mb"] / fp32_bench["peak_vram_mb"])
    ) * 100.0
    latency_speedup = fp32_bench["mean_latency_ms"] / fp16_bench["mean_latency_ms"]

    print("\n" + "=" * 65)
    print("                 MILESTONE 1 EXPERIMENTAL RESULTS                ")
    print("=" * 65)
    print(f"Total Parameters         : {fp32_params['total_params']:,}")
    print(f"Theoretical Size (FP32)  : {fp32_size['model_size_mb']:.2f} MB")
    print(f"Theoretical Size (FP16)  : {fp32_size['model_size_mb'] / 2.0:.2f} MB")
    print("-" * 65)
    print(f"{'Metric':<28} | {'FP32 Baseline':<14} | {'FP16 Baseline':<14}")
    print("-" * 65)
    print(
        f"{'Mean Latency (ms)':<28} | {fp32_bench['mean_latency_ms']:<14.3f} | {fp16_bench['mean_latency_ms']:<14.3f}"
    )
    print(
        f"{'P95 Latency (ms)':<28} | {fp32_bench['p95_latency_ms']:<14.3f} | {fp16_bench['p95_latency_ms']:<14.3f}"
    )
    print(
        f"{'P99 Latency (ms)':<28} | {fp32_bench['p99_latency_ms']:<14.3f} | {fp16_bench['p99_latency_ms']:<14.3f}"
    )
    print(
        f"{'Throughput (samples/sec)':<28} | {fp32_bench['throughput_samples_per_sec']:<14.1f} | {fp16_bench['throughput_samples_per_sec']:<14.1f}"
    )
    print(
        f"{'Peak VRAM Allocated (MB)':<28} | {fp32_bench['peak_vram_mb']:<14.2f} | {fp16_bench['peak_vram_mb']:<14.2f}"
    )
    print(
        f"{'Reserved VRAM (MB)':<28} | {fp32_bench['reserved_vram_mb']:<14.2f} | {fp16_bench['reserved_vram_mb']:<14.2f}"
    )
    print("=" * 65)
    print(f"[SUMMARY] FP16 Peak VRAM Reduction : {vram_saved_pct:.2f}%")
    print(f"[SUMMARY] FP16 Speedup Factor     : {latency_speedup:.2f}x")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_milestone1_benchmark()
