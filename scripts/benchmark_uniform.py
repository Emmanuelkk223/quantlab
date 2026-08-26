"""
quantlab/scripts/benchmark_uniform.py

Runs a full benchmark matrix comparing FP32, FP16, INT8, and INT4 uniform precisions.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from quantlab.quantization.engine import QuantizationEngine
from quantlab.hardware.latency_profiler import LatencyProfiler


def run_uniform_benchmark():
    model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    precisions = ["fp32", "fp16", "int8", "int4"]
    batch_size = 16
    seq_len = 128

    profiler = LatencyProfiler()
    results = {}

    print("=" * 75)
    print("      QUANTLAB UNIFORM PRECISION BENCHMARK (FP32/FP16/INT8/INT4)     ")
    print("=" * 75)

    for prec in precisions:
        print(f"\n[+] Processing Precision Mode: {prec.upper()}")
        wrapper = QuantizationEngine.load_quantized_model(model_name, precision=prec)
        dummy_inputs = wrapper.generate_dummy_inputs(
            batch_size=batch_size, seq_len=seq_len
        )

        bench_data = profiler.profile_module(
            model=wrapper.model,
            sample_inputs=dummy_inputs,
            warmup_steps=30,
            active_steps=100,
        )

        bench_data["model_size_mb"] = wrapper.get_model_size_mb()["model_size_mb"]
        results[prec] = bench_data

        del wrapper
        torch.cuda.empty_cache()

    # --- Scientific Report Generation ---
    fp32_lat = results["fp32"]["mean_latency_ms"]
    fp32_vram = results["fp32"]["peak_vram_mb"]

    print("\n" + "=" * 75)
    print("                 UNIFORM QUANTIZATION EXPERIMENTAL RESULTS               ")
    print("=" * 75)
    print(
        f"{'Precision':<10} | {'Mean Lat (ms)':<14} | {'Speedup':<10} | {'Peak VRAM (MB)':<15} | {'VRAM Saved':<10}"
    )
    print("-" * 75)

    for prec in precisions:
        lat = results[prec]["mean_latency_ms"]
        vram = results[prec]["peak_vram_mb"]
        speedup = fp32_lat / lat
        vram_saved = (1.0 - (vram / fp32_vram)) * 100.0

        print(
            f"{prec.upper():<10} | {lat:<14.3f} | {speedup:<10.2f}x | {vram:<15.2f} | {vram_saved:<9.2f}%"
        )

    print("=" * 75 + "\n")


if __name__ == "__main__":
    run_uniform_benchmark()
