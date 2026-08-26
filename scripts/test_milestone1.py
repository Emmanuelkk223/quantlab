"""
scripts/test_milestone1.py

Milestone 1 Validation Script: Verifies the hardware profiler by measuring
FP32 vs FP16 baselines of DistilBERT on your GPU.
"""

import sys
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Register framework root directory
sys.path.append(".")
from quantlab.hardware.profiler import HardwareProfiler


def run_baseline_comparison():
    model_name = "distilbert-base-uncased"
    batch_size = 16
    seq_len = 128

    print(f"--- Initializing Benchmark for Model: {model_name} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Generate static synthetic input batch (Batch Size x Sequence Length)
    dummy_text = ["Quantization research requires precise measurement."] * batch_size
    inputs = tokenizer(
        dummy_text, padding="max_length", max_length=seq_len, return_tensors="pt"
    )

    profiler = HardwareProfiler()
    print(f"Target Hardware Device: {profiler.device}\n")

    # --- FP32 Baseline Execution ---
    print("[1/2] Loading Model in FP32 Precision...")
    model_fp32 = AutoModelForSequenceClassification.from_pretrained(model_name)
    fp32_results = profiler.benchmark_inference(
        model=model_fp32, dummy_input=inputs, warmup_steps=30, active_steps=100
    )
    del model_fp32  # Free GPU VRAM
    torch.cuda.empty_cache()

    # --- FP16 Baseline Execution ---
    print("[2/2] Loading Model in FP16 Precision...")
    model_fp16 = AutoModelForSequenceClassification.from_pretrained(
        model_name, torch_dtype=torch.float16
    )

    # Ensure inputs are cast to FP16 where floating point inputs exist
    fp16_inputs = {k: v for k, v in inputs.items()}

    fp16_results = profiler.benchmark_inference(
        model=model_fp16, dummy_input=fp16_inputs, warmup_steps=30, active_steps=100
    )
    del model_fp16
    torch.cuda.empty_cache()

    # --- Print Scientific Results Report ---
    print("\n" + "=" * 50)
    print("           QUANTLAB MILESTONE 1 BENCHMARK           ")
    print("=" * 50)
    print(f"{'Metric':<25} | {'FP32 Baseline':<12} | {'FP16 Half':<12}")
    print("-" * 55)
    print(
        f"{'Mean Latency (ms)':<25} | {fp32_results['mean_latency_ms']:<12.3f} | {fp16_results['mean_latency_ms']:<12.3f}"
    )
    print(
        f"{'P95 Latency (ms)':<25} | {fp32_results['p95_latency_ms']:<12.3f} | {fp16_results['p95_latency_ms']:<12.3f}"
    )
    print(
        f"{'Throughput (samples/s)':<25} | {fp32_results['throughput_samples_per_sec']:<12.1f} | {fp16_results['throughput_samples_per_sec']:<12.1f}"
    )
    print(
        f"{'Peak VRAM (MB)':<25} | {fp32_results['peak_vram_mb']:<12.2f} | {fp16_results['peak_vram_mb']:<12.2f}"
    )
    print("=" * 50)

    # Memory & Speedup Calculations
    vram_reduction = (
        1 - (fp16_results["peak_vram_mb"] / fp32_results["peak_vram_mb"])
    ) * 100
    speedup = fp32_results["mean_latency_ms"] / fp16_results["mean_latency_ms"]
    print(
        f"\n[ANALYSIS] VRAM Reduction: {vram_reduction:.2f}% | Latency Speedup: {speedup:.2f}x"
    )


if __name__ == "__main__":
    run_baseline_comparison()
