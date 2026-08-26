"""
quantlab/scripts/benchmark_mixed_precision.py

Evaluates SST-2 task accuracy, VRAM footprint, and latency of a Sensitivity-Guided
Mixed-Precision configuration against uniform baselines.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import BitsAndBytesConfig, AutoModelForSequenceClassification
from quantlab.models.base_model import BaseModelWrapper
from quantlab.experiments.sensitivity_analysis import SensitivityAnalyzer
from quantlab.quantization.precision_map import PrecisionAllocator
from quantlab.datasets.glue_loader import GLUEDataLoader
from quantlab.evaluation.metrics import TaskEvaluator
from quantlab.hardware.latency_profiler import LatencyProfiler


def run_mixed_precision_benchmark():
    model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=" * 75)
    print("      QUANTLAB SENSITIVITY-GUIDED MIXED-PRECISION EXPERIMENT      ")
    print("=" * 75)

    # 1. Run Layer Sensitivity Analysis
    print("\n[1/4] Running Automated Layer Sensitivity Sweep...")
    analyzer = SensitivityAnalyzer(model_name=model_name)
    dummy_input = analyzer.baseline_wrapper.generate_dummy_inputs(
        batch_size=16, seq_len=128
    )
    sensitivity_profile = analyzer.compute_layer_mse_sensitivity(dummy_input)

    # 2. Generate Mixed-Precision Allocation (Protect Top-4 Sensitive Layers)
    print("\n[2/4] Generating Precision Allocation Policy...")
    allocator = PrecisionAllocator(sensitivity_profile)
    policy = allocator.generate_protected_allocation(top_k_sensitive=4)

    protected_layers = policy["protected_layer_names"]
    print(f"[+] Protected High-Sensitivity Layers (Kept in FP16):")
    for name in protected_layers:
        print(f"    - {name}")

    # 3. Load Datasets and Profiling Tools
    print("\n[3/4] Initializing Validation DataLoader and Hardware Profiler...")
    data_loader_builder = GLUEDataLoader(model_name_or_path=model_name, batch_size=32)
    val_loader = data_loader_builder.get_dataloader()
    evaluator = TaskEvaluator(device=device)
    profiler = LatencyProfiler(device=device)

    # 4. Construct BitsAndBytes Mixed Precision Config (INT4 base with FP16 skipped layers)
    print("\n[4/4] Instantiating Mixed-Precision Transformer Architecture...")

    # Extract short module names for BitsAndBytes skip list
    # e.g., 'distilbert.transformer.layer.1.ffn.lin2' -> 'layer.1.ffn.lin2'
    skip_modules = []
    for full_name in protected_layers:
        parts = full_name.split(".")
        if len(parts) > 2:
            skip_modules.append(".".join(parts[-3:]))
        else:
            skip_modules.append(full_name)

    mixed_bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=skip_modules,  # Protect sensitive layers
    )

    mixed_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        quantization_config=mixed_bnb_config,
        device_map={"": device.index or 0},
    )

    # Measure Accuracy
    print(" -> Evaluating Mixed-Precision Task Accuracy on SST-2...")
    mixed_acc_stats = evaluator.evaluate_accuracy(
        model=mixed_model, dataloader=val_loader
    )

    # Measure Latency and Peak VRAM
    print(" -> Profiling Mixed-Precision Hardware Performance...")
    sample_bench_inputs = {
        "input_ids": torch.randint(0, 1000, (16, 128), device=device),
        "attention_mask": torch.ones((16, 128), device=device),
    }
    mixed_hw_stats = profiler.profile_module(
        model=mixed_model,
        sample_inputs=sample_bench_inputs,
        warmup_steps=30,
        active_steps=100,
    )

    # Print Final Comparative Report
    print("\n" + "=" * 75)
    print("            MIXED-PRECISION VS UNIFORM EVALUATION REPORT           ")
    print("=" * 75)
    print(f"Protected Layer Count       : {len(skip_modules)} / 38 Linear Layers")
    print(f"SST-2 Validation Accuracy  : {mixed_acc_stats['accuracy'] * 100.0:.2f}%")
    print(f"Evaluation Cross-Entropy    : {mixed_acc_stats['eval_loss']:.4f}")
    print(f"Mean Inference Latency      : {mixed_hw_stats['mean_latency_ms']:.3f} ms")
    print(f"Peak VRAM Footprint         : {mixed_hw_stats['peak_vram_mb']:.2f} MB")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    run_mixed_precision_benchmark()
