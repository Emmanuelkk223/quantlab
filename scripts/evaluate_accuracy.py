"""
quantlab/scripts/evaluate_accuracy.py

Evaluates SST-2 task accuracy across FP32, FP16, INT8, and INT4 uniform precisions.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from quantlab.quantization.engine import QuantizationEngine
from quantlab.datasets.glue_loader import GLUEDataLoader
from quantlab.evaluation.metrics import TaskEvaluator


def run_accuracy_benchmark():
    model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    precisions = ["fp32", "fp16", "int8", "int4"]

    print("=" * 70)
    print("      QUANTLAB TASK ACCURACY EVALUATION (SST-2 VALIDATION)      ")
    print("=" * 70)

    print("[+] Loading SST-2 Validation Dataset...")
    data_loader_builder = GLUEDataLoader(model_name_or_path=model_name, batch_size=32)
    val_loader = data_loader_builder.get_dataloader()

    evaluator = TaskEvaluator()
    results = {}

    for prec in precisions:
        print(f"\n[+] Evaluating Precision Mode: {prec.upper()}")
        wrapper = QuantizationEngine.load_quantized_model(model_name, precision=prec)

        metrics = evaluator.evaluate_accuracy(
            model=wrapper.model, dataloader=val_loader
        )
        results[prec] = metrics

        del wrapper
        torch.cuda.empty_cache()

    # --- Print Accuracy vs Precision Report ---
    fp32_acc = results["fp32"]["accuracy"]

    print("\n" + "=" * 70)
    print("             TASK ACCURACY VS PRECISION RESULTS             ")
    print("=" * 70)
    print(
        f"{'Precision':<10} | {'Accuracy (%)':<15} | {'Acc Drop (%)':<15} | {'Eval Loss':<12}"
    )
    print("-" * 70)

    for prec in precisions:
        acc = results[prec]["accuracy"] * 100.0
        acc_drop = (results["fp32"]["accuracy"] - results[prec]["accuracy"]) * 100.0
        loss = results[prec]["eval_loss"]

        print(f"{prec.upper():<10} | {acc:<15.2f} | {acc_drop:<15.2f} | {loss:<12.4f}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_accuracy_benchmark()
