"""
quantlab/experiments/sensitivity_analysis.py

Performs isolated single-layer perturbation analysis using simulated NF4 quantization
and real SST-2 calibration data to calculate structural logit MSE.
"""

import torch
import torch.nn as nn
from typing import Dict
from quantlab.datasets.glue_loader import GLUEDataLoader
from quantlab.models.base_model import BaseModelWrapper


class NF4SensitivityProfiler:
    def __init__(self, model_name: str, device: str = "cuda:0", num_samples: int = 256):
        self.device = torch.device(device)
        self.wrapper = BaseModelWrapper(
            model_name, dtype=torch.float32, device=self.device
        )
        self.model = self.wrapper.model
        self.model.eval()

        # Use isolated calibration split
        dataloader_builder = GLUEDataLoader(
            model_name_or_path=model_name, batch_size=32
        )
        calib_loader, _, _ = dataloader_builder.get_splits(
            num_calibration=num_samples, num_search=512
        )

        self.calibration_batches = []
        samples_collected = 0
        for batch in calib_loader:
            self.calibration_batches.append(
                {
                    "input_ids": batch["input_ids"].to(self.device),
                    "attention_mask": batch["attention_mask"].to(self.device),
                }
            )
            samples_collected += batch["input_ids"].size(0)
            if samples_collected >= num_samples:
                break

        # Pre-compute exact NF4 quantiles (standardized)
        self.nf4_quantiles = torch.tensor(
            [
                -1.0,
                -0.6961928,
                -0.5250731,
                -0.3949175,
                -0.2870153,
                -0.1894659,
                -0.0910500,
                0.0,
                0.0795803,
                0.1609302,
                0.2461123,
                0.3379152,
                0.4407098,
                0.5626170,
                0.7229568,
                1.0,
            ],
            device=self.device,
        )

    def _quantize_dequantize_nf4(self, weight: torch.Tensor) -> torch.Tensor:
        absmax = weight.abs().max()
        if absmax == 0:
            return weight
        normalized_weight = weight / absmax
        distances = torch.abs(normalized_weight.unsqueeze(-1) - self.nf4_quantiles)
        nearest_indices = torch.argmin(distances, dim=-1)
        quantized_normalized = self.nf4_quantiles[nearest_indices]
        return quantized_normalized * absmax

    def compute_baseline_logits(self) -> list:
        baseline_logits = []
        with torch.no_grad():
            for batch in self.calibration_batches:
                logits = self.model(**batch).logits
                baseline_logits.append(logits)
        return baseline_logits

    def run_sweep(self) -> Dict[str, float]:
        print("[+] Computing baseline logits on calibration set...")
        baseline_logits = self.compute_baseline_logits()

        sensitivity_scores = {}
        print("[+] Running layer-wise NF4 perturbation sweep...")

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and "classifier" not in name:
                original_weight = module.weight.data.clone()
                module.weight.data = self._quantize_dequantize_nf4(original_weight)

                mse_sum = 0.0
                total_elements = 0
                with torch.no_grad():
                    for batch_idx, batch in enumerate(self.calibration_batches):
                        perturbed_logits = self.model(**batch).logits
                        mse_sum += torch.nn.functional.mse_loss(
                            perturbed_logits,
                            baseline_logits[batch_idx],
                            reduction="sum",
                        ).item()
                        total_elements += perturbed_logits.numel()

                module.weight.data = original_weight
                layer_mse = mse_sum / total_elements
                sensitivity_scores[name] = layer_mse
                print(f"    Layer: {name:<45} | Logit MSE: {layer_mse:.6f}")

        return sensitivity_scores


if __name__ == "__main__":
    profiler = NF4SensitivityProfiler(
        model_name="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )
    scores = profiler.run_sweep()

    # Calculate FFN vs MHSA distortion proportion
    ffn_total = sum(v for k, v in scores.items() if "ffn" in k)
    mhsa_total = sum(
        v
        for k, v in scores.items()
        if any(k_sub in k for k_sub in ["q_lin", "k_lin", "v_lin", "out_lin"])
    )
    grand_total = ffn_total + mhsa_total

    print("\n" + "=" * 50)
    print("       AGGREGATE SENSITIVITY RESULTS SUMMARY       ")
    print("=" * 50)
    if grand_total > 0:
        print(f"FFN Distortion Share:  {(ffn_total / grand_total) * 100:.2f}%")
        print(f"MHSA Distortion Share: {(mhsa_total / grand_total) * 100:.2f}%")
    print("=" * 50)
