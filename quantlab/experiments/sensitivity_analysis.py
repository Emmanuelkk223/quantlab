"""
quantlab/experiments/sensitivity_analysis.py
Definitive Publication Version: 36-Layer NF4 Perturbation Sweep (Seed: 42)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict
from quantlab.datasets.glue_loader import GLUEDataLoader
from transformers import AutoModelForSequenceClassification

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


class NF4SensitivityProfiler:
    def __init__(self, model_name: str, device: str = "cuda:0", num_samples: int = 256):
        self.device = torch.device(device)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            self.device
        )
        self.model.eval()

        # Deterministic 256-sample calibration split from SST-2 training data
        dataloader_builder = GLUEDataLoader(
            model_name_or_path=model_name, batch_size=16
        )
        calib_loader, _, _ = dataloader_builder.get_splits(
            num_calibration=num_samples, num_search=1024
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
        return self.nf4_quantiles[nearest_indices] * absmax

    def compute_baseline_logits(self) -> list:
        baseline_logits = []
        with torch.no_grad():
            for batch in self.calibration_batches:
                logits = self.model(**batch).logits
                baseline_logits.append(logits)
        return baseline_logits

    def run_sweep(self) -> Dict[str, float]:
        baseline_logits = self.compute_baseline_logits()
        sensitivity_scores = {}

        target_layers = [
            name
            for name, module in self.model.named_modules()
            if isinstance(module, nn.Linear) and "classifier" not in name
        ]

        assert (
            len(target_layers) == 36
        ), f"Expected exactly 36 projection layers, found {len(target_layers)}"

        for name in target_layers:
            module = dict(self.model.named_modules())[name]
            original_weight = module.weight.data.clone()
            module.weight.data = self._quantize_dequantize_nf4(original_weight)

            mse_sum = 0.0
            total_elements = 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(self.calibration_batches):
                    perturbed_logits = self.model(**batch).logits
                    mse_sum += torch.nn.functional.mse_loss(
                        perturbed_logits, baseline_logits[batch_idx], reduction="sum"
                    ).item()
                    total_elements += perturbed_logits.numel()

            module.weight.data = original_weight
            sensitivity_scores[name] = mse_sum / total_elements

        return sensitivity_scores


if __name__ == "__main__":
    profiler = NF4SensitivityProfiler(
        model_name="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )
    scores = profiler.run_sweep()

    ffn_scores = {k: v for k, v in scores.items() if "ffn" in k}
    mhsa_scores = {
        k: v
        for k, v in scores.items()
        if any(sub in k for sub in ["q_lin", "k_lin", "v_lin", "out_lin"])
    }

    sum_ffn = sum(ffn_scores.values())
    sum_mhsa = sum(mhsa_scores.values())
    sum_total = sum(scores.values())

    print(f"FFN Single-Layer LMSE Share (S_FFN): {(sum_ffn / sum_total) * 100:.2f}%")
    print(f"MHSA Single-Layer LMSE Share (S_MHSA): {(sum_mhsa / sum_total) * 100:.2f}%")
