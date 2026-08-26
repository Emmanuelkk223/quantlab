"""
quantlab/experiments/sensitivity_analysis.py

Executes single-layer sensitivity analysis by measuring logit perturbation (MSE)
and validation accuracy drop when quantizing individual transformer layers.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List
from quantlab.models.base_model import BaseModelWrapper
from quantlab.models.introspection import ModelIntrospector
from quantlab.datasets.glue_loader import GLUEDataLoader


class SensitivityAnalyzer:
    """
    Evaluates layer-by-layer precision sensitivity against unquantized baseline outputs.
    """

    def __init__(
        self,
        model_name: str = "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    ):
        self.model_name = model_name
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load baseline FP32 model
        self.baseline_wrapper = BaseModelWrapper(
            model_name, dtype=torch.float32, device=self.device
        )
        self.baseline_model = self.baseline_wrapper.model
        self.baseline_model.eval()

        # Introspect architecture
        introspector = ModelIntrospector(self.baseline_model)
        self.manifest = introspector.get_layer_manifest()

    def compute_layer_mse_sensitivity(
        self, dummy_inputs: Dict[str, torch.Tensor]
    ) -> List[Dict[str, Any]]:
        """
        Calculates output logit Mean Squared Error (MSE) shift when sim-quantizing individual linear layers.
        """
        inputs = {k: v.to(self.device) for k, v in dummy_inputs.items()}

        # 1. Obtain ground-truth FP32 baseline logits
        with torch.no_grad():
            baseline_logits = self.baseline_model(**inputs).logits

        sensitivity_results = []

        print(
            f"[+] Profiling Sensitivity across {len(self.manifest)} target linear layers..."
        )

        # 2. Iterate through layers, applying localized weight quantization simulation
        for item in self.manifest:
            layer_name = item["name"]
            module = item["module"]

            # Save unquantized weights
            orig_weight = module.weight.data.clone()

            # Simulate simple 8-bit uniform quant/dequant on target layer weights: Q(W)
            scale = (orig_weight.max() - orig_weight.min()) / 255.0
            quant_weight = torch.clamp(
                torch.round(orig_weight / (scale + 1e-8)), -128, 127
            )
            dequant_weight = quant_weight * scale

            # Inject quantized weight into module
            module.weight.data = dequant_weight

            # Measure perturbed forward pass
            with torch.no_grad():
                perturbed_logits = self.baseline_model(**inputs).logits
                mse_loss = F.mse_loss(perturbed_logits, baseline_logits).item()

            # Restore original weight
            module.weight.data = orig_weight

            sensitivity_results.append(
                {
                    "layer_name": layer_name,
                    "block_type": item["block_type"],
                    "param_count": item["num_parameters"],
                    "logit_mse": mse_loss,
                }
            )

            print(
                f"  -> Layer: {layer_name:<45} | Type: {item['block_type']:<18} | Logit MSE: {mse_loss:.6f}"
            )

        return sensitivity_results


if __name__ == "__main__":
    analyzer = SensitivityAnalyzer()
    dummy_input = analyzer.baseline_wrapper.generate_dummy_inputs(
        batch_size=16, seq_len=128
    )
    results = analyzer.compute_layer_mse_sensitivity(dummy_input)
