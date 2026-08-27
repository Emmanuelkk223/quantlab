"""
quantlab/quantization/precision_map.py

Generates layer-specific numerical precision assignments based on
empirical sensitivity profile metrics.
"""

from typing import Dict, List, Any, Set


class PrecisionAllocator:
    """
    Allocates per-layer precisions using layer sensitivity rankings.
    """

    def __init__(self, sensitivity_profile: List[Dict[str, Any]]):
        # Sort layers by logit MSE in descending order
        self.profile = sorted(
            sensitivity_profile, key=lambda x: x["logit_mse"], reverse=True
        )

    def generate_protected_allocation(self, top_k_sensitive: int = 4) -> Dict[str, Any]:
        """
        Creates a mixed-precision policy protecting the top-K most sensitive layers
        in FP16/INT8 while keeping remaining layers in INT4.

        Args:
            top_k_sensitive: Number of high-sensitivity layers to protect from low-bit quant.

        Returns:
            Dictionary containing protected layer names and full per-layer precision map.
        """
        protected_layers: Set[str] = set()
        precision_map: Dict[str, str] = {}

        for idx, item in enumerate(self.profile):
            layer_name = item["layer_name"]
            if idx < top_k_sensitive:
                protected_layers.add(layer_name)
                precision_map[layer_name] = "fp16"
            else:
                precision_map[layer_name] = "int4"

        return {
            "top_k_protected_count": top_k_sensitive,
            "protected_layer_names": list(protected_layers),
            "precision_map": precision_map,
        }
