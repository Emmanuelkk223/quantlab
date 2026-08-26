"""
quantlab/models/introspection.py

Utilities for parsing Transformer model architectures, classifying layer types
(Attention vs FFN vs Heads), and enabling layer-by-layer precision injection.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any


class ModelIntrospector:
    """
    Parses PyTorch transformer modules and groups linear layers by architectural block.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def get_layer_manifest(self) -> List[Dict[str, Any]]:
        """
        Categorizes linear layers into Attention projections, FFN blocks, and heads.
        """
        manifest = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                block_type = "other"
                if any(
                    k in name
                    for k in [
                        "q_lin",
                        "k_lin",
                        "v_lin",
                        "out_lin",
                        "query",
                        "key",
                        "value",
                        "attention",
                    ]
                ):
                    block_type = "attention"
                elif any(k in name for k in ["lin1", "lin2", "dense", "mlp", "ffn"]):
                    block_type = "ffn"
                elif any(k in name for k in ["classifier", "pre_classifier", "head"]):
                    block_type = "classification_head"

                manifest.append(
                    {
                        "name": name,
                        "module": module,
                        "block_type": block_type,
                        "num_parameters": sum(p.numel() for p in module.parameters()),
                        "in_features": module.in_features,
                        "out_features": module.out_features,
                    }
                )
        return manifest
