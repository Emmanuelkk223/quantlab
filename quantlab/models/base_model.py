"""
quantlab/models/base_model.py

Base model wrapper for Hugging Face transformer architectures, providing
parameter introspection, memory footprint analysis, input generation,
and bitsandbytes quantization dispatch.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, List, Optional
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    BitsAndBytesConfig,
)


class BaseModelWrapper(nn.Module):
    """
    Unified wrapper around Hugging Face transformer models for quantization profiling.
    """

    def __init__(
        self,
        model_name_or_path: str = "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        num_labels: int = 2,
        dtype: torch.dtype = torch.float32,
        quantization_config: Optional[BitsAndBytesConfig] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.dtype = dtype
        self.device = device or (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )

        kwargs: Dict[str, Any] = {
            "num_labels": num_labels,
            "dtype": dtype,
        }

        if quantization_config is not None:
            kwargs["quantization_config"] = quantization_config
            kwargs["device_map"] = {
                "": self.device.index if self.device.type == "cuda" else "cpu"
            }
            self.model: PreTrainedModel = (
                AutoModelForSequenceClassification.from_pretrained(
                    model_name_or_path, **kwargs
                )
            )
        else:
            self.model: PreTrainedModel = (
                AutoModelForSequenceClassification.from_pretrained(
                    model_name_or_path, **kwargs
                ).to(self.device)
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def forward(self, *args, **kwargs) -> Any:
        return self.model(*args, **kwargs)

    def generate_dummy_inputs(
        self, batch_size: int = 16, seq_len: int = 128
    ) -> Dict[str, torch.Tensor]:
        dummy_text = ["QuantLab baseline inference benchmarking step."] * batch_size
        inputs = self.tokenizer(
            dummy_text,
            padding="max_length",
            max_length=seq_len,
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def get_parameter_count(self) -> Dict[str, int]:
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "non_trainable_params": total_params - trainable_params,
        }

    def get_model_size_mb(self) -> Dict[str, float]:
        total_bytes = sum(
            p.numel() * p.element_size() for p in self.model.parameters()
        ) + sum(b.numel() * b.element_size() for b in self.model.buffers())
        bytes_in_mb = 1024.0 * 1024.0
        return {"model_size_mb": total_bytes / bytes_in_mb}

    def get_linear_layers(self) -> List[Tuple[str, nn.Linear]]:
        linear_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers.append((name, module))
        return linear_layers
