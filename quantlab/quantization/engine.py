"""
quantlab/quantization/engine.py

Quantization factory engine for loading, applying, and inspecting
uniform precision modes (FP32, FP16, INT8, INT4) on Transformer models.
"""

import torch
from typing import Optional
from transformers import BitsAndBytesConfig
from quantlab.models.base_model import BaseModelWrapper


class QuantizationEngine:
    """
    Factory for instantiating models under distinct uniform numerical precisions:
    - FP32 (Full Precision)
    - FP16 (Half Precision)
    - INT8 (Vector Quantized 8-bit integer via bitsandbytes)
    - INT4 (NormalFloat4 4-bit quantization via bitsandbytes)
    """

    SUPPORTED_PRECISIONS = ["fp32", "fp16", "int8", "int4"]
    DEFAULT_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

    @staticmethod
    def load_quantized_model(
        model_name_or_path: str = DEFAULT_MODEL,
        precision: str = "fp32",
        num_labels: int = 2,
        device: Optional[torch.device] = None,
    ) -> BaseModelWrapper:
        precision = precision.lower()
        if precision not in QuantizationEngine.SUPPORTED_PRECISIONS:
            raise ValueError(
                f"Unsupported precision '{precision}'. Must be one of {QuantizationEngine.SUPPORTED_PRECISIONS}"
            )

        device = device or (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )

        print(
            f"[QuantizationEngine] Loading '{model_name_or_path}' with Precision: {precision.upper()}"
        )

        if precision == "fp32":
            return BaseModelWrapper(
                model_name_or_path,
                num_labels=num_labels,
                dtype=torch.float32,
                device=device,
            )

        elif precision == "fp16":
            return BaseModelWrapper(
                model_name_or_path,
                num_labels=num_labels,
                dtype=torch.float16,
                device=device,
            )

        elif precision == "int8":
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
            return BaseModelWrapper(
                model_name_or_path,
                num_labels=num_labels,
                dtype=torch.float16,
                quantization_config=bnb_config,
                device=device,
            )

        elif precision == "int4":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            return BaseModelWrapper(
                model_name_or_path,
                num_labels=num_labels,
                dtype=torch.float16,
                quantization_config=bnb_config,
                device=device,
            )

        raise RuntimeError("Reached unreachable precision branch.")
