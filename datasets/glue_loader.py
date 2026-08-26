"""
quantlab/datasets/glue_loader.py

Data loading utilities for evaluating quantized models on GLUE benchmark tasks.
"""

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Dict, Any, Optional


class GLUEDataLoader:
    """
    Handles fetching and tokenizing evaluation datasets (default: SST-2).
    """

    def __init__(
        self,
        model_name_or_path: str,
        task_name: str = "sst2",
        batch_size: int = 32,
        max_length: int = 128,
    ):
        self.task_name = task_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # Fetch validation split using explicit dataset repository
        if task_name == "sst2":
            self.raw_dataset = load_dataset("stanfordnlp/sst2", split="validation")
        else:
            self.raw_dataset = load_dataset("glue", task_name, split="validation")

        self.tokenized_dataset = self.raw_dataset.map(
            self._tokenize_fn,
            batched=True,
            remove_columns=[
                col for col in self.raw_dataset.column_names if col not in ["label"]
            ],
        )

    def _tokenize_fn(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        return self.tokenizer(
            examples["sentence"],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

    def get_dataloader(self) -> DataLoader:
        """Returns PyTorch DataLoader formatted for evaluation loops."""
        dataset = self.tokenized_dataset.with_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
