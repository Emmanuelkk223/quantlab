"""
quantlab/datasets/glue_loader.py
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Subset


class GLUEDataLoader:
    def __init__(
        self, model_name_or_path: str, task: str = "sst2", batch_size: int = 32
    ):
        self.task = task
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # Load the full dataset
        self.raw_datasets = load_dataset("nyu-mll/glue", task)

    def _tokenize_function(self, examples):
        return self.tokenizer(
            examples["sentence"], padding="max_length", truncation=True, max_length=128
        )

    def get_splits(self, num_calibration: int = 256, num_search: int = 1024):
        """
        Generates strictly isolated datasets to prevent selection bias.
        """
        tokenized_datasets = self.raw_datasets.map(
            self._tokenize_function, batched=True
        )
        tokenized_datasets.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )

        train_data = tokenized_datasets["train"]

        # Ensure we don't overlap calibration and search data
        assert num_calibration + num_search <= len(
            train_data
        ), "Requested more samples than available in train split."

        # 1. Calibration Data (For Logit MSE Sweep)
        calib_subset = Subset(train_data, range(0, num_calibration))
        calib_loader = DataLoader(
            calib_subset, batch_size=self.batch_size, shuffle=False
        )

        # 2. Search Data (For Optuna NSGA-II evaluations)
        search_subset = Subset(
            train_data, range(num_calibration, num_calibration + num_search)
        )
        search_loader = DataLoader(
            search_subset, batch_size=self.batch_size, shuffle=False
        )

        # 3. Holdout Test Data (For final reporting only)
        test_data = tokenized_datasets["validation"]
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

        return calib_loader, search_loader, test_loader
