"""
quantlab/evaluation/metrics.py

Task performance evaluation engine for evaluating accuracy and cross-entropy loss.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
from tqdm import tqdm


class TaskEvaluator:
    """
    Evaluates classification models over validation datasets to track accuracy loss degradation.
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )

    def evaluate_accuracy(
        self,
        model: nn.Module,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """
        Runs evaluation pass over dataloader and calculates top-1 accuracy and average loss.
        """
        model.to(self.device)
        model.eval()

        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                loss = criterion(logits, labels)
                total_loss += loss.item() * input_ids.size(0)

                preds = torch.argmax(logits, dim=-1)
                total_correct += (preds == labels).sum().item()
                total_samples += input_ids.size(0)

        return {
            "accuracy": float(total_correct / total_samples),
            "eval_loss": float(total_loss / total_samples),
            "total_samples": total_samples,
        }
