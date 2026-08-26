"""
quantlab/experiments/pareto_search.py

Automated Multi-Objective Pareto Search engine using Optuna (NSGA-II)
to discover optimal layer precision schedules under hardware constraints.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import optuna
from typing import List, Dict, Any, Tuple
from transformers import BitsAndBytesConfig, AutoModelForSequenceClassification

from quantlab.models.base_model import BaseModelWrapper
from quantlab.models.introspection import ModelIntrospector
from quantlab.datasets.glue_loader import GLUEDataLoader
from quantlab.evaluation.metrics import TaskEvaluator
from quantlab.hardware.latency_profiler import LatencyProfiler


class HardwareAwareParetoSearch:
    """
    Executes hardware-in-the-loop multi-objective optimization to find
    Pareto-optimal layer precision configurations.
    """

    def __init__(
        self,
        model_name: str = "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        num_calibration_samples: int = 256,
    ):
        self.model_name = model_name
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Introspect target linear layers
        base_wrapper = BaseModelWrapper(
            model_name, dtype=torch.float32, device=self.device
        )
        introspector = ModelIntrospector(base_wrapper.model)
        self.layer_manifest = introspector.get_layer_manifest()
        self.layer_names = [item["name"] for item in self.layer_manifest]

        # Extract short module names for bitsandbytes skip matching
        self.short_name_map = {}
        for full_name in self.layer_names:
            parts = full_name.split(".")
            self.short_name_map[full_name] = (
                ".".join(parts[-3:]) if len(parts) > 2 else full_name
            )

        # Prepare fast evaluation dataloader split for search speed
        data_loader_builder = GLUEDataLoader(
            model_name_or_path=model_name, batch_size=32
        )
        full_loader = data_loader_builder.get_dataloader()

        # Subsample evaluation batches for search acceleration
        subsampled_batches = []
        sample_count = 0
        for batch in full_loader:
            subsampled_batches.append(batch)
            sample_count += batch["input_ids"].size(0)
            if sample_count >= num_calibration_samples:
                break
        self.fast_loader = subsampled_batches

        self.evaluator = TaskEvaluator(device=self.device)
        self.profiler = LatencyProfiler(device=self.device)

        # Cleanup base initialization memory
        del base_wrapper
        torch.cuda.empty_cache()

    def _objective(self, trial: optuna.Trial) -> Tuple[float, float]:
        """
        Optuna trial objective function.
        Returns:
            Tuple of (Validation Accuracy %, Mean Latency ms)
        """
        skip_modules = []
        for full_name in self.layer_names:
            short_name = self.short_name_map[full_name]
            keep_fp16 = trial.suggest_categorical(
                f"keep_fp16_{short_name}", [True, False]
            )
            if keep_fp16:
                skip_modules.append(short_name)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=skip_modules if skip_modules else None,
        )

        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map={"": self.device.index or 0},
            )

            # Fast Accuracy Evaluation
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for batch in self.fast_loader:
                    inputs = {
                        "input_ids": batch["input_ids"].to(self.device),
                        "attention_mask": batch["attention_mask"].to(self.device),
                    }
                    labels = batch["label"].to(self.device)
                    logits = model(**inputs).logits
                    preds = torch.argmax(logits, dim=-1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            accuracy = (correct / total) * 100.0

            # Hardware Profiling
            dummy_inputs = {
                "input_ids": torch.randint(0, 1000, (16, 128), device=self.device),
                "attention_mask": torch.ones((16, 128), device=self.device),
            }
            hw_stats = self.profiler.profile_module(
                model=model,
                sample_inputs=dummy_inputs,
                warmup_steps=10,
                active_steps=30,
            )
            latency_ms = hw_stats["mean_latency_ms"]

        except Exception as err:
            print(f"[WARNING] Trial failed with error: {err}")
            return 0.0, 999.0

        finally:
            if "model" in locals():
                del model
            torch.cuda.empty_cache()

        return accuracy, latency_ms

    def run_search(self, n_trials: int = 20) -> List[optuna.trial.FrozenTrial]:
        """
        Executes multi-objective NSGA-II search over layer precision configurations.
        """
        print(f"[+] Starting Hardware-Aware Pareto Search ({n_trials} Trials)...")
        sampler = optuna.samplers.NSGAIISampler()
        study = optuna.create_study(
            directions=["maximize", "minimize"],
            sampler=sampler,
        )

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(self._objective, n_trials=n_trials, show_progress_bar=True)

        print("\n" + "=" * 70)
        print("               PARETO OPTIMAL CONFIGURATIONS FOUND             ")
        print("=" * 70)
        best_trials = study.best_trials
        for idx, trial in enumerate(best_trials):
            acc, lat = trial.values
            fp16_count = sum(1 for v in trial.params.values() if v is True)
            print(
                f"Pareto Candidate #{idx+1:<2} | Acc: {acc:.2f}% | Latency: {lat:.3f} ms | FP16 Layers: {fp16_count}/38"
            )

        return best_trials


if __name__ == "__main__":
    search_engine = HardwareAwareParetoSearch(num_calibration_samples=256)
    pareto_candidates = search_engine.run_search(n_trials=15)
