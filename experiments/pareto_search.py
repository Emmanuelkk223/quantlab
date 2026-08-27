"""
quantlab/experiments/pareto_search.py
Definitive Publication Version: 100-Trial NSGA-II Search (Seed: 42, 36 Unique Variables, P50 Latency)
"""

import os
import torch
import optuna
import numpy as np
import pandas as pd
import bitsandbytes as bnb
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig
from quantlab.datasets.glue_loader import GLUEDataLoader

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def verify_and_get_precision_manifest(model, skip_modules: list):
    """Programmatically inspects instantiated model to guarantee exact precision mapping."""
    manifest = {}
    fp16_count = 0
    nf4_count = 0
    for name, module in model.named_modules():
        if not isinstance(module, (torch.nn.Linear, bnb.nn.Linear4bit)):
            continue

        # Exclude classification head/pre-classifier layers to match the 36 transformer target dimensions
        if "classifier" in name:
            continue

        if isinstance(module, bnb.nn.Linear4bit):
            nf4_count += 1
            manifest[name] = "NF4"
            assert (
                name not in skip_modules
            ), f"Invariant Failure: {name} is 4-bit but listed as FP16!"
            assert (
                module.weight.dtype == torch.uint8
            ), f"Invariant Failure: {name} is not uint8!"
        elif isinstance(module, torch.nn.Linear):
            fp16_count += 1
            manifest[name] = "FP16"
            assert (
                name in skip_modules
            ), f"Invariant Failure: {name} is FP16 but not in skip_modules!"
            assert module.weight.dtype in [
                torch.float16,
                torch.bfloat16,
                torch.float32,
            ], f"Invariant Failure: {name} dtype invalid!"

    assert len(manifest) == 36, f"Expected 36 inspected layers, found {len(manifest)}"
    return fp16_count, nf4_count, manifest


class NSGAIIObjective:
    def __init__(self, model_name: str, device: str = "cuda:0"):
        self.model_name = model_name
        self.device = torch.device(device)

        # Explicit 1024-sample deterministic Search Set
        dataloader_builder = GLUEDataLoader(
            model_name_or_path=model_name, batch_size=16
        )
        _, self.search_loader, _ = dataloader_builder.get_splits(
            num_calibration=256, num_search=1024
        )

        temp_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        # Extract full unambiguous layer names to guarantee 36 unique variables
        self.target_layers = [
            name
            for name, module in temp_model.named_modules()
            if isinstance(module, torch.nn.Linear) and "classifier" not in name
        ]
        del temp_model
        assert (
            len(self.target_layers) == 36
        ), f"Expected 36 target layers, found {len(self.target_layers)}"

        self.start_evt = torch.cuda.Event(enable_timing=True)
        self.end_evt = torch.cuda.Event(enable_timing=True)

    def __call__(self, trial: optuna.Trial):
        skip_modules = []
        precision_vector = []

        # Create 36 strictly unique Optuna variables (resolving name collisions)
        for layer_name in self.target_layers:
            # Safe unique variable name (e.g., keep_fp16_distilbert_transformer_layer_0_attention_q_lin)
            safe_var_name = f"keep_fp16_{layer_name.replace('.', '_')}"
            keep_fp16 = trial.suggest_categorical(safe_var_name, [True, False])
            if keep_fp16:
                skip_modules.append(layer_name)
                precision_vector.append("FP16")
            else:
                precision_vector.append("NF4")

        assert (
            len(skip_modules) + precision_vector.count("NF4") == 36
        ), "Dimension mismatch in precision vector"
        trial.set_user_attr("precision_vector", str(precision_vector))
        trial.set_user_attr("fp16_layer_count", len(skip_modules))

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
            model.eval()

            # Programmatic verification of all 36 distinct states
            verify_and_get_precision_manifest(model, skip_modules)

            dummy_batch = next(iter(self.search_loader))
            dummy_inputs = {
                k: v.to(self.device)
                for k, v in dummy_batch.items()
                if k in ["input_ids", "attention_mask"]
            }

            # 30 Warm-up iterations (Matching paper protocol)
            with torch.no_grad():
                for _ in range(30):
                    _ = model(**dummy_inputs)
            torch.cuda.synchronize()

            correct = 0
            total = 0
            latencies = []

            # 100 Active timing samples (Matching paper protocol)
            with torch.no_grad():
                for batch in self.search_loader:
                    inputs = {
                        k: v.to(self.device)
                        for k, v in batch.items()
                        if k in ["input_ids", "attention_mask"]
                    }
                    labels = batch["label"].to(self.device)

                    self.start_evt.record()
                    outputs = model(**inputs)
                    self.end_evt.record()
                    torch.cuda.synchronize()

                    latencies.append(self.start_evt.elapsed_time(self.end_evt))
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)

            accuracy = correct / total
            # Strict Median (P50) Latency Metric
            median_latency = float(np.median(latencies))

        except Exception as err:
            print(f"[TRIAL FAILED] {err}")
            return 0.0, 999.0
        finally:
            if "model" in locals():
                del model
            torch.cuda.empty_cache()

        return accuracy, median_latency


if __name__ == "__main__":
    print("[+] Initializing Definitive 36-Dimensional 100-Trial NSGA-II Search...")
    sampler = optuna.samplers.NSGAIISampler(seed=SEED)
    study = optuna.create_study(
        study_name="quantlab_nsgaii_36dim_100",
        directions=["maximize", "minimize"],
        sampler=sampler,
    )

    objective = NSGAIIObjective(
        "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )
    study.optimize(objective, n_trials=100)

    results = []
    for i, t in enumerate(study.best_trials):
        results.append(
            {
                "candidate_id": i + 1,
                "search_accuracy": t.values[0],
                "median_latency_ms": t.values[1],
                "fp16_count": t.user_attrs.get("fp16_layer_count", 0),
                "precision_vector": t.user_attrs.get("precision_vector", "[]"),
            }
        )

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(results).to_csv("results/nsgaii_pareto_front_36dim.csv", index=False)
    print(
        "[+] Search complete. True 36-dimensional non-dominated configurations saved to results/nsgaii_pareto_front_36dim.csv"
    )
