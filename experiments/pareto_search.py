"""
quantlab/experiments/pareto_search.py

Executes a reproducible, scaled NSGA-II multi-objective search.
Outputs exact layer-wise precision vectors and verifies hardware state.
"""

import os
import torch
import optuna
import numpy as np
import pandas as pd
import bitsandbytes as bnb
from transformers import BitsAndBytesConfig, AutoModelForSequenceClassification
from quantlab.datasets.glue_loader import GLUEDataLoader

# Fixed seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def verify_mixed_precision_state(model, skip_modules: list):
    """Proves the hardware successfully mapped the heterogeneous precision state."""
    fp16_count = 0
    nf4_count = 0
    for name, module in model.named_modules():
        if not isinstance(module, (torch.nn.Linear, bnb.nn.Linear4bit)):
            continue
        short_name = name.split(".")[-1]

        if isinstance(module, bnb.nn.Linear4bit):
            nf4_count += 1
            assert (
                short_name not in skip_modules
            ), f"FAILED: {name} is 4-bit but should be skipped!"
            assert (
                module.weight.dtype == torch.uint8
            ), f"FAILED: {name} weight is not uint8!"
        elif isinstance(module, torch.nn.Linear):
            fp16_count += 1
            assert (
                short_name in skip_modules
            ), f"FAILED: {name} is FP16 but wasn't in skip_modules!"
            assert module.weight.dtype in [
                torch.float16,
                torch.bfloat16,
                torch.float32,
            ], f"FAILED: {name} is not FP16/32!"

    return fp16_count, nf4_count


class NSGAIIObjective:
    def __init__(self, model_name: str, device: str = "cuda:0"):
        self.model_name = model_name
        self.device = torch.device(device)

        # 1. Strict Data Isolation: Use ONLY the 1024-sample Search Set
        dataloader_builder = GLUEDataLoader(
            model_name_or_path=model_name, batch_size=16
        )
        _, self.search_loader, _ = dataloader_builder.get_splits(
            num_calibration=256, num_search=1024
        )

        # 2. Extract the 38 linear layers dynamically
        temp_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.target_layers = [
            name.split(".")[-1]
            for name, module in temp_model.named_modules()
            if isinstance(module, torch.nn.Linear) and "classifier" not in name
        ]
        del temp_model

        self.start_evt = torch.cuda.Event(enable_timing=True)
        self.end_evt = torch.cuda.Event(enable_timing=True)

    def __call__(self, trial: optuna.Trial):
        skip_modules = []
        precision_vector = []

        # NSGA-II binary decisions for all 38 layers
        for layer_name in self.target_layers:
            keep_fp16 = trial.suggest_categorical(
                f"keep_fp16_{layer_name}", [True, False]
            )
            if keep_fp16:
                skip_modules.append(layer_name)
                precision_vector.append("FP16")
            else:
                precision_vector.append("NF4")

        # Save exact vector to trial user attributes for later extraction
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

            # Explicit Hardware Verification
            verify_mixed_precision_state(model, skip_modules)

            # Warmup
            dummy_batch = next(iter(self.search_loader))
            dummy_inputs = {
                k: v.to(self.device)
                for k, v in dummy_batch.items()
                if k in ["input_ids", "attention_mask"]
            }
            with torch.no_grad():
                for _ in range(10):
                    _ = model(**dummy_inputs)
            torch.cuda.synchronize()

            # Measure Accuracy and Median Latency
            correct = 0
            total = 0
            latencies = []

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
            median_latency = float(np.median(latencies))

        except AssertionError as err:
            print(f"[VERIFICATION FAILED] {err}")
            return 0.0, 999.0
        except Exception as err:
            print(f"[TRIAL FAILED] {err}")
            return 0.0, 999.0
        finally:
            if "model" in locals():
                del model
            torch.cuda.empty_cache()

        return accuracy, median_latency


if __name__ == "__main__":
    print("[+] Initializing Scaled NSGA-II Pareto Search...")

    # Setup reproducible NSGA-II sampler
    sampler = optuna.samplers.NSGAIISampler(seed=SEED)
    study = optuna.create_study(
        study_name="quantlab_nsgaii_100",
        directions=["maximize", "minimize"],
        sampler=sampler,
    )

    objective = NSGAIIObjective(
        "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )

    # Budget: 100 trials
    print(
        "[+] Running 100 trials on the Search Split (1024 samples). This will take a few minutes..."
    )
    study.optimize(objective, n_trials=100)

    # Extract Non-Dominated (Pareto) Configurations
    best_trials = study.best_trials

    results = []
    print("\n" + "=" * 80)
    print("       OBSERVED NON-DOMINATED CONFIGURATIONS (PARETO FRONT)       ")
    print("=" * 80)

    for i, t in enumerate(best_trials):
        acc = t.values[0]
        lat = t.values[1]
        fp16_count = t.user_attrs.get("fp16_layer_count", 0)
        vector = t.user_attrs.get("precision_vector", "[]")

        print(
            f"Candidate #{i+1}: Acc = {acc*100:.2f}% | Latency = {lat:.2f}ms | FP16 = {fp16_count}/38"
        )

        results.append(
            {
                "candidate_id": i + 1,
                "accuracy": acc,
                "latency_ms": lat,
                "fp16_count": fp16_count,
                "precision_vector": vector,
            }
        )

    # Save exact vectors to disk for the paper
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv("results/nsgaii_pareto_front.csv", index=False)
    print("=" * 80)
    print(
        "[+] Exact 38-dimensional layer vectors saved to 'results/nsgaii_pareto_front.csv'"
    )
