"""
quantlab/scripts/evaluate_accuracy.py

Evaluates the discovered Pareto candidates and Uniform baselines
on the strictly isolated Holdout Test Set.
"""

import ast
import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig
from quantlab.datasets.glue_loader import GLUEDataLoader


def get_target_layers(model_name):
    temp = AutoModelForSequenceClassification.from_pretrained(model_name)
    layers = [
        name.split(".")[-1]
        for name, module in temp.named_modules()
        if isinstance(module, torch.nn.Linear) and "classifier" not in name
    ]
    del temp
    return layers


def evaluate_model(model, test_loader, device="cuda:0"):
    model.eval()
    correct = 0
    total = 0

    # 1. Warm-up and Latency Profile (100 iterations)
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    latencies = []

    dummy_batch = next(iter(test_loader))
    dummy_inputs = {
        k: v.to(device)
        for k, v in dummy_batch.items()
        if k in ["input_ids", "attention_mask"]
    }

    with torch.no_grad():
        for _ in range(30):  # Warmup
            _ = model(**dummy_inputs)
        torch.cuda.synchronize()

        for _ in range(100):  # Latency measure
            start_evt.record()
            _ = model(**dummy_inputs)
            end_evt.record()
            torch.cuda.synchronize()
            latencies.append(start_evt.elapsed_time(end_evt))

    median_lat = float(np.median(latencies))
    std_lat = float(np.std(latencies))

    # 2. Accuracy Evaluation
    with torch.no_grad():
        for batch in test_loader:
            inputs = {
                k: v.to(device)
                for k, v in batch.items()
                if k in ["input_ids", "attention_mask"]
            }
            labels = batch["label"].to(device)
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return (correct / total) * 100, median_lat, std_lat


if __name__ == "__main__":
    MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    DEVICE = "cuda:0"

    print("[+] Loading Holdout Test Set (872 samples)...")
    loader_builder = GLUEDataLoader(MODEL_NAME, batch_size=16)
    _, _, test_loader = loader_builder.get_splits(num_calibration=256, num_search=1024)
    target_layers = get_target_layers(MODEL_NAME)

    results_log = []

    # --- 1. Evaluate Uniform FP32 Baseline ---
    print("\n[+] Evaluating Uniform FP32 Baseline...")
    model_fp32 = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(
        DEVICE
    )
    acc, lat, std = evaluate_model(model_fp32, test_loader, DEVICE)
    results_log.append(
        {"Config": "FP32 Baseline", "Acc": acc, "Lat": lat, "Std": std, "FP16": "38/38"}
    )
    del model_fp32
    torch.cuda.empty_cache()

    # --- 2. Evaluate Uniform NF4 Baseline ---
    print("[+] Evaluating Uniform NF4 Baseline...")
    bnb_nf4 = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model_nf4 = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, quantization_config=bnb_nf4, device_map={"": 0}
    )
    acc, lat, std = evaluate_model(model_nf4, test_loader, DEVICE)
    results_log.append(
        {"Config": "Uniform NF4", "Acc": acc, "Lat": lat, "Std": std, "FP16": "0/38"}
    )
    del model_nf4
    torch.cuda.empty_cache()

    # --- 3. Evaluate Pareto Candidates ---
    print("[+] Loading Pareto Candidates from Search Phase...")
    try:
        df_pareto = pd.read_csv("results/nsgaii_pareto_front.csv")
        for _, row in df_pareto.iterrows():
            cand_id = row["candidate_id"]
            vec_str = row["precision_vector"]
            vec = ast.literal_eval(vec_str)

            skip_modules = [
                target_layers[i] for i, prec in enumerate(vec) if prec == "FP16"
            ]

            print(f"[+] Evaluating Pareto Candidate #{cand_id}...")
            bnb_mixed = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                llm_int8_skip_modules=skip_modules,
            )
            model_mixed = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME, quantization_config=bnb_mixed, device_map={"": 0}
            )

            acc, lat, std = evaluate_model(model_mixed, test_loader, DEVICE)
            results_log.append(
                {
                    "Config": f"Pareto Cand #{cand_id}",
                    "Acc": acc,
                    "Lat": lat,
                    "Std": std,
                    "FP16": f"{len(skip_modules)}/38",
                }
            )
            del model_mixed
            torch.cuda.empty_cache()

    except FileNotFoundError:
        print(
            "[-] results/nsgaii_pareto_front.csv not found. Did you run pareto_search.py?"
        )

    # --- FINAL REPORT ---
    print("\n" + "=" * 80)
    print("                 FINAL HOLDOUT GENERALIZATION RESULTS                 ")
    print("=" * 80)
    print(
        f"{'Configuration':<20} | {'Test Acc (%)':<15} | {'Median Lat (ms)':<17} | {'FP16 Layers':<15}"
    )
    print("-" * 80)
    for r in results_log:
        print(
            f"{r['Config']:<20} | {r['Acc']:<15.2f} | {r['Lat']:>5.2f} (±{r['Std']:>4.2f}) | {r['FP16']:<15}"
        )
    print("=" * 80)
