"""
quantlab/visualization/pareto_plots.py
"""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Dict, Any


def plot_pareto_frontier(
    results_data: List[Dict[str, Any]],
    output_path: str = "results/pareto_accuracy_vs_latency.png",
    title: str = "QuantLab: SST-2 Accuracy vs Inference Latency (RTX 4060)",
) -> None:
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    df = pd.DataFrame(results_data)

    sns.scatterplot(
        data=df,
        x="latency_ms",
        y="accuracy",
        size="vram_mb",
        hue="name",
        sizes=(120, 450),
        ax=ax,
        palette="viridis",
        edgecolor="black",
        linewidth=1.2,
    )

    # Stagger label offsets to prevent collisions in dense clusters
    stagger_offsets = {
        "FP16 Half": (-10, 12),
        "Uniform INT4 (NF4)": (10, -18),
        "Protected Mixed-Precision": (0, 12),
        "FP32 Baseline": (-45, 10),
        "Uniform INT8": (0, 10),
    }

    for _, row in df.iterrows():
        name = row["name"]
        offset = stagger_offsets.get(name, (5, 5))

        ax.annotate(
            f"{name}\n({row['accuracy']:.1f}%, {row['latency_ms']:.1f}ms)",
            (row["latency_ms"], row["accuracy"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=8.5,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"),
        )

    ax.set_xlabel(
        "Mean Inference Latency (ms) [Lower is Better]", fontsize=11, labelpad=8
    )
    ax.set_ylabel(
        "SST-2 Validation Accuracy (%) [Higher is Better]", fontsize=11, labelpad=8
    )
    ax.set_title(title, fontsize=13, pad=12, weight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

    plt.tight_layout()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
