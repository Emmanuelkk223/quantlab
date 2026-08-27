# QuantLab
### Hardware-Aware Mixed-Precision Quantization and Pareto Optimization for Transformer Encoder Inference

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License:MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**QuantLab** is an open-source, research-grade framework designed to systematically profile, evaluate, and optimize layer-wise numerical precisions for Transformer encoder models under real hardware constraints. 

Traditional Post-Training Quantization (PTQ) enforces uniform bit-widths across all layers, leading to task accuracy degradation or poor wall-clock speedups due to silicon execution bottlenecks. QuantLab bridges the gap between theoretical quantization mathematics and physical GPU execution by combining high-resolution CUDA stream event timing ($L_{\text{P50}}$), simulated 4-bit NormalFloat (NF4) logit perturbation sensitivity analysis, and unconstrained Multi-Objective NSGA-II Pareto optimization over a true 36-dimensional layer precision space.

---

## 📂 Repository Structure

```text
quantlab/
├── calibration/          # Calibration routines and data partitioning
├── configs/              # YAML configuration files for experiments
├── datasets/             # Isolated GLUE benchmark data loaders (SST-2 four-tier splits)
├── evaluation/           # Task metrics, loss calculations, and evaluation utilities
├── experiments/          # 36-dimensional sensitivity analysis & NSGA-II search drivers
├── hardware/             # High-resolution CUDA event latency and memory profilers
├── models/               # BaseModelWrapper & transformer layer introspection parser
├── quantization/         # Uniform/mixed precision engines and precision mappers
├── quantlab/             # Core package visual assets and artifact outputs
├── results/              # CSV search logs, Pareto data exports, and manifests
├── scripts/              # Executable benchmarks, accuracy evaluation, and reporting scripts
├── tests/                # Unit tests for hardware profilers and quantization invariants
├── visualization/        # Pareto frontier, latency, and heatmap plotting tools
├── pyproject.toml        # Project metadata and build configuration
├── requirements.txt      # Python dependencies specification
└── README.md             # Project documentation

```

---

## 🛠️ Key Framework Features

1. **High-Resolution CUDA Profiling:** Uses non-blocking CUDA stream hardware events (`torch.cuda.Event`) with explicit warm-up iterations ($N_{\text{warmup}} = 30$) and 100 active samples to record stable median wall-clock latency ($L_{\text{P50}}$), avoiding thermal throttling and cold-start bias.
2. **True 36-Dimensional Optimization Space:** Uniquely parameterizes all 36 linear projection layers of DistilBERT (6 transformer blocks $\times$ 6 projections) into independent binary optimization variables, resolving module-name collision bugs.
3. **Simulated NF4 Perturbation Sensitivity:** Quantifies structural layer vulnerability via isolated 4-bit NormalFloat quantization, establishing that FFN projections account for **99.34%** of aggregate single-layer logit $\mathcal{L}_{\text{MSE}}$.
4. **Strict Four-Tier Data Protocol:** Enforces total isolation between the 256-sample calibration set, 1,024-sample search set, and 872-sample locked holdout validation set (official SST-2 validation split) to prevent data leakage and selection bias.

---

## ⚙️ Installation

1. Clone the repository and check out the publication tag:
```bash
git clone https://github.com/Emmanuelkk223/quantlab.git
cd quantlab
git checkout v1.0-paper

```


2. Create and activate a Python virtual environment:
```bash
conda create -n quantlab python=3.10 -y
conda activate quantlab

```


3. Install dependencies:
```bash
pip install -r requirements.txt

```



---

## 🚀 Reproducibility & Usage

### 1. Run Simulated NF4 Sensitivity Analysis

Execute the deterministic 256-sample calibration sweep (Seed: 42) to calculate single-layer logit Mean Squared Error ($\mathcal{L}_{\text{MSE}}$):

```bash
PYTHONPATH=.. python experiments/sensitivity_analysis.py

```

### 2. Execute 36-Dimensional NSGA-II Pareto Search

Run the 100-trial evolutionary search over the 36 independent precision choices, enforcing programmatic verification of instantiated layers:

```bash
PYTHONPATH=.. python experiments/pareto_search.py

```

### 3. Evaluate on Locked Holdout Validation Set

Benchmark the discovered Pareto candidates against uniform precision baselines on the official SST-2 validation split:

```bash
PYTHONPATH=.. python scripts/evaluate_accuracy.py

```

---

## 📊 Summary of Final Empirical Results (NVIDIA RTX 4060 Laptop GPU)

Evaluated on `distilbert-base-uncased-finetuned-sst-2-english` ($B=16, L=128$):

| Configuration | Holdout Accuracy ($N=872$) | Median Latency ($L_{\text{P50}}$) | FP16 Layers Allocated |
| --- | --- | --- | --- |
| **FP32 Baseline** | 91.06% | 28.90 ms ($\pm0.35$) | 36 / 36 |
| **Uniform NF4** | 90.83% | 13.70 ms ($\pm0.26$) | 0 / 36 |
| **Uniform INT8** | 54.24% | 17.04 ms ($\pm0.41$) | 0 / 36 |
| **Pareto Candidate #3** | **91.17%** | **14.79 ms ($\pm0.13$)} | **6 / 36** |

* **Key Takeaway:** **Pareto Candidate #3** achieves task accuracy preservation comparable to the unquantized FP32 baseline while slashing median physical inference latency by **48.8%** (28.90 ms down to 14.79 ms).

---

## 📜 Citation

If you build upon this framework or reference its empirical findings, please cite the manuscript:

```bibtex
@article{kakari2026quantlab,
  title={QuantLab: Hardware-Aware Mixed-Precision Quantization and Pareto Optimization for Transformer Encoder Inference},
  author={Kakari, Ameyaw Emmanuel},
  journal={Department of Computer Science \& IT, Garden City University College},
  year={2026}
}

```

---

## 👤 Author

**Ameyaw Emmanuel Kakari**

*Department of Computer Science & IT*

*Garden City University College, Kumasi, Ghana*

📧 Email: aemmanuelkakari@gmail.com

---

*Licensed under the [MIT License](https://www.google.com/search?q=LICENSE).*