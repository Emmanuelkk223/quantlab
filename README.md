# QuantLab
### Hardware-Aware Mixed-Precision Quantization and Pareto Optimization for Transformer Architectures

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**QuantLab** is an open-source, research-grade framework designed to systematically profile, evaluate, and optimize layer-wise numerical precisions for Transformer models under real hardware constraints[cite: 1]. 

Traditional Post-Training Quantization (PTQ) often enforces uniform bit-widths across all layers, leading to catastrophic accuracy collapses or disappointing wall-clock speedups due to silicon execution bottlenecks. QuantLab bridges the gap between theoretical quantization mathematics and physical GPU execution by combining microsecond-accurate CUDA event timing, single-layer logit perturbation sensitivity analysis, and hardware-in-the-loop Multi-Objective NSGA-II Pareto optimization.

---

## 📂 Repository Structure

```text
quantlab/
├── quantlab/
│   ├── datasets/         # GLUE benchmark data loaders (SST-2)
│   ├── evaluation/       # Task accuracy, loss, and F1 evaluation engines
│   ├── experiments/      # Single-layer sensitivity analysis & NSGA-II Pareto search
│   ├── hardware/         # Asynchronous CUDA event latency profiler & VRAM tracker
│   ├── models/           # BaseModelWrapper & transformer layer introspection parser
│   ├── quantization/     # Uniform precision engine (FP32, FP16, INT8, NF4) & precision mapper
│   └── visualization/    # Publication-ready Pareto frontier plotting tools
├── scripts/              # Executable entry-point benchmarks & report generators
├── results/              # Saved figures and experimental evaluation artifacts
├── .gitignore
└── README.md

```

---

## 🛠️ Key Features

1. **Hardware-Aware Profiling:** Uses non-blocking CUDA stream events (`torch.cuda.Event`) and explicit warm-up iterations ($N_{\text{warmup}} = 30$) to stabilize hardware clocks against mobile Dynamic Boost thermal envelopes and eliminate cold-start timing bias.


2. **Layer Introspection & Sensitivity Analysis:** Recursively parses transformer architectures to isolate individual linear layers ($Q, K, V, O$ and FFN blocks) and quantify vulnerability using logit Mean Squared Error ($\mathcal{L}_{\text{MSE}}$) perturbation sweeps.
3. **Multi-Objective NSGA-II Search:** Leverages Optuna to run hardware-in-the-loop multi-objective optimization over discrete layer precision choices ($\text{INT4}$ vs. $\text{FP16}$), discovering Pareto-optimal schedules that balance validation accuracy and wall-clock latency.


4. **Empirical Verification:** Extensively benchmarked on an **NVIDIA GeForce RTX 4060 Laptop GPU** using DistilBERT on the Stanford Sentiment Treebank (SST-2) validation split.



---

## ⚙️ Installation

1. Clone the repository:
```bash
git clone [https://github.com/Emmanuelkk223/quantlab.git](https://github.com/Emmanuelkk223/quantlab.git)
cd quantlab

```


2. Create and activate a Python virtual environment:
```bash
conda create -n quantlab python=3.10 -y
conda activate quantlab

```


3. Install dependencies:
```bash
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install transformers datasets accelerate bitsandbytes optuna matplotlib seaborn pandas tqdm

```



---

## 🚀 Usage

### 1. Run Uniform Precision Benchmarks

Compare memory footprints, peak VRAM allocations, and inference latencies across FP32, FP16, INT8 (`llm_int8`), and INT4 (`bitsandbytes` NF4):

```bash
python quantlab/scripts/benchmark_uniform.py

```

### 2. Evaluate Task Accuracy on SST-2

Measure downstream cross-entropy loss and top-1 validation accuracy across precision modes:

```bash
python quantlab/scripts/evaluate_accuracy.py

```

### 3. Run Layer Sensitivity Sweep

Isolate and perturb individual transformer layers to map structural vulnerability:

```bash
python quantlab/experiments/sensitivity_analysis.py

```

### 4. Execute Multi-Objective Pareto Search

Run hardware-in-the-loop NSGA-II optimization to discover optimal mixed-precision layer configurations:

```bash
python quantlab/experiments/pareto_search.py

```

---

## 📊 Summary of Empirical Findings

* **FFN Vulnerability Supremacy:** Feed-Forward Network linear layers ($\text{lin1}, \text{lin2}$) account for over **82.4% of total network logit distortion** under low-bit quantization, whereas Multi-Head Self-Attention (MHSA) projections remain highly resilient.
* **Quantization Noise Regularization:** Our hardware-in-the-loop Pareto search identified mixed-precision schedules achieving up to **92.19% validation accuracy**, outperforming the unquantized FP32 baseline (91.06%) while cutting inference latency by **30.0%**.


* **Kernel Switching Overhead:** Heterogeneous layer precision schedules introduce runtime latency floors due to un-fused hardware kernel transitions between cuBLAS FP16 kernels and `bitsandbytes` NF4 weight-dequantization routines on consumer mobile GPUs.



---

## 📜 Citation

If you use this framework or build upon its findings in your research, please cite:

```bibtex
@article{kakari2026quantlab,
  title={QuantLab: Hardware-Aware Mixed-Precision Quantization and Pareto Optimization for Transformer Inference},
  author={Kakari, Ameyaw Emmanuel},
  journal={Department of Computer Science, Garden City University College},
  year={2026}
}

```

---

## 👤 Author

**Ameyaw Emmanuel Kakari**

*Department of Computer Science & IT*

*Garden City University College, Kumasi, Ghana*

📧 Email: aemmanuelkakari@gmail.com / ameyaw.kakari@gcuc.edu.gh

---

*Licensed under the [MIT License](https://www.google.com/search?q=LICENSE).*
