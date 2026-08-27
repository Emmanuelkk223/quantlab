# QuantLab
### Hardware-Aware Mixed-Precision Quantization and Pareto Optimization for Transformer Inference

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License:MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**QuantLab** is an open-source, research-grade framework designed to systematically profile, evaluate, and optimize layer-wise numerical precisions for Transformer models under real hardware constraints. 

Traditional Post-Training Quantization (PTQ) often enforces uniform bit-widths across all layers, leading to catastrophic accuracy collapses or disappointing wall-clock speedups due to silicon execution bottlenecks. QuantLab bridges the gap between theoretical quantization mathematics and physical GPU execution by combining high-resolution CUDA event timing, simulated 4-bit NormalFloat (NF4) logit perturbation sensitivity analysis, and hardware-in-the-loop Multi-Objective NSGA-II Pareto optimization.

---

## 📂 Repository Structure

```text
quantlab/
├── calibration/          # Calibration routines and dataloader separation
├── configs/              # YAML configuration files for experiments
├── datasets/             # Isolated GLUE benchmark data loaders (SST-2 splits)
├── evaluation/           # Task accuracy, loss, and hardware profiling engines
├── experiments/          # Single-layer NF4 sensitivity analysis & NSGA-II Pareto search
├── hardware/             # High-resolution CUDA event latency profiler & memory tracker
├── models/               # BaseModelWrapper & transformer layer introspection parser
├── quantization/         # Uniform precision engine (FP32, FP16, INT8, NF4) & precision mapper
├── scripts/              # Executable entry-point benchmarks & test scripts
├── tests/                # Unit tests for hardware profiling and quantization modules
├── visualization/        # Publication-ready Pareto frontier plotting tools
├── .gitignore
└── README.md

```

---

## 🛠️ Key Framework Features

1. **Hardware-Aware Profiling:** Utilizes non-blocking CUDA stream events (`torch.cuda.Event`) and explicit warm-up iterations ($N_{\text{warmup}} = 30$) to stabilize hardware clocks against mobile Dynamic Boost thermal envelopes and eliminate cold-start timing bias.
2. **Layer Introspection & Sensitivity Analysis:** Recursively parses transformer architectures to isolate individual linear layers and quantify vulnerability using simulated 4-bit NormalFloat (NF4) logit Mean Squared Error ($\mathcal{L}_{\text{MSE}}$) sweeps.
3. **Multi-Objective NSGA-II Search:** Leverages Optuna to run hardware-in-the-loop multi-objective optimization over discrete layer precision choices ($\text{INT4}$ vs. $\text{FP16}$), discovering Pareto-optimal schedules that balance validation accuracy and wall-clock latency.
4. **Data Isolation:** Implements strict separation between calibration sets, search optimization sets, and holdout test sets to prevent selection bias and validation leakage.

---

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/Emmanuelkk223/quantlab.git
cd quantlab

```


2. Create and activate a Python virtual environment:
```bash
conda create -n quantlab python=3.10 -y
conda activate quantlab

```


3. Install dependencies:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate bitsandbytes optuna matplotlib seaborn pandas tqdm

```



---

## 🚀 Usage

### 1. Run Simulated NF4 Sensitivity Sweep

Isolate and perturb individual transformer layers using simulated 4-bit NormalFloat quantization to map structural vulnerability:

```bash
PYTHONPATH=.. python experiments/sensitivity_analysis.py

```

### 2. Execute Multi-Objective NSGA-II Pareto Search

Run hardware-in-the-loop NSGA-II optimization constrained by sensitivity profiles to discover optimal mixed-precision layer configurations:

```bash
PYTHONPATH=.. python experiments/pareto_search.py

```

---

## 📊 Summary of Empirical Findings (RTX 4060 Laptop GPU)

* **FFN Vulnerability Supremacy:** Feed-Forward Network linear layers account for **99.34% of aggregate logit distortion** under simulated 4-bit (NF4) quantization, whereas Multi-Head Self-Attention (MHSA) projections remain highly resilient.
* **Pareto Frontier Optimization:** Heterogeneous precision schedules identified by QuantLab achieved up to **92.19% holdout validation accuracy** (surpassing the unquantized FP32 baseline of 91.06%) while cutting measured inference latency significantly.
* **Kernel Switching Overhead:** Mixed-precision schedules introduce a runtime latency floor ($18.46\text{--}20.33\text{ ms}$) compared to uniform INT4 ($8.7\text{ ms}$) due to un-fused hardware kernel transitions on general-purpose GPU execution stacks.

---

## 📜 Citation

If you use this framework or build upon its findings in your research, please cite the manuscript:

```bibtex
@article{kakari2026quantlab,
  title={QuantLab: Hardware-Aware Mixed-Precision Quantization and Pareto Optimization for Transformer Inference},
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