# QuantLab

### Hardware-Aware Mixed-Precision Quantization and Pareto Optimization for Transformer Encoder Inference

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

QuantLab is an open-source research framework for studying **hardware-aware mixed-precision quantization** of Transformer encoder models. It combines layer-isolated sensitivity analysis, low-bit NF4 weight quantization, CUDA-based latency measurement, and multi-objective NSGA-II search to investigate accuracy–latency trade-offs on physical GPU hardware.

The experimental study reported with this repository evaluates **DistilBERT on SST-2 using an NVIDIA GeForce RTX 4060 Laptop GPU**.

## Research Overview

QuantLab is motivated by two practical observations:

1. Transformer components can have substantially different sensitivity to quantization error.
2. Lower numerical precision does not necessarily produce a proportional reduction in measured wall-clock latency on physical hardware.

QuantLab therefore combines **layer-level perturbation analysis** with **direct hardware measurement** and a **multi-objective precision-allocation search**.

## Key Components

### Layer-isolated NF4 sensitivity analysis

QuantLab perturbs one target projection layer at a time using simulated NF4 quantization/dequantization and measures the resulting logit mean squared error (LMSE) on a deterministic calibration subset.

Final aggregate sensitivity result:

- **FFN single-layer LMSE share:** 99.34%
- **MHSA single-layer LMSE share:** 0.66%

The evaluated DistilBERT model also shows particularly high sensitivity in the FFN output projections (`ffn.lin2`).

### 36-dimensional mixed-precision search

The search assigns each of the 36 target projection layers an independent precision choice:

- `NF4`
- `FP16`

The resulting binary search space contains:

\[
2^{36}=68,719,476,736
\]

possible precision assignments. The study evaluates 100 NSGA-II trials, so the reported configurations are **observed non-dominated solutions**, not an exhaustive global Pareto frontier.

### Hardware profiling

Latency is measured with CUDA events using fixed input shapes:

- Batch size: `16`
- Sequence length: `128`
- Warm-up iterations: `30`
- Active timing samples: `100`
- Optimization latency statistic: median (`P50`)

The resulting latency values are specific to the evaluated GPU and software stack.

## Repository Structure

```text
quantlab/
├── calibration/          # Calibration utilities
├── configs/              # Experiment configuration files
├── datasets/             # Dataset loaders and split utilities
├── evaluation/           # Evaluation metrics and helpers
├── experiments/          # Sensitivity analysis and NSGA-II search
├── hardware/             # CUDA latency and GPU memory profiling
├── models/               # Model wrappers and layer introspection
├── quantization/         # Quantization engines and precision mapping
├── quantlab/             # Generated package/result assets
├── results/              # Search results and exported artifacts
├── scripts/              # Benchmarking and evaluation scripts
├── tests/                # Unit tests
├── visualization/        # Plotting utilities
├── pyproject.toml        # Project metadata and build configuration
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Emmanuelkk223/quantlab.git
cd quantlab
```

### 2. Check out the publication version

```bash
git checkout v1.0-paper
```

### 3. Create a Python environment

Using Conda:

```bash
conda create -n quantlab python=3.10 -y
conda activate quantlab
```

Or using Python `venv`:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Reproducing the Experiments

Run the following commands from the **repository root**.

### 1. Run NF4 sensitivity analysis

```bash
PYTHONPATH=. python experiments/sensitivity_analysis.py
```

Expected aggregate result:

```text
FFN Single-Layer LMSE Share (S_FFN): 99.34%
MHSA Single-Layer LMSE Share (S_MHSA): 0.66%
```

### 2. Run the 36-dimensional NSGA-II search

```bash
PYTHONPATH=. python experiments/pareto_search.py
```

The search evaluates 100 trials over 36 independent binary precision decisions and exports the resulting search data and precision manifests under `results/`.

### 3. Evaluate the discovered candidates on the locked holdout validation split

```bash
PYTHONPATH=. python scripts/evaluate_accuracy.py
```

## Final Reported Results

The final generalization evaluation uses the **872-example official SST-2 validation split as a locked holdout validation set**. It is not the hidden-label SST-2 test split.

| Configuration | Holdout Accuracy | Median Latency | FP16 Layers |
|---|---:|---:|---:|
| FP32 Baseline | 91.06% | 28.83 ± 0.51 ms | 38 / 38 |
| Uniform NF4 | 90.83% | 13.92 ± 0.21 ms | 0 / 38 |
| Pareto Candidate #1 | 90.83% | 16.11 ± 0.15 ms | 12 / 38 |
| Pareto Candidate #2 | 90.94% | 17.07 ± 0.18 ms | 18 / 38 |
| **Pareto Candidate #3** | **91.17%** | **14.79 ± 0.16 ms** | **6 / 38** |
| Pareto Candidate #4 | 90.48% | 21.55 ± 0.23 ms | 18 / 38 |

### Main observed result

Pareto Candidate #3 maintained accuracy comparable to FP32 while reducing median inference latency from:

```text
28.83 ms  →  14.79 ms
```

This corresponds to an approximate **48.7% reduction in median latency**.

The 0.11 percentage-point difference between Candidate #3 and FP32 is small and is not presented as a statistically established accuracy improvement.

## Search-Stage Result

The highest observed search-set accuracy was **98.54%**.

This is a search-stage result used during NSGA-II optimization and is **not** the final generalization result. The final locked holdout result for Candidate #3 is **91.17%**.

## Experimental Data Protocol

The study uses three evaluation tiers:

1. **Calibration set:** 256 examples from the SST-2 training split, sampled deterministically with seed 42 for sensitivity analysis.
2. **Search set:** 1,024 examples used during NSGA-II optimization.
3. **Locked holdout validation set:** the official 872-example SST-2 validation split, reserved for post-search evaluation.

The hidden-label SST-2 test split is not used for the reported generalization results.

## Reproducibility

For reproducibility, record and preserve the exact:

- Python version
- PyTorch version
- CUDA version
- NVIDIA driver version
- Transformers version
- BitsAndBytes version
- Optuna version
- random seeds
- experiment configuration
- repository commit/tag
- precision manifests
- raw experiment logs

The publication version is identified by the `v1.0-paper` tag.

## Testing

Run the repository tests with:

```bash
python -m pytest -q
```

## Citation

If you use QuantLab or build on the results reported in the associated manuscript, please cite the paper:

```bibtex
@article{kakari2026quantlab,
  title  = {QuantLab: Hardware-Aware Mixed-Precision Quantization and Pareto Optimization for Transformer Encoder Inference},
  author = {Kakari, Ameyaw Emmanuel},
  year   = {2026},
  note   = {Department of Computer Science \& IT, Garden City University College}
}
```

## License

This project is released under the MIT License. See `LICENSE`.

## Author

**Ameyaw Emmanuel Kakari**  
Department of Computer Science & IT  
Garden City University College  
Kumasi, Ghana  
Email: `aemmanuelkakari@gmail.com`
