# QuantLab

### Hardware-Aware Mixed-Precision Quantization and Pareto Optimization for Transformer Encoder Inference

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

QuantLab is an open-source research framework for studying **hardware-aware mixed-precision quantization** of Transformer encoder models. It combines layer-isolated sensitivity analysis, low-bit NF4 weight quantization, CUDA-based latency measurement, and multi-objective NSGA-II search to investigate accuracy–latency trade-offs on physical GPU hardware.

The experimental study reported with this repository evaluates **DistilBERT on SST-2 using an NVIDIA GeForce RTX 4060 Laptop GPU**.

---

## Research Overview

QuantLab is motivated by two practical observations:

1. Transformer components can have substantially different sensitivity to quantization error.
2. Lower numerical precision does not necessarily produce a proportional reduction in measured wall-clock latency on physical hardware.

QuantLab therefore combines **layer-level perturbation analysis**, **direct hardware measurement**, and **multi-objective precision-allocation search**.

---

## Key Components

### Layer-Isolated NF4 Sensitivity Analysis

QuantLab perturbs one target projection layer at a time using simulated NF4 quantization/dequantization and measures the resulting logit mean squared error (LMSE) on a deterministic calibration subset.

Final aggregate sensitivity result:

* **FFN single-layer LMSE share:** 99.34%
* **MHSA single-layer LMSE share:** 0.66%

The evaluated DistilBERT model also shows particularly high sensitivity in the FFN output projections (`ffn.lin2`).

### 36-Dimensional Mixed-Precision Search

The search assigns each of the 36 target projection layers an independent precision choice:

* `NF4`
* `FP16`

The resulting binary search space contains:

$$
2^{36}=68,719,476,736
$$

possible precision assignments.

The study evaluates 100 NSGA-II trials. Therefore, the reported configurations are **observed non-dominated solutions**, not an exhaustive global Pareto frontier.

### Hardware Profiling

Latency is measured with CUDA events using fixed input shapes:

* Batch size: `16`
* Sequence length: `128`
* Warm-up iterations: `30`
* Active timing samples: `100`
* Optimization latency statistic: median (`P50`)

The resulting latency values are specific to the evaluated GPU and software stack.

---

## Repository Structure

QuantLab follows a conventional Python package layout:

```text
quantlab/
├── configs/
│   └── base_config.yaml
│
├── quantlab/
│   ├── __init__.py
│   ├── calibration/
│   ├── datasets/
│   ├── evaluation/
│   ├── experiments/
│   ├── hardware/
│   ├── models/
│   ├── quantization/
│   ├── visualization/
│   └── results/
│
├── results/
│   ├── nsgaii_pareto_front.csv
│   └── nsgaii_pareto_front_36dim.csv
│
├── scripts/
│   ├── benchmark_baseline.py
│   ├── benchmark_mixed_precision.py
│   ├── benchmark_uniform.py
│   ├── evaluate_accuracy.py
│   ├── generate_pareto_report.py
│   ├── test_milestone1.py
│   └── verify_env.py
│
├── tests/
│   ├── test_hardware_profiler.py
│   └── test_quantization.py
│
├── pyproject.toml
├── requirements.txt
└── README.md
```

The `quantlab/` directory contains the installable Python package. This separation is intentional and prevents collisions between QuantLab's internal modules and third-party packages such as Hugging Face's `datasets` package.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Emmanuelkk223/quantlab.git
cd quantlab
```

### 2. Check Out the Publication Version

```bash
git checkout v1.0-paper
```

### 3. Create a Python Environment

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

### 4. Install Dependencies

Upgrade `pip`:

```bash
python -m pip install --upgrade pip
```

Install the project in editable mode:

```bash
python -m pip install -e .
```

If the project dependencies are maintained separately in `requirements.txt`, install them with:

```bash
python -m pip install -r requirements.txt
```

The editable installation is recommended for development because it makes the `quantlab` package directly importable without requiring `PYTHONPATH` modifications.

### 5. Verify the Installation

Run:

```bash
python -c "from quantlab.datasets.glue_loader import GLUEDataLoader; print('QuantLab installation: OK')"
```

A successful installation should produce:

```text
QuantLab installation: OK
```

---

## Reproducing the Experiments

Run the following commands from the **repository root** after installing QuantLab with:

```bash
python -m pip install -e .
```

### 1. Run NF4 Sensitivity Analysis

```bash
python -m quantlab.experiments.sensitivity_analysis
```

Expected aggregate result:

```text
FFN Single-Layer LMSE Share (S_FFN): 99.34%
MHSA Single-Layer LMSE Share (S_MHSA): 0.66%
```

### 2. Run the 36-Dimensional NSGA-II Search

```bash
python -m quantlab.experiments.pareto_search
```

The search evaluates 100 trials over 36 independent binary precision decisions and exports the resulting search data and precision manifests under `results/`.

### 3. Evaluate the Discovered Candidates

Evaluate the discovered candidates on the locked holdout validation split:

```bash
python scripts/evaluate_accuracy.py
```

---

## Final Reported Results

The final generalization evaluation uses the **872-example official SST-2 validation split as a locked holdout validation set**. It is not the hidden-label SST-2 test split.

| Configuration           | Holdout Accuracy |      Median Latency | FP16 Layers |
| ----------------------- | ---------------: | ------------------: | ----------: |
| FP32 Baseline           |           91.06% |     28.83 ± 0.51 ms |     38 / 38 |
| Uniform NF4             |           90.83% |     13.92 ± 0.21 ms |      0 / 38 |
| Pareto Candidate #1     |           90.83% |     16.11 ± 0.15 ms |     12 / 38 |
| Pareto Candidate #2     |           90.94% |     17.07 ± 0.18 ms |     18 / 38 |
| **Pareto Candidate #3** |       **91.17%** | **14.79 ± 0.16 ms** |  **6 / 38** |
| Pareto Candidate #4     |           90.48% |     21.55 ± 0.23 ms |     18 / 38 |

### Main Observed Result

Pareto Candidate #3 maintained accuracy comparable to FP32 while reducing median inference latency from:

```text
28.83 ms → 14.79 ms
```

This corresponds to an approximate **48.7% reduction in median latency**.

The 0.11 percentage-point difference between Candidate #3 and FP32 is small and is **not presented as a statistically established accuracy improvement**.

---

## Search-Stage Result

The highest observed search-set accuracy was:

**98.54%**

This is a search-stage result used during NSGA-II optimization and is **not** the final generalization result.

The final locked holdout result for Pareto Candidate #3 is:

**91.17%**

This distinction is important because the search-set score was obtained during optimization, whereas the 91.17% result was obtained after the candidate was selected and evaluated on the locked holdout validation set.

---

## Experimental Data Protocol

The study uses three evaluation tiers.

### 1. Calibration Set

**256 examples** from the SST-2 training split, sampled deterministically with seed `42`.

Used for:

* Layer-isolated sensitivity analysis
* NF4 perturbation analysis
* Logit MSE measurement

### 2. Search Set

**1,024 examples** used during NSGA-II optimization.

Used for:

* Candidate evaluation
* Multi-objective optimization
* Accuracy–latency trade-off exploration

### 3. Locked Holdout Validation Set

The official **872-example SST-2 validation split**.

Reserved for:

* Final post-search evaluation
* Generalization assessment
* Reporting the final candidate results

The hidden-label SST-2 test split is **not used** for the reported generalization results.

---

## Reproducibility

For reproducibility, record and preserve the exact:

* Python version
* PyTorch version
* CUDA version
* NVIDIA driver version
* Transformers version
* BitsAndBytes version
* Optuna version
* Random seeds
* Experiment configuration
* Repository commit/tag
* Precision manifests
* Raw experiment logs
* GPU model and hardware configuration

The publication version is identified by the:

```text
v1.0-paper
```

Git tag.

Because latency measurements are hardware-dependent, reproducing the exact numerical latency values requires comparable GPU hardware and software configuration.

---

## Testing

Run the repository test suite with:

```bash
python -m pytest -q
```

For a more verbose test run:

```bash
python -m pytest -v
```

---

## Development

After installing the package in editable mode:

```bash
python -m pip install -e .
```

QuantLab modules can be imported using the package namespace:

```python
from quantlab.datasets.glue_loader import GLUEDataLoader
from quantlab.models.base_model import BaseModelWrapper
from quantlab.hardware.profiler import HardwareProfiler
from quantlab.quantization.engine import QuantizationEngine
```

This package namespace also prevents collisions with third-party dependencies. For example:

```python
from datasets import load_dataset
```

refers to the Hugging Face `datasets` library, while:

```python
from quantlab.datasets.glue_loader import GLUEDataLoader
```

refers to QuantLab's dataset utilities.

---

## Citation

If you use QuantLab or build on the results reported in the associated manuscript, please cite:

```bibtex
@article{kakari2026quantlab,
  title  = {QuantLab: Hardware-Aware Mixed-Precision Quantization and Pareto Optimization for Transformer Encoder Inference},
  author = {Ameyaw, Emmanuel Kakari},
  year   = {2026},
  note   = {Department of Computer Science \& IT, Garden City University}
}
```

---

## License

This project is released under the MIT License. See `LICENSE`.

---

## Author

**Ameyaw Emmanuel Kakari**

Department of Computer Science & IT
Garden City University College
Kumasi, Ghana

Email: `aemmanuelkakari@gmail.com`
