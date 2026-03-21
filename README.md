# EnKF with Knowledge Transfer for CHO Bioprocess Modelling

Reproducible research code accompanying the paper:

> **[Paper title]**
> [Authors]
> [Journal, Year]

This repository implements an **Ensemble Kalman Filter (EnKF)** for dual state and parameter estimation in Chinese Hamster Ovary (CHO) cell culture bioprocesses, with cross-condition knowledge transfer.

---

## Overview

The EnKF is applied to six CHO cell culture datasets covering two cell lines (T127 and GS46), multiple bioreactor scales, and different feeding strategies. The workflow:

1. Integrates the bioreactor volume ODE
2. Runs a nominal forward simulation with literature parameters (Kotidis et al. 2019)
3. Tunes ensemble size by sweeping RMSE across sizes `[10, 25, 50, 75]`
4. Performs long-term state prediction using the best ensemble size
5. Tests robustness with an irregular 48/72 h measurement schedule
6. Quantifies prior width sensitivity and ±20% mean parameter sensitivity

---

## Repository Structure

```
BiotechBioeng/
│
├── cho_enkf/                   # Python package
│   ├── config.py               # All constants: RUN_NAME, paths, parameters, noise
│   ├── data_loader.py          # Load Excel datasets
│   ├── model.py                # Volume integration, kinetic model (ODE step)
│   ├── enkf.py                 # EnKF classes + all runner functions
│   ├── analysis.py             # R², convergence tables, correlation matrix
│   ├── plotting.py             # All publication-quality figure functions
│   └── io_utils.py             # Pickle save/load, path construction
│
├── scripts/                    # Numbered execution pipeline
│   ├── 01_ensemble_tuning.py   # Load data → run EnKF sweep → save pkl
│   ├── 02_longterm_pred.py     # Long-term forecasting with best ensemble size
│   ├── 03_irregular.py         # EnKF with 48/72 h irregular measurements
│   ├── 04_sensitivity.py       # Prior width + ±20% parameter sensitivity
│   └── 05_figures.py           # Load all pkl → generate all publication figures
│
├── data/                       # CHO experimental datasets (Excel)
│   ├── CHO_T127_flask_PMJ.xlsx
│   ├── CHO_T127_SNS_36.5.xlsx
│   ├── CHO_T127_SNS_32.xlsx
│   ├── CHO_GS46_F_C_Inv.xlsx
│   ├── CHO_GS46_F_all.xlsx
│   └── CHO_GS46_F_all_pl40.xlsx
│
├── results/                    # Generated outputs (gitignored)
│   └── run_v1/
│       ├── pkl/                # All intermediate pickle files
│       │   └── sensitivity/    # Sensitivity analysis pickles
│       └── figures/            # All PNG figures
│           └── sensitivity/    # Sensitivity figures
│
├── original.ipynb              # Original monolithic notebook (reference only)
├── pyproject.toml
├── poetry.lock
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
# Install Poetry (if not already installed)
pip install poetry

# Create virtual environment and install all dependencies
poetry install

# Activate the environment
poetry shell
# OR: source .venv/Scripts/activate   (Windows)
#     source .venv/bin/activate        (Linux/macOS)
```

### 2. Set the run name (optional)

Edit `cho_enkf/config.py`:
```python
RUN_NAME = "run_v1"   # all outputs go to results/run_v1/
```

Change this string to version a new experiment. Old results are preserved.

### 3. Run the pipeline

Each script is independent and numbered. Run them in order, or skip steps by loading pre-computed pkl files.

```bash
# Step 1 (~2–4 h depending on hardware): ensemble tuning
poetry run python scripts/01_ensemble_tuning.py

# Step 2 (~30–60 min): long-term prediction with best ensemble size
poetry run python scripts/02_longterm_pred.py

# Step 3 (~30–60 min): irregular measurement schedule
poetry run python scripts/03_irregular.py

# Step 4 (~2–4 h): sensitivity analyses (prior width + ±20% params)
poetry run python scripts/04_sensitivity.py

# Step 5 (< 5 min): generate all publication figures
poetry run python scripts/05_figures.py
```

> **Tip**: Steps 1–4 are compute-heavy. Run them once. Step 5 can be re-run repeatedly
> to adjust figure aesthetics without re-running any EnKF.

### 4. Find outputs

All pickle files → `results/run_v1/pkl/`
All figures → `results/run_v1/figures/`

---

## Datasets

| Dataset | Cell Line | Condition |
|---|---|---|
| `CHO_T127_flask_PMJ` | T127 (Cell Line A) | Shake flask, 36.5°C |
| `CHO_T127_SNS_36.5` | T127 (Cell Line A) | Bioreactor, 36.5°C |
| `CHO_T127_SNS_32` | T127 (Cell Line A) | Bioreactor, 32°C |
| `CHO_GS46_F_C_Inv` | GS46 (Cell Line B) | Feed C |
| `CHO_GS46_F_all` | GS46 (Cell Line B) | Feed U |
| `CHO_GS46_F_all_pl40` | GS46 (Cell Line B) | Feed U +40% |

Each Excel file contains three sheets: `schedule` (feed schedule), `feed` (feed concentrations), `exp_meas` (measured state variables ± std).

---

## State Variables

| Symbol | Description | Unit |
|---|---|---|
| Xv | Viable cell density | cell L⁻¹ |
| mAb | Monoclonal antibody titre | mg L⁻¹ |
| Glc | Glucose | mM |
| Amm | Ammonia | mM |
| Gln | Glutamine | mM |
| Lac | Lactate | mM |
| Glu | Glutamate | mM |
| Asn | Asparagine | mM |

---

## Key Configuration (`cho_enkf/config.py`)

| Variable | Description |
|---|---|
| `RUN_NAME` | Experiment version tag; controls output folder |
| `TUNING_ENSEMBLE_SIZES` | Sizes swept in Step 1 (default `[10, 25, 50, 75]`) |
| `BEST_ENSEMBLE_SIZES` | Best size per dataset (set after reviewing Step 1 results) |
| `MEAN_PARAMETERS` | Nominal model parameters from Kotidis et al. 2019 |
| `PARAMETERS_ENSEMBLE_COVARIANCE` | Prior width for each parameter |
| `DATASET_NOISE_VARIANCES` | Process (Q) and observation (R) noise per dataset |
| `KQ_DICT` / `KR_DICT` | Scaling factors for Q and R matrices |
| `PRIOR_WIDTH_SCALES` | Scales tested in sensitivity analysis |
| `PARAM_SENS_PERTURBATION` | Fraction for ±% sensitivity (default 0.20) |

---

## Package API Summary

### `cho_enkf.enkf`
- `run_enkf_with_tuning(...)` — sweep ensemble sizes
- `enkf_long_pred_best_ensemble_size(...)` — long-term prediction
- `run_pipeline_irregular_48_72(...)` — irregular measurement schedule
- `run_enkf_with_mean_params(...)` — parameter sensitivity runs

### `cho_enkf.analysis`
- `compute_r2_table(...)` — R² for all datasets
- `compute_overall_convergence_table(...)` — parameter convergence %
- `get_posterior_param_matrix(...)` — posterior ensemble for correlation

### `cho_enkf.plotting`
- `plot_rmse_variance_and_computation_time_all(...)` — tuning figure
- `overlay_T127_subplots_with_errorbars(...)` — T127 comparison
- `overlay_gs46_subplots_with_errorbars(...)` — GS46 comparison
- `plot_longterm_pred_ensemble_simulation_errorbar(...)` — long-term pred
- `plot_parameter_comparison_across_datasets(...)` — cross-dataset params
- `plot_posterior_param_correlation(...)` — correlation heatmap
- `plot_prior_width_sensitivity_rmse(...)` — prior width RMSE bar
- `plot_param_sensitivity_comparison(...)` — ±20% sensitivity

---

## Dependencies

| Package | Purpose |
|---|---|
| numpy, scipy | Numerical computation, ODE integration |
| pandas | Data loading and tabular analysis |
| matplotlib, seaborn | Plotting |
| openpyxl | Reading Excel files |
| tqdm | Progress bars |
| jupyter, ipykernel | Interactive exploration (optional) |

---

## Nominal Model

Parameters from **Kotidis et al. (2019)** for CHO-T127 shake flask cultures.
The model describes growth, death, and metabolite dynamics via Monod-type kinetics,
with ammonia and lactate inhibition, and a full yield-based metabolic network.

---

## Citation

If you use this code, please cite:

```bibtex
@article{[key],
  title   = {[Title]},
  author  = {[Authors]},
  journal = {[Journal]},
  year    = {[Year]},
  doi     = {[DOI]}
}
```

---

## License

[License]
