# Model-Enabled Knowledge Transfer across Cell Lines, Culture Scales and Conditions

[![DOI](https://img.shields.io/badge/DOI-10.1002/bit.70269-blue.svg)](https://doi.org/10.1002/bit.70269)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the code accompanying the paper:

> **Model-Enabled Knowledge Transfer across Cell Lines, Culture Scales and Conditions**
> Luxi Yu, Antonio del Rio Chanona, Cleo Kontoravdi
> *Biotechnology and Bioengineering* (2026). doi: [10.1002/bit.70269](https://doi.org/10.1002/bit.70269)

## Overview

We present an **Ensemble Kalman Filter (EnKF)** framework for dual state and parameter estimation in Chinese Hamster Ovary (CHO) cell culture bioprocesses. The EnKF recursively assimilates process measurements to update uncertain kinetic parameters and predict system states, enabling a mechanistic model calibrated on one system to be transferred to new cell lines, scales, and operating conditions — **without reparametrisation and using only a single experimental dataset**.

The framework is evaluated across six CHO cell experimental datasets differing in scale, cell line, temperature, and feeding strategy, demonstrating accurate reconstruction of system dynamics and progressive improvement in long-term predictions as new data become available.

## Repository Structure

```
enkf-knowledge-transfer/
├── cho_enkf/                          # Python package
│   ├── config.py                      # Constants, parameters, noise definitions
│   ├── data_loader.py                 # Load Excel datasets (3 sheets each)
│   ├── model.py                       # Mechanistic ODE model (RK4/LSODA)
│   ├── enkf.py                        # EnKF classes and runner functions
│   ├── analysis.py                    # R², convergence, correlation analysis
│   ├── plotting.py                    # Publication-quality figure generation
│   └── io_utils.py                    # Pickle I/O and path management
│
├── scripts/                           # Numbered execution pipeline
│   ├── 01_ensemble_tuning.py          # Sweep ensemble sizes
│   ├── 02_longterm_pred.py            # Long-term prediction
│   ├── 03_irregular.py                # Irregular measurement schedule (48/72 h)
│   ├── 04_priorcov_sensitivity.py     # Prior covariance width sensitivity
│   ├── 05_priormean_sensitivity.py    # Prior mean perturbation sensitivity
│   └── 06_comparisons.py              # EnKF vs reparametrised model
│
├── data/                              # 6 CHO experimental datasets (Excel)
├── results/                           # Generated outputs (gitignored)
└── pyproject.toml                     # Poetry project metadata
```

## Datasets

| # | Cell Line | Type | Volume (mL) | Temp (°C) | Feed |
|---|-----------|------|-------------|-----------|------|
| 1 | A (CHO-T127) | Shake flask | 100 | 36.5 | Feed C |
| 2 | A (CHO-T127) | Bioreactor | 900 | 36.5 | Feed C |
| 3 | A (CHO-T127) | Bioreactor | 900 | 32* | Feed C |
| 4 | B (CHO-GS46) | Shake flask | 50 | 36.5 | Feed C |
| 5 | B (CHO-GS46) | Shake flask | 50 | 36.5 | Feed U |
| 6 | B (CHO-GS46) | Shake flask | 50 | 36.5 | Feed U +40% |

*Temperature downshift from 36.5°C to 32°C on day 6.

Each Excel file contains three sheets: `schedule` (feed timing), `feed` (feed concentrations), and `exp_meas` (measured state variables with standard deviations).

## State Variables

Xv (viable cell density), mAb (antibody titre), Glc (glucose), Amm (ammonia), Gln (glutamine), Lac (lactate), Glu (glutamate), Asn (asparagine).

## Quick Start

### Install dependencies

```bash
pip install poetry
poetry install
```

### Run the pipeline

Scripts should be run in numbered order. Each saves intermediate `.pkl` files so figures can be regenerated without re-running the EnKF (set `LOAD_FROM_PKL = True` at the top of each script).

```bash
poetry run python scripts/01_ensemble_tuning.py       # ~2-4 h
poetry run python scripts/02_longterm_pred.py          # ~30-60 min
poetry run python scripts/03_irregular.py              # ~30-60 min
poetry run python scripts/04_priorcov_sensitivity.py   # ~2-4 h
poetry run python scripts/05_priormean_sensitivity.py  # ~2-4 h
poetry run python scripts/06_comparisons.py            # < 5 min
```

Outputs are saved to `results/{RUN_NAME}/` (controlled by `RUN_NAME` in `cho_enkf/config.py`).

## Citation

If you use this code, please cite:

```bibtex
@article{yu2026model,
  title     = {Model-Enabled Knowledge Transfer across Cell Lines, Culture Scales and Conditions},
  author    = {Yu, Luxi and del Rio Chanona, Antonio and Kontoravdi, Cleo},
  journal   = {Biotechnology and Bioengineering},
  volume    = {0},
  pages     = {e70269},
  year      = {2026},
  doi       = {10.1002/bit.70269}
}
```

## Supplementary Information

The supplementary information for the paper is available in this repository: [Supplementary_Information_Model_Enabled_Knowledge_Transfer_across_Cell_Lines__Culture_Scales_and_Conditions.pdf](Supplementary_Information_Model_Enabled_Knowledge_Transfer_across_Cell_Lines__Culture_Scales_and_Conditions.pdf)

## License

This project is licensed under the [MIT License](LICENSE).
