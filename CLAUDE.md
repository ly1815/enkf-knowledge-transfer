# BiotechBioeng — CLAUDE.md

## Project Overview

Research project implementing **Ensemble Kalman Filter (EnKF) with knowledge transfer** for modeling Chinese Hamster Ovary (CHO) cell culture bioprocesses. Active development with reviewer feedback incorporation.

## Repository Structure

```
BiotechBioeng/
├── EnKF-knowledge_transfer.ipynb   # Main computation notebook (EnKF implementation)
├── correlation.ipynb               # Parameter correlation analysis notebook
├── data/                           # Input CHO cell culture datasets (Excel)
├── correction_v1_ensemble_size/    # Ensemble tuning results (~2GB, gitignored)
├── correction_v2/                  # Selected best results (pickle files)
├── pyproject.toml                  # Poetry project config
└── poetry.lock                     # Locked dependencies
```

## Virtual Environment

**Tool**: Poetry
**Location**: `.venv/` (in project root, gitignored)

### Setup
```bash
# Install Poetry if not already installed
pip install poetry

# Install dependencies and create venv
poetry install

# Activate the virtual environment
poetry shell
# OR use: source .venv/Scripts/activate  (Windows)
```

### Key Dependencies
- numpy, pandas, scipy — numerical/data processing
- matplotlib, seaborn — visualization
- tqdm — progress bars
- openpyxl — Excel file reading
- jupyter, ipykernel, notebook — notebook environment

## Key Commands

```bash
# Launch Jupyter notebook
poetry run jupyter notebook

# Run a specific notebook non-interactively
poetry run jupyter nbconvert --to notebook --execute EnKF-knowledge_transfer.ipynb

# Add a new dependency
poetry add <package>

# Update dependencies
poetry update
```

## Data

Input data in `data/` — CHO cell culture experiments:
- `CHO_GS46_F_*` — GS46 strain datasets
- `CHO_T127_*` — T127 strain datasets
- `dataset_summary_table.xlsx` — overview

## Notes

- No test suite — research project with all code in notebooks
- Pickle outputs (`*.pkl`) are gitignored; large result dirs (`correction_v1_ensemble_size/`) are gitignored
- `correlation.ipynb` is currently modified (reviewer comments in progress)
