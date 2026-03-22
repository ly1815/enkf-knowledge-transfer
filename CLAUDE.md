# BiotechBioeng — CLAUDE.md

## Project Overview

Research project implementing **Ensemble Kalman Filter (EnKF) with knowledge transfer** for modelling Chinese Hamster Ovary (CHO) cell culture bioprocesses. Companion code for a research paper.

## Repository Structure

```
BiotechBioeng/
├── cho_enkf/                   # Python package (all logic)
│   ├── config.py               # RUN_NAME, paths, all constants — edit here first
│   ├── data_loader.py
│   ├── model.py                # Volume ODE + process model
│   ├── enkf.py                 # EnKF classes and all runner functions
│   ├── analysis.py             # R², convergence, correlation
│   ├── plotting.py             # All publication figures
│   └── io_utils.py             # save_pkl / load_pkl wrappers
├── scripts/                    # Numbered pipeline — run in order
│   ├── 01_ensemble_tuning.py   # Figs A (tuning), B (T127 overlay), C (GS46 overlay)
│   ├── 02_longterm_pred.py     # Figs D (long-term), E+F (params), G (correlation), R²/conv tables
│   ├── 03_irregular.py         # Fig K (irregular profiles)
│   ├── 04_sensitivity.py       # Figs H+I (prior width), J (param sensitivity)
│   └── 05_comparisons.py       # EnKF vs reparametrised model (Reviewer 2, point 5)
├── data/                       # Input Excel datasets (never modify)
├── results/{RUN_NAME}/         # All outputs (gitignored)
│   ├── pkl/                    # Pickle files
│   └── figures/                # PNG figures
└── original.ipynb              # Reference only — do not edit
```

## Virtual Environment

**Tool**: Poetry | **Location**: `.venv/` (gitignored)

```bash
poetry install          # create venv + install deps
poetry shell            # activate
# OR: source .venv/Scripts/activate  (Windows)
```

## Key Commands

```bash
# Run the pipeline (in order)
poetry run python scripts/01_ensemble_tuning.py   # ~2–4 h
poetry run python scripts/02_longterm_pred.py      # ~30–60 min
poetry run python scripts/03_irregular.py          # ~30–60 min
poetry run python scripts/04_sensitivity.py        # ~2–4 h
poetry run python scripts/05_comparisons.py        # < 5 min (requires step 1 pkl)

# Add a dependency
poetry add <package>
```

## Output Organisation

- Change `RUN_NAME` in `cho_enkf/config.py` to version a new experiment
- All pkl files → `results/{RUN_NAME}/pkl/`
- All figures → `results/{RUN_NAME}/figures/`
- Each script generates its own figures immediately after computation

## Data

Input Excel files in `data/`:
- `CHO_T127_*` — Cell Line A (T127 strain): shake flask + bioreactor at 36.5°C and 32°C
- `CHO_GS46_F_*` — Cell Line B (GS46 strain): different feed strategies
- Each file has sheets: `schedule`, `feed`, `exp_meas`

## Reviewer Comments Context (Reviewer 2)

The manuscript is under revision. Points 1–4 and 6–7 are addressed in code and text. Point 5 (quantitative comparison vs reparametrisation) is addressed by `05_comparisons.py`:
- Compares EnKF (CHO_GS46_F_C_Inv, ensemble size 75, full data) against an open-loop simulation using `CHO_GS46_F_C_Inv_PARAMETERS` (dataset-specific reparametrised parameters in `config.py`)
- The `documents/` folder (paper, reviewer comments) is gitignored

## Gotchas

- `results/` is gitignored — pkl files and figures do not sync via git
- `original.ipynb` is the legacy monolithic notebook in `old_notebooks/`; the `cho_enkf` package is canonical
- No test suite — research project
- Scripts must be run from the project root (they add the root to `sys.path`)
- `CHO_GS46_F_C_Inv_PARAMETERS` in `config.py` holds dataset-specific reparametrised parameters (from Kotidis 2019 strategic framework) used in `06_comparisons.py`
