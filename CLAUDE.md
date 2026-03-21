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
│   ├── 01_ensemble_tuning.py
│   ├── 02_longterm_pred.py
│   ├── 03_irregular.py
│   ├── 04_sensitivity.py
│   └── 05_figures.py           # Decoupled from computation; re-run freely
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
poetry run python scripts/05_figures.py            # < 5 min

# Add a dependency
poetry add <package>
```

## Output Organisation

- Change `RUN_NAME` in `cho_enkf/config.py` to version a new experiment
- All pkl files → `results/{RUN_NAME}/pkl/`
- All figures → `results/{RUN_NAME}/figures/`
- `05_figures.py` loads pkl and regenerates figures — no EnKF re-run needed

## Data

Input Excel files in `data/`:
- `CHO_T127_*` — Cell Line A (T127 strain): shake flask + bioreactor at 36.5°C and 32°C
- `CHO_GS46_F_*` — Cell Line B (GS46 strain): different feed strategies
- Each file has sheets: `schedule`, `feed`, `exp_meas`

## Gotchas

- `results/` is gitignored — pkl files and figures do not sync via git
- `original.ipynb` is the legacy monolithic notebook; the `cho_enkf` package is canonical
- No test suite — research project
- Scripts must be run from the project root (they add the root to `sys.path`)
