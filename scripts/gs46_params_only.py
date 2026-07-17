"""
gs46_params_only.py
===================
Minimal, fast extraction of the EnKF converged parameter estimates for the
three GS46 (Cell Line B) datasets at ensemble size 75 only.

Skips: the ensemble-size sweep (10/25/50), the T127 datasets, all figures,
and long-term prediction. Just runs the EnKF fit and writes the parameter
deliverable (Excel + console tables).

Run from project root:
    poetry run python scripts/gs46_params_only.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cho_enkf.config import (
    DATA_DIR, INITIAL_VOLUMES,
    MEAN_PARAMETERS, CHO_GS46_F_C_Inv_PARAMETERS,
    PARAMETERS_ENSEMBLE_COVARIANCE,
    DATASET_NOISE_VARIANCES, KQ_DICT, KR_DICT,
    PARAMETER_KEYS, STATE_NAMES,
)
from cho_enkf.data_loader import load_datasets
from cho_enkf.model import compute_volume_results
from cho_enkf.enkf import run_enkf_with_tuning
from cho_enkf.analysis import compute_r2_table, compute_parameter_convergence_table

# ── Scope: GS46 only, size 75 only ────────────────────────────────────────────
GS46_FILES = [
    "CHO_GS46_F_C_Inv.xlsx",
    "CHO_GS46_F_all.xlsx",
    "CHO_GS46_F_all_pl40.xlsx",
]
ENSEMBLE_SIZE = 75
SHORT_NAMES = {
    "CHO_GS46_F_C_Inv":    "B-Feed C",
    "CHO_GS46_F_all":      "B-Feed U",
    "CHO_GS46_F_all_pl40": "B-Feed U+40%",
}
OUT_XLSX = Path(__file__).resolve().parent.parent / "GS46_EnKF_parameters.xlsx"

print("=" * 60)
print("GS46 EnKF parameter extraction (size 75, no figures)")
print("=" * 60)

# ── Run the EnKF for the 3 GS46 datasets at size 75 ───────────────────────────
datasets       = load_datasets(DATA_DIR, GS46_FILES)
volume_results = compute_volume_results(datasets, INITIAL_VOLUMES)

(ensemble_tuning, mae_results, computation_times,
 PX_records_all, para_records_all,
 _Xf_para, _Xf_state, _Xpost, _Z) = run_enkf_with_tuning(
    datasets, DATASET_NOISE_VARIANCES, volume_results,
    [ENSEMBLE_SIZE], MEAN_PARAMETERS,
    PARAMETERS_ENSEMBLE_COVARIANCE, KQ_DICT, KR_DICT,
)

best_sizes = {ds: ENSEMBLE_SIZE for ds in datasets}

# ── Sheet 1: converged posterior means (+ prior + Kotidis Feed C reference) ───
means = {
    "Prior (T127, Kotidis 2019)": {k: MEAN_PARAMETERS[k] for k in PARAMETER_KEYS},
    "Kotidis 2019 reparam. Feed C": {k: CHO_GS46_F_C_Inv_PARAMETERS[k] for k in PARAMETER_KEYS},
}
for ds in datasets:
    final_ensemble = PX_records_all[ds][ENSEMBLE_SIZE][-1]  # list of member dicts
    means[SHORT_NAMES[ds]] = {
        k: float(np.mean([m[k] for m in final_ensemble])) for k in PARAMETER_KEYS
    }
df_means = pd.DataFrame(means).reindex(PARAMETER_KEYS)
df_means.index.name = "Parameter"

# ── Sheet 2: convergence (initial vs final ensemble spread) per dataset ───────
conv_cols = {}
for ds in datasets:
    tbl = compute_parameter_convergence_table(ds, ENSEMBLE_SIZE, datasets, PX_records_all)
    if tbl is not None:
        conv_cols[SHORT_NAMES[ds]] = tbl.set_index("Parameter")["Convergence (%)"]
df_conv = pd.DataFrame(conv_cols).reindex(PARAMETER_KEYS).round(1)
df_conv.index.name = "Parameter"

# ── Sheet 3+: full parameter trajectory (value at each measurement update) ────
traj_frames = {}
for ds in datasets:
    records = para_records_all[ds][ENSEMBLE_SIZE]  # [prior, after update 1, 2, ...]
    df_t = pd.DataFrame(records)[PARAMETER_KEYS]
    df_t.index.name = "Update # (0 = prior)"
    traj_frames[ds] = df_t

# ── R² fit quality ────────────────────────────────────────────────────────────
df_r2 = compute_r2_table(datasets, ensemble_tuning, best_sizes, STATE_NAMES)

# ── Write Excel workbook ──────────────────────────────────────────────────────
with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xl:
    df_means.to_excel(xl, sheet_name="Converged means")
    df_conv.to_excel(xl,  sheet_name="Convergence %")
    df_r2.to_excel(xl,    sheet_name="R2 fit quality")
    for ds, df_t in traj_frames.items():
        df_t.to_excel(xl, sheet_name=f"traj {SHORT_NAMES[ds]}"[:31])

# ── Console summary ───────────────────────────────────────────────────────────
with pd.option_context("display.width", None, "display.max_columns", None):
    print("\n--- Converged posterior means ---")
    print(df_means.to_string(float_format=lambda v: f"{v:.4g}"))
    print("\n--- Convergence (%) ---")
    print(df_conv.to_string())
    print("\n--- R2 fit quality ---")
    print(df_r2.to_string())

print(f"\nWrote deliverable: {OUT_XLSX}")
print("Done.")
