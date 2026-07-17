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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cho_enkf.config import (
    DATA_DIR, INITIAL_VOLUMES,
    MEAN_PARAMETERS, CHO_GS46_F_C_Inv_PARAMETERS,
    PARAMETERS_ENSEMBLE_COVARIANCE,
    DATASET_NOISE_VARIANCES, KQ_DICT, KR_DICT,
    PARAMETER_KEYS, STATE_NAMES, AXIS_NAMES, LATEX_LABELS,
)
from cho_enkf.data_loader import load_datasets
from cho_enkf.model import compute_volume_results, simulate_all_datasets
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
FIG_DIR  = Path(__file__).resolve().parent.parent / "GS46_figures"
FIG_DIR.mkdir(exist_ok=True)
DT_MODEL = 0.01

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

# ── R²: closed-loop (EnKF filtered estimate vs obs) ───────────────────────────
# Optimistic: the filter assimilates each measurement, so states are pulled
# toward the data at every update regardless of parameter quality.
df_r2_closed = compute_r2_table(datasets, ensemble_tuning, best_sizes, STATE_NAMES)

# ── R²: open-loop (fixed converged params, forward sim, NO updates vs obs) ─────
# Honest test of whether the converged parameter set reproduces the data on its
# own — this is the regime the student's HCP-extended forward model will run in.
openloop_tuning = {}
for ds in datasets:
    conv_params = {k: float(df_means.loc[k, SHORT_NAMES[ds]]) for k in PARAMETER_KEYS}
    sim = simulate_all_datasets({ds: datasets[ds]}, volume_results, conv_params)
    openloop_tuning[ds] = {ENSEMBLE_SIZE: sim[ds]["full_simulation"]}
df_r2_open = compute_r2_table(datasets, openloop_tuning, best_sizes, STATE_NAMES)

# ── Figures ───────────────────────────────────────────────────────────────────
COL_OPEN, COL_CLOSED, COL_DATA = "#C0392B", "#2980B9", "#2C3E50"

# (1) Per-dataset state fits: data + open-loop (fixed params) + closed-loop EnKF
for ds in datasets:
    exp   = datasets[ds]["exp_meas"]
    t_exp = exp["Time (hours)"].values.astype(float)
    ol    = np.asarray(openloop_tuning[ds][ENSEMBLE_SIZE], dtype=float)
    cl    = np.asarray(ensemble_tuning[ds][ENSEMBLE_SIZE], dtype=float)
    t_mod = np.arange(ol.shape[0]) * DT_MODEL

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for j, (state, axis_name) in enumerate(zip(STATE_NAMES, AXIS_NAMES)):
        ax = axes.flat[j]
        ax.plot(t_mod, ol[:, j], "-",  color=COL_OPEN,   lw=1.8,
                label="Open-loop (fixed converged params)")
        ax.plot(t_mod, cl[:, j], "--", color=COL_CLOSED, lw=1.8,
                label="EnKF (closed-loop estimate)")
        obs = pd.to_numeric(exp[state], errors="coerce").values
        std = (pd.to_numeric(exp[f"{state}_std"], errors="coerce").values
               if f"{state}_std" in exp.columns else None)
        ax.errorbar(t_exp, obs, yerr=std, fmt="o", ms=5, color=COL_DATA,
                    ecolor="grey", capsize=3, label="Experimental data")
        ax.set_title(state, fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel(axis_name, fontsize=8)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=11,
               frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"{SHORT_NAMES[ds]} — state fits", y=1.05, fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = FIG_DIR / f"state_fit_{SHORT_NAMES[ds].replace(' ', '_').replace('+', 'plus')}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure: {out}")

# (2) Parameter shift from prior (converged / prior ratio) across the 3 conditions
fig, ax = plt.subplots(figsize=(16, 6))
x = np.arange(len(PARAMETER_KEYS))
ds_list = list(datasets)
width = 0.8 / len(ds_list)
for i, ds in enumerate(ds_list):
    ratios = [df_means.loc[k, SHORT_NAMES[ds]] / MEAN_PARAMETERS[k] for k in PARAMETER_KEYS]
    ax.bar(x + i * width, ratios, width, label=SHORT_NAMES[ds])
ax.axhline(1.0, color="k", lw=1, ls=":")
ax.set_yscale("log")
ax.set_xticks(x + width * (len(ds_list) - 1) / 2)
ax.set_xticklabels([LATEX_LABELS[k] for k in PARAMETER_KEYS], rotation=90)
ax.set_ylabel("Converged / prior  (log scale; 1 = unchanged)")
ax.set_title("Parameter shift from prior — knowledge transfer per feed condition",
             fontsize=13, fontweight="bold")
ax.legend()
fig.tight_layout()
out = FIG_DIR / "parameter_shift_from_prior.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Figure: {out}")

# (3) Parameter identifiability: convergence % per parameter across conditions
fig, ax = plt.subplots(figsize=(16, 6))
for i, ds in enumerate(ds_list):
    ax.bar(x + i * width, df_conv[SHORT_NAMES[ds]].values, width, label=SHORT_NAMES[ds])
ax.set_xticks(x + width * (len(ds_list) - 1) / 2)
ax.set_xticklabels([LATEX_LABELS[k] for k in PARAMETER_KEYS], rotation=90)
ax.set_ylabel("Convergence (%)  — ensemble spread reduction")
ax.set_title("Parameter identifiability (higher = better informed by data; low ≈ still at prior)",
             fontsize=13, fontweight="bold")
ax.legend()
fig.tight_layout()
out = FIG_DIR / "parameter_convergence.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Figure: {out}")

# ── Write Excel workbook ──────────────────────────────────────────────────────
with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xl:
    df_means.to_excel(xl,     sheet_name="Converged means")
    df_conv.to_excel(xl,      sheet_name="Convergence %")
    df_r2_open.to_excel(xl,   sheet_name="R2 open-loop (fixed params)")
    df_r2_closed.to_excel(xl, sheet_name="R2 closed-loop (EnKF)")
    for ds, df_t in traj_frames.items():
        df_t.to_excel(xl, sheet_name=f"traj {SHORT_NAMES[ds]}"[:31])

# ── Console summary ───────────────────────────────────────────────────────────
with pd.option_context("display.width", None, "display.max_columns", None):
    print("\n--- Converged posterior means ---")
    print(df_means.to_string(float_format=lambda v: f"{v:.4g}"))
    print("\n--- Convergence (%) ---")
    print(df_conv.to_string())
    print("\n--- R2 OPEN-LOOP (fixed converged params, no updates vs obs) ---")
    print(df_r2_open.to_string())
    print("\n--- R2 CLOSED-LOOP (EnKF filtered estimate vs obs; optimistic) ---")
    print(df_r2_closed.to_string())

print(f"\nWrote deliverable: {OUT_XLSX}")
print(f"Wrote figures to:  {FIG_DIR}")
print("Done.")
