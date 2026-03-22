"""
05_comparisons.py
=================
Step 5: Compare EnKF (full data, best ensemble size) against an open-loop
simulation with parameters reparametrised specifically for CHO_GS46_F_C_Inv.

Addresses Reviewer 2, point 5:
  "The manuscript positions EnKF favourably relative to other estimators and
   to conventional reparametrisation, but does not provide quantitative
   comparisons."

Prerequisites
-------------
- results/{RUN_NAME}/pkl/ensemble_tuning.pkl     (from script 01)
- results/{RUN_NAME}/pkl/simulation_results.pkl  (from script 01)
- results/{RUN_NAME}/pkl/volume_results.pkl      (from script 01)
- cho_enkf/config.py: CHO_GS46_F_C_Inv_PARAMETERS fully filled in

Run from project root:
    poetry run python scripts/06_comparisons.py
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cho_enkf.config import (
    RESULTS_DIR,
    DATA_DIR, DATASET_FILES, STATE_NAMES, AXIS_NAMES,
    BEST_ENSEMBLE_SIZES, DATASET_COLOURS, DATASET_MARKERS,
    CUSTOM_TITLES, MEAN_PARAMETERS,
    CHO_GS46_F_C_Inv_PARAMETERS,
)
from cho_enkf.data_loader import load_datasets
from cho_enkf.model import simulate_all_datasets
from cho_enkf.analysis import compute_r2
from cho_enkf.io_utils import set_dirs, ensure_dirs, load_pkl, fig_path
from cho_enkf import plotting as pl

S01_PKL = RESULTS_DIR / "01_ensemble_tuning" / "pkl"
S05_PKL = RESULTS_DIR / "05_comparisons" / "pkl"
S05_FIG = RESULTS_DIR / "05_comparisons" / "figures"
set_dirs(S05_PKL, S05_FIG)
ensure_dirs()

DS = 'CHO_GS46_F_C_Inv'
BEST_N = BEST_ENSEMBLE_SIZES[DS]   # 75

print("=" * 60)
print("Step 5: EnKF vs Reparametrised Model Comparison")
print(f"  Dataset : {DS}")
print(f"  Ensemble: {BEST_N}")
print("=" * 60)

# ── Load data and pre-computed results ────────────────────────────────────────
datasets           = load_datasets(DATA_DIR, DATASET_FILES)
volume_results     = load_pkl('volume_results.pkl',     subdir=S01_PKL)
simulation_results = load_pkl('simulation_results.pkl', subdir=S01_PKL)
ensemble_tuning    = load_pkl('ensemble_tuning.pkl',    subdir=S01_PKL)

# ── EnKF trajectory (full data, best ensemble) ───────────────────────────────
enkf_traj = ensemble_tuning[DS][BEST_N]          # (n_steps, n_states)

# ── Nominal (Kotidis) simulation ─────────────────────────────────────────────
nominal_sim = simulation_results[DS]['full_simulation']   # (n_steps, n_states)

# ── Reparametrised open-loop simulation ──────────────────────────────────────
print("\nRunning reparametrised simulation ...")
reparam_results = simulate_all_datasets(
    {DS: datasets[DS]},
    {DS: volume_results[DS]},
    CHO_GS46_F_C_Inv_PARAMETERS,
)
reparam_sim = reparam_results[DS]['full_simulation']      # (n_steps, n_states)

# ── RMSE comparison ──────────────────────────────────────────────────────────
exp_meas = datasets[DS]['exp_meas']
exp_vals = exp_meas.iloc[:, 1:9].values          # (n_obs, n_states)
dt_kf    = 0.01

time_hours = exp_meas['Time (hours)'].values
time_indices = [round(t / dt_kf) for t in time_hours]
time_indices = [min(idx, enkf_traj.shape[0] - 1) for idx in time_indices]

enkf_at_obs    = enkf_traj[time_indices]
reparam_at_obs = reparam_sim[np.minimum(time_indices, reparam_sim.shape[0] - 1)]
nominal_at_obs = nominal_sim[np.minimum(time_indices, nominal_sim.shape[0] - 1)]

print("\n--- RMSE per state ---")
header = f"{'State':<8}  {'EnKF':>12}  {'Reparametrised':>16}  {'Nominal':>10}"
print(header)
print("-" * len(header))

overall_enkf, overall_reparam, overall_nominal = [], [], []
for i, sname in enumerate(STATE_NAMES):
    obs = exp_vals[:, i]
    valid = ~np.isnan(obs)
    if not valid.any():
        continue
    rmse_enkf    = float(np.sqrt(np.mean((enkf_at_obs[valid, i]    - obs[valid]) ** 2)))
    rmse_reparam = float(np.sqrt(np.mean((reparam_at_obs[valid, i] - obs[valid]) ** 2)))
    rmse_nominal = float(np.sqrt(np.mean((nominal_at_obs[valid, i] - obs[valid]) ** 2)))
    print(f"{sname:<8}  {rmse_enkf:>12.4g}  {rmse_reparam:>16.4g}  {rmse_nominal:>10.4g}")
    overall_enkf.append(rmse_enkf)
    overall_reparam.append(rmse_reparam)
    overall_nominal.append(rmse_nominal)

print("-" * len(header))
print(f"{'Mean':<8}  {np.mean(overall_enkf):>12.4g}  "
      f"{np.mean(overall_reparam):>16.4g}  {np.mean(overall_nominal):>10.4g}")

# ── Figure ────────────────────────────────────────────────────────────────────
print("\nGenerating comparison figure ...")
pl.plot_enkf_vs_reparametrised(
    dataset=datasets[DS],
    enkf_traj=enkf_traj,
    reparam_sim=reparam_sim,
    nominal_sim=nominal_sim,
    state_names=STATE_NAMES,
    axis_names=AXIS_NAMES,
    dataset_colour=DATASET_COLOURS[DS],
    dataset_marker=DATASET_MARKERS[DS],
    custom_title=CUSTOM_TITLES[DS],
    save_path=fig_path("enkf_vs_reparametrised.png"),
)

print("\nStep 5 complete. Figure saved to results/{RUN_NAME}/figures/")
