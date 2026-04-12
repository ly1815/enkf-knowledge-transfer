"""
06_comparisons.py
=================
Step 6: Compare EnKF (full data, best ensemble size) against an open-loop
simulation with parameters reparametrised specifically for CHO_GS46_F_C_Inv.

Addresses Reviewer 2, point 5:
  "The manuscript positions EnKF favourably relative to other estimators and
   to conventional reparametrisation, but does not provide quantitative
   comparisons."

Standalone — no dependency on other scripts' pkl files.

Run from project root:
    poetry run python scripts/06_comparisons.py
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cho_enkf.config import (
    RESULTS_DIR,
    DATA_DIR, DATASET_FILES, INITIAL_VOLUMES, STATE_NAMES, AXIS_NAMES,
    MEAN_PARAMETERS, PARAMETERS_ENSEMBLE_COVARIANCE,
    DATASET_NOISE_VARIANCES, KQ_DICT, KR_DICT,
    BEST_ENSEMBLE_SIZES, DATASET_COLOURS, DATASET_MARKERS,
    CUSTOM_TITLES, CHO_GS46_F_C_Inv_PARAMETERS,
)
from cho_enkf.data_loader import load_datasets
from cho_enkf.model import compute_volume_results, simulate_all_datasets
from cho_enkf.enkf import run_enkf_with_tuning
from cho_enkf.io_utils import set_dirs, ensure_dirs, has_results, save_pkl, load_pkl, fig_path
from cho_enkf import plotting as pl

S06_PKL = RESULTS_DIR / "06_comparisons" / "pkl"
S06_FIG = RESULTS_DIR / "06_comparisons" / "figures"
set_dirs(S06_PKL, S06_FIG)
ensure_dirs()

DS     = 'CHO_GS46_F_C_Inv'
BEST_N = BEST_ENSEMBLE_SIZES[DS]   # 75

print("=" * 60)
print("Step 6: EnKF vs Reparametrised Model Comparison")
print(f"  Dataset : {DS}")
print(f"  Ensemble: {BEST_N}")
print("=" * 60)

datasets = load_datasets(DATA_DIR, DATASET_FILES)

LOAD_FROM_PKL = has_results()
print(f"\n{'Loading from pkl' if LOAD_FROM_PKL else 'No existing results — running from scratch'} ...")

if LOAD_FROM_PKL:
    enkf_traj   = load_pkl('enkf_traj.pkl')
    nominal_sim = load_pkl('nominal_sim.pkl')
    reparam_sim = load_pkl('reparam_sim.pkl')
else:
    # ── Volume integration ────────────────────────────────────────────────────
    print("\nComputing volume profile ...")
    volume_results = compute_volume_results(
        {DS: datasets[DS]}, {DS: INITIAL_VOLUMES[DS]}
    )
    save_pkl(volume_results, 'volume_results.pkl')

    # ── Nominal (Kotidis) simulation ──────────────────────────────────────────
    print("Running nominal simulation ...")
    nominal_results = simulate_all_datasets(
        {DS: datasets[DS]}, {DS: volume_results[DS]}, MEAN_PARAMETERS
    )
    nominal_sim = nominal_results[DS]['full_simulation']
    save_pkl(nominal_sim, 'nominal_sim.pkl')

    # ── EnKF (full data, best ensemble size) ──────────────────────────────────
    print(f"Running EnKF (ensemble size {BEST_N}) ...")
    (enkf_tuning, _, _, _, _, _, _, _, _) = run_enkf_with_tuning(
        {DS: datasets[DS]},
        {DS: DATASET_NOISE_VARIANCES[DS]},
        {DS: volume_results[DS]},
        [BEST_N], MEAN_PARAMETERS,
        PARAMETERS_ENSEMBLE_COVARIANCE,
        {DS: KQ_DICT[DS]}, {DS: KR_DICT[DS]},
    )
    enkf_traj = enkf_tuning[DS][BEST_N]
    save_pkl(enkf_traj, 'enkf_traj.pkl')

    # ── Reparametrised open-loop simulation ───────────────────────────────────
    print("Running reparametrised simulation ...")
    reparam_results = simulate_all_datasets(
        {DS: datasets[DS]}, {DS: volume_results[DS]}, CHO_GS46_F_C_Inv_PARAMETERS
    )
    reparam_sim = reparam_results[DS]['full_simulation']
    save_pkl(reparam_sim, 'reparam_sim.pkl')

# ── RMSE comparison ───────────────────────────────────────────────────────────
exp_meas = datasets[DS]['exp_meas']
exp_vals = exp_meas.iloc[:, 1:9].values
dt_kf    = 0.01

time_hours   = exp_meas['Time (hours)'].values
time_indices = [round(t / dt_kf) for t in time_hours]
time_indices = [min(idx, enkf_traj.shape[0] - 1) for idx in time_indices]

enkf_at_obs    = enkf_traj[time_indices]
reparam_at_obs = reparam_sim[np.minimum(time_indices, reparam_sim.shape[0] - 1)]
nominal_at_obs = nominal_sim[np.minimum(time_indices, nominal_sim.shape[0] - 1)]

def _r2(pred, obs):
    ss_res = np.sum((pred - obs) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')

print("\n--- RMSE per state ---")
header = f"{'State':<8}  {'EnKF':>12}  {'Reparametrised':>16}  {'Nominal':>10}"
print(header)
print("-" * len(header))

overall_enkf, overall_reparam, overall_nominal = [], [], []
for i, sname in enumerate(STATE_NAMES):
    obs   = exp_vals[:, i]
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

print("\n--- R² per state ---")
header_r2 = f"{'State':<8}  {'EnKF':>12}  {'Reparametrised':>16}  {'Nominal':>10}"
print(header_r2)
print("-" * len(header_r2))

r2_enkf_all, r2_reparam_all, r2_nominal_all = [], [], []
for i, sname in enumerate(STATE_NAMES):
    obs   = exp_vals[:, i]
    valid = ~np.isnan(obs)
    if not valid.any():
        continue
    r2_enkf    = _r2(enkf_at_obs[valid, i],    obs[valid])
    r2_reparam = _r2(reparam_at_obs[valid, i],  obs[valid])
    r2_nominal = _r2(nominal_at_obs[valid, i],  obs[valid])
    print(f"{sname:<8}  {r2_enkf:>12.4f}  {r2_reparam:>16.4f}  {r2_nominal:>10.4f}")
    r2_enkf_all.append(r2_enkf)
    r2_reparam_all.append(r2_reparam)
    r2_nominal_all.append(r2_nominal)

print("-" * len(header_r2))
print(f"{'Mean':<8}  {np.mean(r2_enkf_all):>12.4f}  "
      f"{np.mean(r2_reparam_all):>16.4f}  {np.mean(r2_nominal_all):>10.4f}")

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

print(f"\nStep 6 complete.")
