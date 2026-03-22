"""
01_ensemble_tuning.py
=====================
Step 1: Load data, run nominal simulation, sweep ensemble sizes, plot results.

Run from project root:
    poetry run python scripts/01_ensemble_tuning.py

Outputs saved to results/{RUN_NAME}/pkl/ and results/{RUN_NAME}/figures/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cho_enkf.config import (
    RESULTS_DIR,
    DATA_DIR, DATASET_FILES, INITIAL_VOLUMES, STATE_NAMES, AXIS_NAMES,
    MEAN_PARAMETERS, PARAMETERS_ENSEMBLE_COVARIANCE,
    DATASET_NOISE_VARIANCES, KQ_DICT, KR_DICT,
    TUNING_ENSEMBLE_SIZES, BEST_ENSEMBLE_SIZES,
    DATASET_COLOURS, DATASET_MARKERS, CUSTOM_TITLES,
)
from cho_enkf.data_loader import load_datasets
from cho_enkf.model import compute_volume_results, simulate_all_datasets
from cho_enkf.enkf import run_enkf_with_tuning
from cho_enkf.io_utils import set_dirs, ensure_dirs, save_pkl, load_pkl, fig_path
from cho_enkf import plotting as pl

# ── Setup ─────────────────────────────────────────────────────────────────────
S01_PKL = RESULTS_DIR / "01_ensemble_tuning" / "pkl"
S01_FIG = RESULTS_DIR / "01_ensemble_tuning" / "figures"
set_dirs(S01_PKL, S01_FIG)
ensure_dirs()

print("=" * 60)
print("Step 1: Ensemble Tuning")
print("=" * 60)

# ── Load data ─────────────────────────────────────────────────────────────────
datasets = load_datasets(DATA_DIR, DATASET_FILES)

# ── Load saved results (skip re-running simulation) ───────────────────────────
LOAD_FROM_PKL = True  # set False to re-run from scratch

if LOAD_FROM_PKL:
    print("\nLoading saved results from pkl ...")
    volume_results                              = load_pkl('volume_results.pkl')
    simulation_results                          = load_pkl('simulation_results.pkl')
    ensemble_tuning                             = load_pkl('ensemble_tuning.pkl')
    rmse_results                                = load_pkl('rmse_results.pkl')
    computation_times                           = load_pkl('computation_times.pkl')
    PX_records_all                              = load_pkl('PX_records_all.pkl')
    para_records_all                            = load_pkl('para_records_all.pkl')
    X_forecast_for_parameters_records_all       = load_pkl('X_forecast_for_parameters_records_all.pkl')
    X_forecast_for_states_records_all           = load_pkl('X_forecast_for_states_records_all.pkl')
    X_posterior_records_all                     = load_pkl('X_posterior_records_all.pkl')
    Z_all                                       = load_pkl('Z_all.pkl')
else:
    # ── Volume integration ────────────────────────────────────────────────────
    print("\nComputing volume profiles ...")
    volume_results = compute_volume_results(datasets, INITIAL_VOLUMES)
    save_pkl(volume_results, 'volume_results.pkl')

    # ── Nominal simulation (Kotidis 2019 parameters) ──────────────────────────
    print("\nRunning nominal model simulation ...")
    simulation_results = simulate_all_datasets(datasets, volume_results, MEAN_PARAMETERS)
    save_pkl(simulation_results, 'simulation_results.pkl')

    # ── EnKF tuning sweep ─────────────────────────────────────────────────────
    print(f"\nRunning EnKF sweep over ensemble sizes: {TUNING_ENSEMBLE_SIZES}")
    (ensemble_tuning,
     rmse_results,
     computation_times,
     PX_records_all,
     para_records_all,
     X_forecast_for_parameters_records_all,
     X_forecast_for_states_records_all,
     X_posterior_records_all,
     Z_all) = run_enkf_with_tuning(
        datasets, DATASET_NOISE_VARIANCES, volume_results,
        TUNING_ENSEMBLE_SIZES, MEAN_PARAMETERS,
        PARAMETERS_ENSEMBLE_COVARIANCE, KQ_DICT, KR_DICT,
    )

    # ── Save results ──────────────────────────────────────────────────────────
    print("\nSaving results ...")
    save_pkl(ensemble_tuning,                         'ensemble_tuning.pkl')
    save_pkl(rmse_results,                            'rmse_results.pkl')
    save_pkl(computation_times,                       'computation_times.pkl')
    save_pkl(PX_records_all,                          'PX_records_all.pkl')
    save_pkl(para_records_all,                        'para_records_all.pkl')
    save_pkl(X_forecast_for_parameters_records_all,   'X_forecast_for_parameters_records_all.pkl')
    save_pkl(X_forecast_for_states_records_all,       'X_forecast_for_states_records_all.pkl')
    save_pkl(X_posterior_records_all,                 'X_posterior_records_all.pkl')
    save_pkl(Z_all,                                   'Z_all.pkl')
    save_pkl(DATASET_NOISE_VARIANCES,                 'dataset_noise_variances.pkl')
    save_pkl(KQ_DICT,                                 'kQ_dict.pkl')
    save_pkl(KR_DICT,                                 'kR_dict.pkl')

# ── Figure A: Ensemble size tuning (RMSE + runtime) ───────────────────────────
print("\n[A] Ensemble tuning RMSE/runtime ...")
pl.plot_rmse_variance_and_computation_time_all(
    rmse_results, computation_times,
    datasets_to_include=list(BEST_ENSEMBLE_SIZES.keys()),
    custom_titles=CUSTOM_TITLES,
    save_path=fig_path("ensemble_tuning.png"),
)

# ── Figure B: T127 overlay ────────────────────────────────────────────────────
print("[B] T127 overlay ...")
pl.overlay_T127_subplots_with_errorbars(
    simulation_results, datasets, ensemble_tuning,
    BEST_ENSEMBLE_SIZES, STATE_NAMES, AXIS_NAMES,
    save_path=fig_path("T127_overlay_errorbars.png"),
)

# ── Figure C: GS46 overlay ────────────────────────────────────────────────────
print("[C] GS46 overlay ...")
pl.overlay_gs46_subplots_with_errorbars(
    simulation_results, datasets, ensemble_tuning,
    BEST_ENSEMBLE_SIZES, STATE_NAMES, AXIS_NAMES,
    save_path=fig_path("gs46_overlay_errorbars.png"),
)

print("\nStep 1 complete.")
