"""
01_ensemble_tuning.py
=====================
Step 1: Load data, run nominal simulation, sweep ensemble sizes.

Run from project root:
    poetry run python scripts/01_ensemble_tuning.py

Outputs saved to results/{RUN_NAME}/pkl/
"""

import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cho_enkf.config import (
    DATA_DIR, DATASET_FILES, INITIAL_VOLUMES, STATE_NAMES,
    MEAN_PARAMETERS, PARAMETERS_ENSEMBLE_COVARIANCE,
    DATASET_NOISE_VARIANCES, KQ_DICT, KR_DICT,
    TUNING_ENSEMBLE_SIZES,
)
from cho_enkf.data_loader import load_datasets
from cho_enkf.model import compute_volume_results, simulate_all_datasets
from cho_enkf.enkf import run_enkf_with_tuning
from cho_enkf.io_utils import ensure_dirs, save_pkl

# ── Setup ─────────────────────────────────────────────────────────────────────
ensure_dirs()

print("=" * 60)
print("Step 1: Ensemble Tuning")
print("=" * 60)

# ── Load data ─────────────────────────────────────────────────────────────────
datasets = load_datasets(DATA_DIR, DATASET_FILES)

# ── Volume integration ────────────────────────────────────────────────────────
print("\nComputing volume profiles ...")
volume_results = compute_volume_results(datasets, INITIAL_VOLUMES)
save_pkl(volume_results, 'volume_results.pkl')

# ── Nominal simulation (Kotidis 2019 parameters) ──────────────────────────────
print("\nRunning nominal model simulation ...")
simulation_results = simulate_all_datasets(datasets, volume_results, MEAN_PARAMETERS)
save_pkl(simulation_results, 'simulation_results.pkl')

# ── EnKF tuning sweep ─────────────────────────────────────────────────────────
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

# ── Save results ──────────────────────────────────────────────────────────────
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

print("\nStep 1 complete.")
