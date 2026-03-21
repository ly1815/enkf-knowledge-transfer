"""
02_longterm_pred.py
===================
Step 2: Long-term prediction using the best ensemble size per dataset.

Requires Step 1 outputs in results/{RUN_NAME}/pkl/.

Run from project root:
    poetry run python scripts/02_longterm_pred.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cho_enkf.config import (
    DATA_DIR, DATASET_FILES, INITIAL_VOLUMES,
    MEAN_PARAMETERS, PARAMETERS_ENSEMBLE_COVARIANCE,
    DATASET_NOISE_VARIANCES, KQ_DICT, KR_DICT,
    BEST_ENSEMBLE_SIZES,
)
from cho_enkf.data_loader import load_datasets
from cho_enkf.enkf import enkf_long_pred_best_ensemble_size
from cho_enkf.io_utils import ensure_dirs, save_pkl, load_pkl

ensure_dirs()

print("=" * 60)
print("Step 2: Long-term Prediction")
print("=" * 60)

datasets       = load_datasets(DATA_DIR, DATASET_FILES)
volume_results = load_pkl('volume_results.pkl')

print(f"\nBest ensemble sizes: {BEST_ENSEMBLE_SIZES}")

(PX_records_best,
 para_records_best,
 X_forecast_for_parameters_records_best,
 X_forecast_for_states_records_best,
 X_posterior_records_best,
 Z_best,
 simulation_trajectories_best,
 long_term_forecasts_best) = enkf_long_pred_best_ensemble_size(
    datasets, DATASET_NOISE_VARIANCES, volume_results,
    BEST_ENSEMBLE_SIZES, MEAN_PARAMETERS,
    PARAMETERS_ENSEMBLE_COVARIANCE, KQ_DICT, KR_DICT,
)

print("\nSaving results ...")
save_pkl(PX_records_best,                          'PX_records_best.pkl')
save_pkl(para_records_best,                        'para_records_best.pkl')
save_pkl(X_forecast_for_parameters_records_best,   'X_forecast_for_parameters_records_best.pkl')
save_pkl(X_forecast_for_states_records_best,       'X_forecast_for_states_records_best.pkl')
save_pkl(X_posterior_records_best,                 'X_posterior_records_best.pkl')
save_pkl(Z_best,                                   'Z_best.pkl')
save_pkl(simulation_trajectories_best,             'simulation_trajectories_best.pkl')
save_pkl(long_term_forecasts_best,                 'long_term_forecasts_best.pkl')

print("\nStep 2 complete.")
