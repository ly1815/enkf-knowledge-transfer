"""
03_irregular.py
===============
Step 3: EnKF with an irregular (48/72 h alternating) measurement schedule.

Requires Step 1 outputs in results/{RUN_NAME}/pkl/.

Run from project root:
    poetry run python scripts/03_irregular.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cho_enkf.config import (
    DATA_DIR, DATASET_FILES,
    MEAN_PARAMETERS, PARAMETERS_ENSEMBLE_COVARIANCE,
    DATASET_NOISE_VARIANCES, KQ_DICT, KR_DICT,
    BEST_ENSEMBLE_SIZES, IRREGULAR_PATTERN_HOURS,
)
from cho_enkf.data_loader import load_datasets
from cho_enkf.enkf import run_pipeline_irregular_48_72
from cho_enkf.io_utils import ensure_dirs, save_pkl, load_pkl

ensure_dirs()

print("=" * 60)
print("Step 3: Irregular Measurements (48/72 h pattern)")
print("=" * 60)

datasets       = load_datasets(DATA_DIR, DATASET_FILES)
volume_results = load_pkl('volume_results.pkl')

datasets_irregular, results_irregular = run_pipeline_irregular_48_72(
    datasets, DATASET_NOISE_VARIANCES, volume_results,
    BEST_ENSEMBLE_SIZES, MEAN_PARAMETERS,
    PARAMETERS_ENSEMBLE_COVARIANCE, KQ_DICT, KR_DICT,
    pattern_hours=IRREGULAR_PATTERN_HOURS,
)

(PX_records_irregular,
 para_records_irregular,
 Xf_para_irregular,
 Xf_state_irregular,
 Xpost_irregular,
 Z_irregular,
 sim_irregular,
 runtime_irregular,
 update_times_irregular) = results_irregular

print("\nSaving results ...")
save_pkl(PX_records_irregular,      'PX_records_irregular.pkl')
save_pkl(para_records_irregular,    'para_records_irregular.pkl')
save_pkl(Xf_para_irregular,         'Xf_para_irregular.pkl')
save_pkl(Xf_state_irregular,        'Xf_state_irregular.pkl')
save_pkl(Xpost_irregular,           'Xpost_irregular.pkl')
save_pkl(Z_irregular,               'Z_irregular.pkl')
save_pkl(sim_irregular,             'sim_irregular.pkl')
save_pkl(runtime_irregular,         'runtime_irregular.pkl')
save_pkl(update_times_irregular,    'update_times_irregular.pkl')
save_pkl(datasets_irregular,        'datasets_irregular.pkl')
save_pkl(BEST_ENSEMBLE_SIZES,       'dataset_ensemble_sizes_irregular.pkl')

print("\nStep 3 complete.")
