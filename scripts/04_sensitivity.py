"""
04_sensitivity.py
=================
Step 4: Two sensitivity analyses
  (a) Prior width scaling (0.5×, 1.0×, 1.5×, 2.0×)
  (b) ±20% perturbation of nominal mean parameters

Requires Step 1 outputs in results/{RUN_NAME}/pkl/.

Run from project root:
    poetry run python scripts/04_sensitivity.py
"""

import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cho_enkf.config import (
    DATA_DIR, DATASET_FILES,
    MEAN_PARAMETERS, PARAMETERS_ENSEMBLE_COVARIANCE,
    DATASET_NOISE_VARIANCES, KQ_DICT, KR_DICT,
    BEST_ENSEMBLE_SIZES, PRIOR_WIDTH_SCALES, PARAM_SENS_PERTURBATION,
)
from cho_enkf.data_loader import load_datasets
from cho_enkf.enkf import run_enkf_with_tuning, run_enkf_with_mean_params
from cho_enkf.io_utils import ensure_dirs, save_pkl, load_pkl, save_sens_pkl

ensure_dirs()

print("=" * 60)
print("Step 4: Sensitivity Analyses")
print("=" * 60)

datasets       = load_datasets(DATA_DIR, DATASET_FILES)
volume_results = load_pkl('volume_results.pkl')

# ── (a) Prior width sensitivity ───────────────────────────────────────────────
print("\n--- (a) Prior width sensitivity ---")
size_to_datasets = defaultdict(dict)
for ds_name, best_n in BEST_ENSEMBLE_SIZES.items():
    size_to_datasets[best_n][ds_name] = datasets[ds_name]

prior_width_rmse = {}
prior_width_sim  = {}

for scale in PRIOR_WIDTH_SCALES:
    print(f"\nScale: {scale}×")
    scaled_cov = {k: v * scale for k, v in PARAMETERS_ENSEMBLE_COVARIANCE.items()}
    prior_width_rmse[scale] = {}
    prior_width_sim[scale]  = {}

    for best_n, ds_subset in size_to_datasets.items():
        print(f"  Ensemble size {best_n}: {list(ds_subset.keys())}")
        (sens_tuning, sens_rmse, _, _, _, _, _, _, _) = run_enkf_with_tuning(
            ds_subset, DATASET_NOISE_VARIANCES, volume_results,
            [best_n], MEAN_PARAMETERS, scaled_cov, KQ_DICT, KR_DICT,
        )
        for ds_name in ds_subset:
            prior_width_rmse[scale][ds_name] = sens_rmse[ds_name][best_n]
            prior_width_sim[scale][ds_name]  = sens_tuning[ds_name][best_n]

save_pkl(prior_width_rmse,   'prior_width_rmse.pkl')
save_pkl(prior_width_sim,    'prior_width_sim.pkl')
save_pkl(PRIOR_WIDTH_SCALES, 'prior_width_scales.pkl')
save_pkl(BEST_ENSEMBLE_SIZES,'best_ensemble_sizes_prior.pkl')
print("Prior width results saved.")

# ── (b) ±20% parameter sensitivity ───────────────────────────────────────────
print("\n--- (b) ±20% mean parameter sensitivity ---")
p = PARAM_SENS_PERTURBATION
mean_params_baseline = MEAN_PARAMETERS.copy()
mean_params_plus20   = {k: v * (1 + p) for k, v in MEAN_PARAMETERS.items()}
mean_params_minus20  = {k: v * (1 - p) for k, v in MEAN_PARAMETERS.items()}

print("  Running baseline ...")
sim_baseline, para_baseline = run_enkf_with_mean_params(
    mean_params_baseline, datasets, DATASET_NOISE_VARIANCES, volume_results,
    BEST_ENSEMBLE_SIZES, PARAMETERS_ENSEMBLE_COVARIANCE, KQ_DICT, KR_DICT,
)

print(f"  Running +{p*100:.0f}% ...")
sim_plus20, para_plus20 = run_enkf_with_mean_params(
    mean_params_plus20, datasets, DATASET_NOISE_VARIANCES, volume_results,
    BEST_ENSEMBLE_SIZES, PARAMETERS_ENSEMBLE_COVARIANCE, KQ_DICT, KR_DICT,
)

print(f"  Running -{p*100:.0f}% ...")
sim_minus20, para_minus20 = run_enkf_with_mean_params(
    mean_params_minus20, datasets, DATASET_NOISE_VARIANCES, volume_results,
    BEST_ENSEMBLE_SIZES, PARAMETERS_ENSEMBLE_COVARIANCE, KQ_DICT, KR_DICT,
)

save_sens_pkl(sim_baseline,          'sim_baseline.pkl')
save_sens_pkl(sim_plus20,            'sim_plus20.pkl')
save_sens_pkl(sim_minus20,           'sim_minus20.pkl')
save_sens_pkl(para_baseline,         'para_baseline.pkl')
save_sens_pkl(para_plus20,           'para_plus20.pkl')
save_sens_pkl(para_minus20,          'para_minus20.pkl')
save_sens_pkl(mean_params_baseline,  'mean_params_baseline.pkl')
save_sens_pkl(mean_params_plus20,    'mean_params_plus20.pkl')
save_sens_pkl(mean_params_minus20,   'mean_params_minus20.pkl')
print("Parameter sensitivity results saved.")

print("\nStep 4 complete.")
