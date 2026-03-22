"""
04_sensitivity.py
=================
Step 4: Two sensitivity analyses
  (a) Prior width scaling (0.5×, 1.0×, 1.5×, 2.0×)
  (b) ±10%/±20%/±30% perturbation of nominal mean parameters

Requires Step 1 outputs in results/01_ensemble_tuning/pkl/.

Run from project root:
    poetry run python scripts/04_sensitivity.py

Outputs saved to results/04_sensitivity/pkl/ and results/04_sensitivity/figures/
"""

import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cho_enkf.config import (
    RESULTS_DIR,
    DATA_DIR, DATASET_FILES, STATE_NAMES, AXIS_NAMES,
    MEAN_PARAMETERS, PARAMETERS_ENSEMBLE_COVARIANCE,
    DATASET_NOISE_VARIANCES, KQ_DICT, KR_DICT,
    BEST_ENSEMBLE_SIZES, PRIOR_WIDTH_SCALES, PARAM_SENS_PERTURBATIONS,
    DATASET_COLOURS, DATASET_MARKERS, CUSTOM_TITLES,
)
from cho_enkf.data_loader import load_datasets
from cho_enkf.enkf import run_enkf_with_tuning, run_enkf_with_mean_params
from cho_enkf.io_utils import set_dirs, ensure_dirs, has_results, save_pkl, load_pkl, fig_path
from cho_enkf import plotting as pl

S01_PKL = RESULTS_DIR / "01_ensemble_tuning" / "pkl"
S04_PKL = RESULTS_DIR / "04_sensitivity" / "pkl"
S04_FIG = RESULTS_DIR / "04_sensitivity" / "figures"
set_dirs(S04_PKL, S04_FIG)
ensure_dirs()

print("=" * 60)
print("Step 4: Sensitivity Analyses")
print("=" * 60)

datasets       = load_datasets(DATA_DIR, DATASET_FILES)
volume_results = load_pkl('volume_results.pkl', subdir=S01_PKL)

LOAD_FROM_PKL = has_results()
print(f"\n{'Loading from pkl' if LOAD_FROM_PKL else 'No existing results — running from scratch'} ...")

# ── (a) Prior width sensitivity ───────────────────────────────────────────────
print("\n--- (a) Prior width sensitivity ---")

if LOAD_FROM_PKL:
    print("Loading saved prior width results from pkl ...")
    prior_width_rmse = load_pkl('prior_width_rmse.pkl')
    prior_width_sim  = load_pkl('prior_width_sim.pkl')
else:
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

# ── Figure H: Prior width sensitivity RMSE bar chart ──────────────────────────
print("\n[H] Prior width sensitivity RMSE ...")
pl.plot_prior_width_sensitivity_rmse(
    prior_width_rmse, PRIOR_WIDTH_SCALES, BEST_ENSEMBLE_SIZES,
    save_path=fig_path("prior_width_sensitivity_rmse.png"),
    custom_titles=CUSTOM_TITLES,
)

# ── Figure I: Prior width state profiles (one per dataset) ────────────────────
print("[I] Prior width state profiles ...")
scale_colors = dict(zip(PRIOR_WIDTH_SCALES,
                        ["#4C72B0", "#55A868", "#C44E52", "#DD8452"]))
scale_labels = {s: f"{s}× prior width" for s in PRIOR_WIDTH_SCALES}
pl.plot_prior_width_state_profiles(
    prior_width_sim, PRIOR_WIDTH_SCALES, BEST_ENSEMBLE_SIZES,
    datasets, STATE_NAMES, AXIS_NAMES,
    scale_colors=scale_colors, scale_labels=scale_labels,
    custom_titles=CUSTOM_TITLES,
    save_dir=S04_FIG,
)

# ── (b) ±% parameter sensitivity ─────────────────────────────────────────────
print(f"\n--- (b) mean parameter sensitivity: {[f'±{int(p*100)}%' for p in PARAM_SENS_PERTURBATIONS]} ---")

if LOAD_FROM_PKL:
    print("Loading saved parameter sensitivity results from pkl ...")
    sim_baseline  = load_pkl('sim_baseline.pkl')
    para_baseline = load_pkl('para_baseline.pkl')
    perturb_sims  = {}
    for p in PARAM_SENS_PERTURBATIONS:
        tag = f"{int(p * 100)}"
        perturb_sims[p] = {
            'plus':  load_pkl(f'sim_plus{tag}.pkl'),
            'minus': load_pkl(f'sim_minus{tag}.pkl'),
        }
else:
    mean_params_baseline = MEAN_PARAMETERS.copy()

    print("  Running baseline ...")
    sim_baseline, para_baseline = run_enkf_with_mean_params(
        mean_params_baseline, datasets, DATASET_NOISE_VARIANCES, volume_results,
        BEST_ENSEMBLE_SIZES, PARAMETERS_ENSEMBLE_COVARIANCE, KQ_DICT, KR_DICT,
    )
    save_pkl(sim_baseline,         'sim_baseline.pkl')
    save_pkl(para_baseline,        'para_baseline.pkl')
    save_pkl(mean_params_baseline, 'mean_params_baseline.pkl')

    perturb_sims = {}
    for p in PARAM_SENS_PERTURBATIONS:
        tag = f"{int(p * 100)}"
        mean_params_plus  = {k: v * (1 + p) for k, v in MEAN_PARAMETERS.items()}
        mean_params_minus = {k: v * (1 - p) for k, v in MEAN_PARAMETERS.items()}

        print(f"  Running +{tag}% ...")
        sim_plus, para_plus = run_enkf_with_mean_params(
            mean_params_plus, datasets, DATASET_NOISE_VARIANCES, volume_results,
            BEST_ENSEMBLE_SIZES, PARAMETERS_ENSEMBLE_COVARIANCE, KQ_DICT, KR_DICT,
        )

        print(f"  Running -{tag}% ...")
        sim_minus, para_minus = run_enkf_with_mean_params(
            mean_params_minus, datasets, DATASET_NOISE_VARIANCES, volume_results,
            BEST_ENSEMBLE_SIZES, PARAMETERS_ENSEMBLE_COVARIANCE, KQ_DICT, KR_DICT,
        )

        save_pkl(sim_plus,          f'sim_plus{tag}.pkl')
        save_pkl(sim_minus,         f'sim_minus{tag}.pkl')
        save_pkl(para_plus,         f'para_plus{tag}.pkl')
        save_pkl(para_minus,        f'para_minus{tag}.pkl')
        save_pkl(mean_params_plus,  f'mean_params_plus{tag}.pkl')
        save_pkl(mean_params_minus, f'mean_params_minus{tag}.pkl')

        perturb_sims[p] = {'plus': sim_plus, 'minus': sim_minus}

    save_pkl(PARAM_SENS_PERTURBATIONS, 'param_sens_perturbations.pkl')
    print("Parameter sensitivity results saved.")

# ── Figure J: ±% parameter sensitivity (one per dataset) ──────────────────────
print("\n[J] Parameter sensitivity ...")
pl.plot_param_sensitivity_comparison(
    datasets, perturb_sims, sim_baseline, PARAM_SENS_PERTURBATIONS,
    STATE_NAMES, AXIS_NAMES, BEST_ENSEMBLE_SIZES,
    DATASET_COLOURS, DATASET_MARKERS,
    save_dir=S04_FIG,
)

print("\nStep 4 complete.")
