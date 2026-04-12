"""
05_priormean_sensitivity.py
===========================
Step 5: Prior mean parameter sensitivity analysis
  Perturbs all nominal mean parameters by ±10%, ±20% and compares
  the effect on filter performance.

Requires Step 1 outputs in results/01_ensemble_tuning/pkl/.

Run from project root:
    poetry run python scripts/05_priormean_sensitivity.py

Outputs saved to results/05_priormean_sensitivity/pkl/ and results/05_priormean_sensitivity/figures/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cho_enkf.config import (
    RESULTS_DIR,
    DATA_DIR, DATASET_FILES, STATE_NAMES, AXIS_NAMES,
    MEAN_PARAMETERS, PARAMETERS_ENSEMBLE_COVARIANCE,
    DATASET_NOISE_VARIANCES, KQ_DICT, KR_DICT,
    BEST_ENSEMBLE_SIZES, PARAM_SENS_PERTURBATIONS,
    DATASET_COLOURS, DATASET_MARKERS,
)
from cho_enkf.data_loader import load_datasets
from cho_enkf.enkf import run_enkf_with_mean_params
from cho_enkf.io_utils import set_dirs, ensure_dirs, has_results, save_pkl, load_pkl, fig_path
from cho_enkf import plotting as pl

S01_PKL = RESULTS_DIR / "01_ensemble_tuning" / "pkl"
S05_PKL = RESULTS_DIR / "05_priormean_sensitivity" / "pkl"
S05_FIG = RESULTS_DIR / "05_priormean_sensitivity" / "figures"
set_dirs(S05_PKL, S05_FIG)
ensure_dirs()

print("=" * 60)
print("Step 5: Prior Mean Parameter Sensitivity")
print("=" * 60)

datasets       = load_datasets(DATA_DIR, DATASET_FILES)
volume_results = load_pkl('volume_results.pkl', subdir=S01_PKL)

LOAD_FROM_PKL = has_results()
print(f"\n{'Loading from pkl' if LOAD_FROM_PKL else 'No existing results — running from scratch'} ...")

# ── ±% parameter sensitivity ──────────────────────────────────────────────────
print(f"\n--- Mean parameter sensitivity: {[f'±{int(p*100)}%' for p in PARAM_SENS_PERTURBATIONS]} ---")

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
ylim_overrides = {
    'CHO_GS46_F_all':   ['Asn'],
    'CHO_GS46_F_C_Inv': ['Asn'],
}
pl.plot_param_sensitivity_comparison(
    datasets, perturb_sims, sim_baseline, PARAM_SENS_PERTURBATIONS,
    STATE_NAMES, AXIS_NAMES, BEST_ENSEMBLE_SIZES,
    DATASET_COLOURS, DATASET_MARKERS,
    save_dir=S05_FIG,
    ylim_scale_overrides=ylim_overrides,
)

print("\nStep 5 complete.")
