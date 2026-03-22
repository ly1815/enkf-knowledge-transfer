"""
02_longterm_pred.py
===================
Step 2: Long-term prediction using the best ensemble size per dataset,
plus parameter comparison and correlation figures.

Requires Step 1 outputs in results/{RUN_NAME}/pkl/.

Run from project root:
    poetry run python scripts/02_longterm_pred.py

Outputs saved to results/{RUN_NAME}/pkl/ and results/{RUN_NAME}/figures/
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cho_enkf.config import (
    RESULTS_DIR,
    DATA_DIR, DATASET_FILES, INITIAL_VOLUMES,
    MEAN_PARAMETERS, PARAMETERS_ENSEMBLE_COVARIANCE,
    DATASET_NOISE_VARIANCES, KQ_DICT, KR_DICT,
    BEST_ENSEMBLE_SIZES, STATE_NAMES, AXIS_NAMES,
    PARAMETER_KEYS, LATEX_LABELS,
    DATASET_COLOURS, DATASET_MARKERS, CUSTOM_TITLES, FORECAST_INDICES,
)
from cho_enkf.data_loader import load_datasets
from cho_enkf.enkf import enkf_long_pred_best_ensemble_size
from cho_enkf.analysis import (
    compute_r2_table, compute_overall_convergence_table,
    get_posterior_param_matrix,
)
from cho_enkf.io_utils import set_dirs, ensure_dirs, has_results, save_pkl, load_pkl, fig_path
from cho_enkf import plotting as pl

S01_PKL = RESULTS_DIR / "01_ensemble_tuning" / "pkl"
S02_PKL = RESULTS_DIR / "02_longterm_pred" / "pkl"
S02_FIG = RESULTS_DIR / "02_longterm_pred" / "figures"
set_dirs(S02_PKL, S02_FIG)
ensure_dirs()

print("=" * 60)
print("Step 2: Long-term Prediction")
print("=" * 60)

datasets           = load_datasets(DATA_DIR, DATASET_FILES)
volume_results     = load_pkl('volume_results.pkl',     subdir=S01_PKL)
simulation_results = load_pkl('simulation_results.pkl', subdir=S01_PKL)
ensemble_tuning    = load_pkl('ensemble_tuning.pkl',    subdir=S01_PKL)

print(f"\nBest ensemble sizes: {BEST_ENSEMBLE_SIZES}")

LOAD_FROM_PKL = has_results()
print(f"\n{'Loading from pkl' if LOAD_FROM_PKL else 'No existing results — running from scratch'} ...")

if LOAD_FROM_PKL:
    print("\nLoading saved results from pkl ...")
    PX_records_best                          = load_pkl('PX_records_best.pkl')
    para_records_best                        = load_pkl('para_records_best.pkl')
    X_forecast_for_parameters_records_best   = load_pkl('X_forecast_for_parameters_records_best.pkl')
    X_forecast_for_states_records_best       = load_pkl('X_forecast_for_states_records_best.pkl')
    X_posterior_records_best                 = load_pkl('X_posterior_records_best.pkl')
    Z_best                                   = load_pkl('Z_best.pkl')
    simulation_trajectories_best             = load_pkl('simulation_trajectories_best.pkl')
    long_term_forecasts_best                 = load_pkl('long_term_forecasts_best.pkl')
else:
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

# Merge best-size trajectories into ensemble_tuning for overlay plotters
for ds_name, best_n in BEST_ENSEMBLE_SIZES.items():
    ensemble_tuning.setdefault(ds_name, {})[best_n] = simulation_trajectories_best[ds_name]

# ── Figure D: Long-term prediction ────────────────────────────────────────────
print("\n[D] Long-term predictions ...")
pl.plot_longterm_pred_ensemble_simulation_errorbar(
    simulation_results, datasets, ensemble_tuning,
    BEST_ENSEMBLE_SIZES, long_term_forecasts_best,
    AXIS_NAMES, STATE_NAMES,
    dataset_colours=DATASET_COLOURS,
    dataset_markers=DATASET_MARKERS,
    selected_forecast_indices=FORECAST_INDICES,
    save_dir=S02_FIG,
)

# ── Figure E: Parameter comparison — T127 (scale effects) ─────────────────────
print("[E] Parameter comparison T127 (scale) ...")
T127_datasets = ["CHO_T127_flask_PMJ", "CHO_T127_SNS_36.5", "CHO_T127_SNS_32"]
T127_ens      = [BEST_ENSEMBLE_SIZES[ds] for ds in T127_datasets]
T127_legends  = ['Cell Line A Shake Flask', 'Cell Line A Bioreactor 36.5°C',
                  'Cell Line A Bioreactor 32°C']
T127_colours  = {ds: DATASET_COLOURS[ds] for ds in T127_datasets}
pl.plot_parameter_comparison_across_datasets(
    T127_datasets, T127_ens, datasets, PX_records_best, LATEX_LABELS,
    selected_keys=['mu_max', 'mu_d_max', 'm_mAb', 'Yx_gln',
                   'Yx_amm', 'Yx_lac', 'Lac_max_1', 'Lac_max_2'],
    colours=T127_colours,
    custom_legends=T127_legends,
    save_path=fig_path("para_compare_scale.png"),
)

# ── Figure F: Parameter comparison — GS46 (feed effects) ──────────────────────
print("[F] Parameter comparison GS46 (feed) ...")
GS46_datasets = ["CHO_GS46_F_C_Inv", "CHO_GS46_F_all", "CHO_GS46_F_all_pl40"]
GS46_ens      = [BEST_ENSEMBLE_SIZES[ds] for ds in GS46_datasets]
GS46_legends  = ['Cell Line B Feed C', 'Cell Line B Feed U', 'Cell Line B Feed U+40%']
GS46_colours  = {ds: DATASET_COLOURS[ds] for ds in GS46_datasets}
pl.plot_parameter_comparison_across_datasets(
    GS46_datasets, GS46_ens, datasets, PX_records_best, LATEX_LABELS,
    selected_keys=['mu_d_max', 'm_mAb', 'Yx_glc', 'Yx_asn',
                   'KIamm', 'Yx_lac', 'Lac_max_1', 'Lac_max_2'],
    colours=GS46_colours,
    custom_legends=GS46_legends,
    save_path=fig_path("para_compare_feed.png"),
)

# ── Figure G: Posterior parameter correlation ──────────────────────────────────
print("[G] Posterior parameter correlation ...")
corr_matrices = {}
for ds_name, best_n in BEST_ENSEMBLE_SIZES.items():
    mat = get_posterior_param_matrix(ds_name, best_n, PX_records_best, PARAMETER_KEYS)
    corr_matrices[ds_name] = np.corrcoef(mat.T)

dataset_titles = {
    'CHO_T127_flask_PMJ':  'Cell Line A - Shake Flask',
    'CHO_T127_SNS_36.5':   'Cell Line A - Bioreactor 36.5°C',
    'CHO_T127_SNS_32':     'Cell Line A - Bioreactor 32°C',
    'CHO_GS46_F_C_Inv':    'Cell Line B - Feed C',
    'CHO_GS46_F_all':      'Cell Line B - Feed U',
    'CHO_GS46_F_all_pl40': 'Cell Line B - Feed U+40%',
}
pl.plot_posterior_param_correlation(
    corr_matrices, BEST_ENSEMBLE_SIZES, PARAMETER_KEYS, LATEX_LABELS,
    dataset_titles=dataset_titles,
    save_path=fig_path("posterior_param_correlation.png"),
)

# ── R² summary table ──────────────────────────────────────────────────────────
print("\n--- R² Summary (best ensemble size per dataset) ---")
df_r2 = compute_r2_table(datasets, ensemble_tuning, BEST_ENSEMBLE_SIZES, STATE_NAMES)
print(df_r2.to_string())

# ── Parameter convergence table ───────────────────────────────────────────────
print("\n--- Parameter Convergence Summary ---")
dataset_info = [(ds, BEST_ENSEMBLE_SIZES[ds]) for ds in BEST_ENSEMBLE_SIZES]
conv_df = compute_overall_convergence_table(dataset_info, datasets, PX_records_best)
print(conv_df.to_string())

print("\nStep 2 complete.")
