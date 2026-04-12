"""
04_priorcov_sensitivity.py
==========================
Step 4: Prior covariance width sensitivity analysis
  Scales the prior parameter ensemble covariance by 0.5×, 1.0×, 1.5×, 2.0×
  and compares the effect on filter performance.

Requires Step 1 outputs in results/01_ensemble_tuning/pkl/.

Run from project root:
    poetry run python scripts/04_priorcov_sensitivity.py

Outputs saved to results/04_priorcov_sensitivity/pkl/ and results/04_priorcov_sensitivity/figures/
"""

import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cho_enkf.config import (
    RESULTS_DIR,
    DATA_DIR, DATASET_FILES, STATE_NAMES, AXIS_NAMES,
    MEAN_PARAMETERS, PARAMETERS_ENSEMBLE_COVARIANCE,
    DATASET_NOISE_VARIANCES, KQ_DICT, KR_DICT,
    BEST_ENSEMBLE_SIZES, PRIOR_WIDTH_SCALES,
    CUSTOM_TITLES, DATASET_COLOURS, DATASET_MARKERS,
)
from cho_enkf.data_loader import load_datasets
from cho_enkf.enkf import run_enkf_with_tuning
from cho_enkf.io_utils import set_dirs, ensure_dirs, has_results, save_pkl, load_pkl, fig_path
from cho_enkf.analysis import detect_instability
from cho_enkf import plotting as pl

S01_PKL = RESULTS_DIR / "01_ensemble_tuning" / "pkl"
S04_PKL = RESULTS_DIR / "04_priorcov_sensitivity" / "pkl"
S04_FIG = RESULTS_DIR / "04_priorcov_sensitivity" / "figures"
set_dirs(S04_PKL, S04_FIG)
ensure_dirs()

print("=" * 60)
print("Step 4: Prior Covariance Width Sensitivity")
print("=" * 60)

datasets       = load_datasets(DATA_DIR, DATASET_FILES)
volume_results = load_pkl('volume_results.pkl', subdir=S01_PKL)

LOAD_FROM_PKL = has_results()
print(f"\n{'Loading from pkl' if LOAD_FROM_PKL else 'No existing results — running from scratch'} ...")

# ── Prior width sensitivity ────────────────────────────────────────────────────
print("\n--- Prior covariance width sensitivity ---")

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
    print("Prior covariance width results saved.")

# ── Stability detection ───────────────────────────────────────────────────────
print("\nDetecting instability per state / dataset / scale ...")
stability_flags = {}   # {scale: {ds_name: ndarray(n_states, bool)}}
for scale in PRIOR_WIDTH_SCALES:
    stability_flags[scale] = {}
    for ds_name in BEST_ENSEMBLE_SIZES:
        # 1× is the nominal reference — always stable by definition
        if scale == 1.0:
            stability_flags[scale][ds_name] = np.zeros(len(STATE_NAMES), dtype=bool)
            continue
        if ds_name not in prior_width_sim.get(scale, {}):
            stability_flags[scale][ds_name] = np.zeros(len(STATE_NAMES), dtype=bool)
            continue
        traj     = np.asarray(prior_width_sim[scale][ds_name])
        ref_traj = np.asarray(prior_width_sim[1.0][ds_name])
        exp_meas = datasets[ds_name]['exp_meas']
        stability_flags[scale][ds_name] = detect_instability(
            traj, ref_traj, exp_meas, STATE_NAMES, threshold=0.5
        )

# Report summary
for scale in PRIOR_WIDTH_SCALES:
    for ds_name, flags in stability_flags[scale].items():
        n_unstable = flags.sum()
        if n_unstable:
            bad = [STATE_NAMES[i] for i in np.where(flags)[0]]
            print(f"  [{scale}×] {ds_name}: UNSTABLE — {bad}")
        else:
            print(f"  [{scale}×] {ds_name}: stable")

save_pkl(stability_flags, 'stability_flags.pkl')

# ── Stability heatmap ─────────────────────────────────────────────────────────
print("\n[H] Stability heatmap ...")
pl.plot_stability_heatmap(
    stability_flags, PRIOR_WIDTH_SCALES, BEST_ENSEMBLE_SIZES,
    STATE_NAMES, CUSTOM_TITLES,
    save_path=S04_FIG / "stability_heatmap.png",
)

# ── Figure I: Prior width state profiles (one per dataset) ────────────────────
print("[I] Prior width state profiles ...")
scale_labels = {s: f"{s}× prior covariance" for s in PRIOR_WIDTH_SCALES}
ylim_overrides = {
    'CHO_GS46_F_all_pl40': ['Amm', 'Gln', 'Asn'],
    'CHO_GS46_F_all':      ['Amm', 'Gln', 'Lac', 'Asn'],
    'CHO_GS46_F_C_Inv':    ['Glc', 'Amm', 'Gln', 'Glu', 'Asn'],
    'CHO_T127_SNS_32':     ['Asn'],
}
pl.plot_prior_width_state_profiles(
    prior_width_sim, PRIOR_WIDTH_SCALES, BEST_ENSEMBLE_SIZES,
    datasets, STATE_NAMES, AXIS_NAMES,
    dataset_colours=DATASET_COLOURS, dataset_markers=DATASET_MARKERS,
    scale_labels=scale_labels,
    custom_titles=CUSTOM_TITLES,
    save_dir=S04_FIG,
    ylim_scale_overrides=ylim_overrides,
)

print("\nStep 4 complete.")
