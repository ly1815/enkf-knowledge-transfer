"""
03_irregular.py
===============
Step 3: EnKF with an irregular (48/72 h alternating) measurement schedule.

Requires Step 1 outputs in results/{RUN_NAME}/pkl/.

Run from project root:
    poetry run python scripts/03_irregular.py

Outputs saved to results/{RUN_NAME}/pkl/ and results/{RUN_NAME}/figures/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cho_enkf.config import (
    RESULTS_DIR,
    DATA_DIR, DATASET_FILES, STATE_NAMES, AXIS_NAMES,
    MEAN_PARAMETERS, PARAMETERS_ENSEMBLE_COVARIANCE,
    DATASET_NOISE_VARIANCES, KQ_DICT, KR_DICT,
    BEST_ENSEMBLE_SIZES, IRREGULAR_PATTERN_HOURS,
    DATASET_COLOURS, DATASET_MARKERS, CUSTOM_TITLES,
)
from cho_enkf.data_loader import load_datasets
from cho_enkf.enkf import run_pipeline_irregular_48_72
from cho_enkf.io_utils import set_dirs, ensure_dirs, has_results, save_pkl, load_pkl, fig_path
from cho_enkf import plotting as pl

S01_PKL = RESULTS_DIR / "01_ensemble_tuning" / "pkl"
S03_PKL = RESULTS_DIR / "03_irregular" / "pkl"
S03_FIG = RESULTS_DIR / "03_irregular" / "figures"
set_dirs(S03_PKL, S03_FIG)
ensure_dirs()

print("=" * 60)
print("Step 3: Irregular Measurements (48/72 h pattern)")
print("=" * 60)

datasets       = load_datasets(DATA_DIR, DATASET_FILES)
volume_results = load_pkl('volume_results.pkl', subdir=S01_PKL)

LOAD_FROM_PKL = has_results()
print(f"\n{'Loading from pkl' if LOAD_FROM_PKL else 'No existing results — running from scratch'} ...")

if LOAD_FROM_PKL:
    print("\nLoading saved results from pkl ...")
    PX_records_irregular    = load_pkl('PX_records_irregular.pkl')
    para_records_irregular  = load_pkl('para_records_irregular.pkl')
    Xf_para_irregular       = load_pkl('Xf_para_irregular.pkl')
    Xf_state_irregular      = load_pkl('Xf_state_irregular.pkl')
    Xpost_irregular         = load_pkl('Xpost_irregular.pkl')
    Z_irregular             = load_pkl('Z_irregular.pkl')
    sim_irregular           = load_pkl('sim_irregular.pkl')
    runtime_irregular       = load_pkl('runtime_irregular.pkl')
    update_times_irregular  = load_pkl('update_times_irregular.pkl')
    datasets_irregular      = load_pkl('datasets_irregular.pkl')
else:
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

# ── Figure K: Irregular measurement state profiles ────────────────────────────
print("\n[K] Irregular measurement profiles ...")
pl.plot_all_datasets_state_profiles(
    datasets_irregular, sim_irregular,
    axis_name=AXIS_NAMES,
    state_names=STATE_NAMES,
    dataset_colours=DATASET_COLOURS,
    dataset_markers=DATASET_MARKERS,
    dataset_ensemble_sizes=BEST_ENSEMBLE_SIZES,
    exp_meas_key="exp_meas_incomplete_48_72",
    save_dir=S03_FIG,
)

# ── MAE comparison: full data vs irregular schedule ──────────────────────────
print("\n" + "=" * 60)
print("MAE Comparison: Full Data vs Irregular Schedule")
print("=" * 60)

import numpy as np

ensemble_tuning = load_pkl('ensemble_tuning.pkl', subdir=S01_PKL)

dt_kf = 0.01

def _mae_per_state(traj, exp_meas, state_names):
    """Compute MAE per state between trajectory and experimental measurements."""
    exp_vals   = exp_meas.iloc[:, 1:9].values
    time_hours = exp_meas['Time (hours)'].values
    time_idx   = [min(round(t / dt_kf), traj.shape[0] - 1) for t in time_hours]
    pred       = traj[time_idx]
    mae = {}
    for i, sname in enumerate(state_names):
        obs   = exp_vals[:, i]
        valid = ~np.isnan(obs)
        if valid.any():
            mae[sname] = float(np.mean(np.abs(pred[valid, i] - obs[valid])))
        else:
            mae[sname] = float('nan')
    return mae

all_mae_full      = {}
all_mae_irregular = {}

for ds_name in BEST_ENSEMBLE_SIZES:
    best_n    = BEST_ENSEMBLE_SIZES[ds_name]
    exp_meas  = datasets[ds_name]['exp_meas']
    traj_full = ensemble_tuning[ds_name][best_n]
    traj_irr  = sim_irregular[ds_name]

    all_mae_full[ds_name]      = _mae_per_state(traj_full, exp_meas, STATE_NAMES)
    all_mae_irregular[ds_name] = _mae_per_state(traj_irr,  exp_meas, STATE_NAMES)

# Print comparison table
ds_order = [
    'CHO_T127_flask_PMJ', 'CHO_T127_SNS_36.5', 'CHO_T127_SNS_32',
    'CHO_GS46_F_C_Inv', 'CHO_GS46_F_all', 'CHO_GS46_F_all_pl40',
]
short_names = {
    'CHO_T127_flask_PMJ':  'A-Flask',
    'CHO_T127_SNS_36.5':   'A-Bio36',
    'CHO_T127_SNS_32':     'A-Bio32',
    'CHO_GS46_F_C_Inv':    'B-FeedC',
    'CHO_GS46_F_all':      'B-FeedU',
    'CHO_GS46_F_all_pl40': 'B-FeedU+',
}

for sname in STATE_NAMES:
    print(f"\n--- {sname} ---")
    print(f"{'Schedule':<12}", end="")
    for ds_name in ds_order:
        print(f"  {short_names[ds_name]:>10}", end="")
    print()
    print(f"{'Daily':<12}", end="")
    for ds_name in ds_order:
        print(f"  {all_mae_full[ds_name][sname]:>10.4g}", end="")
    print()
    print(f"{'Irregular':<12}", end="")
    for ds_name in ds_order:
        print(f"  {all_mae_irregular[ds_name][sname]:>10.4g}", end="")
    print()

save_pkl(all_mae_full,      'mae_full_data.pkl')
save_pkl(all_mae_irregular, 'mae_irregular.pkl')

print("\nStep 3 complete.")
