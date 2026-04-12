"""
07_ukf.py
=========
Step 7: Dual UKF comparison on CHO_GS46_F_C_Inv.

Runs a Dual UKF with the same Q, R, and initial parameter prior as the
EnKF benchmark (BEST_ENSEMBLE_SIZES['CHO_GS46_F_C_Inv'] = 75) and compares
state trajectories and RMSE.

UKF hyperparameters (Merwe scaled sigma points — defaults):
  alpha = 1e-3  (sigma point spread)
  beta  = 2.0   (optimal for Gaussian)
  kappa = 0.0   (secondary scaling)

Requires Step 1 outputs in results/{RUN_NAME}/01_ensemble_tuning/pkl/.

Run from project root:
    poetry run python scripts/07_ukf.py

Outputs saved to results/{RUN_NAME}/07_ukf/pkl/ and .../figures/
"""

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm

from cho_enkf.config import (
    RESULTS_DIR,
    DATA_DIR, DATASET_FILES, STATE_NAMES, AXIS_NAMES,
    MEAN_PARAMETERS, PARAMETERS_ENSEMBLE_COVARIANCE,
    DATASET_NOISE_VARIANCES, KQ_DICT, KR_DICT,
    BEST_ENSEMBLE_SIZES,
)
from cho_enkf.data_loader import load_datasets
from cho_enkf.io_utils import set_dirs, ensure_dirs, has_results, save_pkl, load_pkl, fig_path
from cho_enkf.ukf import UKF_DualParameterEstimation

# ── Config ────────────────────────────────────────────────────────────────────

DS_NAME = "CHO_GS46_F_C_Inv"

# UKF hyperparameters (Merwe scaled sigma points)
UKF_ALPHA = 0.5    # spread of sigma points (1e-3 causes negative Wc[0] → non-PD cov)
UKF_BETA  = 2.0    # optimal for Gaussian prior
UKF_KAPPA = 0.0    # secondary scaling

S01_PKL = RESULTS_DIR / "01_ensemble_tuning" / "pkl"
S07_PKL = RESULTS_DIR / "07_ukf" / "pkl"
S07_FIG = RESULTS_DIR / "07_ukf" / "figures"
set_dirs(S07_PKL, S07_FIG)
ensure_dirs()

print("=" * 60)
print("Step 7: Dual UKF — CHO_GS46_F_C_Inv")
print("=" * 60)
print(f"  alpha={UKF_ALPHA}, beta={UKF_BETA}, kappa={UKF_KAPPA}")
print(f"  Parameter sigma points: {2*len(MEAN_PARAMETERS)+1}  "
      f"(2×{len(MEAN_PARAMETERS)}+1)")
print(f"  State sigma points:     {2*len(STATE_NAMES)+1}  "
      f"(2×{len(STATE_NAMES)}+1)")

# ── Load data ─────────────────────────────────────────────────────────────────
datasets       = load_datasets(DATA_DIR, DATASET_FILES)
volume_results = load_pkl('volume_results.pkl', subdir=S01_PKL)
dataset        = datasets[DS_NAME]

LOAD_FROM_PKL = has_results()
print(f"\n{'Loading from pkl' if LOAD_FROM_PKL else 'Running from scratch'} ...")

# ── Run or load UKF ───────────────────────────────────────────────────────────

if LOAD_FROM_PKL:
    print("Loading saved UKF results ...")
    ukf_trajectory  = load_pkl('ukf_trajectory.pkl')
    ukf_rmse        = load_pkl('ukf_rmse.pkl')
    ukf_para_records = load_pkl('ukf_para_records.pkl')
    computation_time = load_pkl('ukf_computation_time.pkl')

else:
    print(f"\nRunning Dual UKF on {DS_NAME} ...")

    Fin     = dataset["schedule"]["Fin"].values
    Fout    = dataset["schedule"]["Fout"].values
    Fin_glc = dataset["schedule"]["Fin_glc"].values
    V       = volume_results[DS_NAME]
    feed    = dataset["feed"].set_index("Metabolite")["Concentration (mM)"].to_dict()

    state_cols  = ['Xv', 'mAb', 'Glc', 'Amm', 'Gln', 'Lac', 'Glu', 'Asn']
    state_init  = dataset["exp_meas"].iloc[0][state_cols].values.astype(float)
    time_exp    = dataset["exp_meas"]["Time (hours)"].values
    time_steps_A = [round(i * 0.01, 2) for i in range(len(Fin))]
    time_steps_B = [round(t, 2) for t in time_exp]

    kQ = KQ_DICT[DS_NAME]
    kR = KR_DICT[DS_NAME]
    Q  = kQ * np.diag(list(DATASET_NOISE_VARIANCES[DS_NAME]["process_var"].values()))
    R  = kR * np.diag(list(DATASET_NOISE_VARIANCES[DS_NAME]["obs_var"].values()))

    # Initial parameter covariance: variance = (std dev from tuning)²
    P_theta_init = np.diag([v**2 for v in PARAMETERS_ENSEMBLE_COVARIANCE.values()])

    ukf = UKF_DualParameterEstimation(
        DS_NAME, datasets, DATASET_NOISE_VARIANCES,
        dt_model=0.01, mean_parameters=MEAN_PARAMETERS,
        alpha=UKF_ALPHA, beta=UKF_BETA, kappa=UKF_KAPPA,
    )
    ukf.Q       = Q.copy()
    ukf.R       = R.copy()
    ukf.H       = np.eye(len(state_init))
    ukf.P_x     = Q.copy()          # initial state covariance = Q (same as EnKF)
    ukf.P_theta = P_theta_init.copy()
    ukf.x       = state_init.copy()

    trajectory = [state_init.copy()]
    t0 = time.time()

    for idx_A, step_A in enumerate(tqdm(time_steps_A, desc="Dual UKF")):
        controls = (Fin[idx_A], Fout[idx_A], Fin_glc[idx_A], V[idx_A], feed)

        if step_A in time_steps_B:
            idx_B = int(np.searchsorted(time_steps_B, step_A))
            ukf.forecast_for_parameters(controls)
            ukf.parameters_update(idx_B)

        ukf.forecast_for_states(controls)

        for idx_B, step_B in enumerate(time_steps_B):
            if step_A == step_B:
                ukf.states_update(idx_B)

        trajectory.append(ukf.x.copy())

    computation_time = time.time() - t0
    ukf_trajectory   = np.array(trajectory)

    time_indices = np.clip((time_exp * 100).astype(int), 0, len(Fin) - 1)
    ukf_rmse     = np.sqrt(
        (ukf_trajectory[time_indices] - dataset["exp_meas"].iloc[:, 1:9].values) ** 2
    )
    ukf_para_records = ukf.para_records

    print(f"\nRuntime: {computation_time:.1f} s")

    save_pkl(ukf_trajectory,   'ukf_trajectory.pkl')
    save_pkl(ukf_rmse,         'ukf_rmse.pkl')
    save_pkl(ukf_para_records, 'ukf_para_records.pkl')
    save_pkl(computation_time, 'ukf_computation_time.pkl')
    print("UKF results saved.")

# ── Load EnKF benchmark ───────────────────────────────────────────────────────

print("\nLoading EnKF benchmark from Step 1 pkl ...")
ensemble_tuning = load_pkl('ensemble_tuning.pkl', subdir=S01_PKL)
enkf_rmse_all   = load_pkl('rmse_results.pkl',    subdir=S01_PKL)

best_n          = BEST_ENSEMBLE_SIZES[DS_NAME]
enkf_trajectory = ensemble_tuning[DS_NAME][best_n]
enkf_rmse       = enkf_rmse_all[DS_NAME][best_n]

# ── RMSE comparison ───────────────────────────────────────────────────────────

print("\n--- Mean RMSE per state ---")
print(f"{'State':<8}  {'EnKF (N='+str(best_n)+')':<18}  {'Dual UKF':<18}")
print("-" * 48)
for i, s in enumerate(STATE_NAMES):
    e_rmse = float(np.mean(enkf_rmse[:, i]))
    u_rmse = float(np.mean(ukf_rmse[:, i]))
    print(f"{s:<8}  {e_rmse:<18.4g}  {u_rmse:<18.4g}")

overall_enkf = float(np.mean(enkf_rmse))
overall_ukf  = float(np.mean(ukf_rmse))
print("-" * 48)
print(f"{'Overall':<8}  {overall_enkf:<18.4g}  {overall_ukf:<18.4g}")

# ── Figure: state trajectory comparison ──────────────────────────────────────

exp_meas = dataset["exp_meas"]
t_meas   = exp_meas["Time (hours)"].values / 24.0

n_enkf   = len(enkf_trajectory)
n_ukf    = len(ukf_trajectory)
t_enkf   = np.linspace(0, n_enkf * 0.01 / 24, n_enkf)
t_ukf    = np.linspace(0, n_ukf  * 0.01 / 24, n_ukf)

fig, axes = plt.subplots(2, 4, figsize=(22, 10))
axes = axes.ravel()
sub_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']

for i, (sname, ylabel) in enumerate(zip(STATE_NAMES, AXIS_NAMES)):
    ax = axes[i]
    ax.plot(t_enkf, enkf_trajectory[:, i],
            color='#2980B9', linewidth=2, label=f'EnKF (N={best_n})')
    ax.plot(t_ukf,  ukf_trajectory[:, i],
            color='#C0392B', linewidth=2, linestyle='--', label='Dual UKF')

    std_col = f"{sname}_std"
    if std_col in exp_meas.columns:
        ax.errorbar(t_meas, exp_meas[sname].values,
                    yerr=exp_meas[std_col].values,
                    fmt='o', color='black', ecolor='grey',
                    elinewidth=1.5, capsize=3, markersize=5,
                    label='Experiment (±std)', zorder=5)
    else:
        ax.scatter(t_meas, exp_meas[sname].values,
                   color='black', s=30, zorder=5, label='Experiment')

    ax.set_title(sub_labels[i], fontsize=13, fontweight='bold', loc='left')
    ax.set_xlabel('Time (Days)', fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
    ax.set_xticks(np.arange(0, max(t_enkf.max(), t_ukf.max()) + 1, 2))
    ax.tick_params(axis='both', direction='in', length=4, width=1.2)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.grid(True, alpha=0.25)

handles, lbls = axes[0].get_legend_handles_labels()
fig.legend(handles, lbls, loc='lower center', ncol=3,
           bbox_to_anchor=(0.5, -0.02), fontsize=12, frameon=False,
           prop={'weight': 'bold'})
fig.suptitle(f'EnKF vs Dual UKF — {DS_NAME}', fontsize=14,
             fontweight='bold', y=1.01)
fig.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(fig_path("ukf_vs_enkf_trajectories.png"),
            dpi=150, bbox_inches='tight')
plt.show()

# ── Figure: RMSE bar chart ────────────────────────────────────────────────────

enkf_mean_rmse = np.mean(enkf_rmse, axis=0)   # (8,)
ukf_mean_rmse  = np.mean(ukf_rmse,  axis=0)   # (8,)

x      = np.arange(len(STATE_NAMES))
width  = 0.35

fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.bar(x - width/2, enkf_mean_rmse, width,
        label=f'EnKF (N={best_n})', color='#2980B9', alpha=0.85)
ax2.bar(x + width/2, ukf_mean_rmse,  width,
        label='Dual UKF',            color='#C0392B', alpha=0.85)
ax2.set_xticks(x)
ax2.set_xticklabels(STATE_NAMES, fontsize=11, fontweight='bold')
ax2.set_ylabel('Mean RMSE', fontsize=12, fontweight='bold')
ax2.set_title(f'Mean RMSE comparison — {DS_NAME}', fontsize=13, fontweight='bold')
ax2.legend(fontsize=11, prop={'weight': 'bold'})
ax2.grid(axis='y', linestyle='--', alpha=0.5)
ax2.tick_params(axis='both', direction='in', length=4, width=1.2)
for spine in ax2.spines.values():
    spine.set_linewidth(1.2)
plt.tight_layout()
plt.savefig(fig_path("ukf_vs_enkf_rmse.png"), dpi=150, bbox_inches='tight')
plt.show()

print("\nStep 7 complete.")
