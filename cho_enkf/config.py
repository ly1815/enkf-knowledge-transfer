"""
config.py
=========
Single source of truth for all configuration, constants, and tuning parameters.

Outputs are organised by script under results/:
  results/01_ensemble_tuning/  results/02_longterm_pred/
  results/03_irregular/        results/04_sensitivity/
  results/05_comparisons/
Each script calls io_utils.set_dirs() to point to its own subfolder.
"""

from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
RESULTS_DIR  = PROJECT_ROOT / "results"

# ─── Dataset file names (relative to DATA_DIR) ───────────────────────────────
DATASET_FILES = [
    "CHO_T127_flask_PMJ.xlsx",
    "CHO_T127_SNS_36.5.xlsx",
    "CHO_T127_SNS_32.xlsx",
    "CHO_GS46_F_C_Inv.xlsx",
    "CHO_GS46_F_all.xlsx",
    "CHO_GS46_F_all_pl40.xlsx",
]

# ─── State variables ─────────────────────────────────────────────────────────
STATE_NAMES = ['Xv', 'mAb', 'Glc', 'Amm', 'Gln', 'Lac', 'Glu', 'Asn']
STATE_NUM   = len(STATE_NAMES)
MEAS_NUM    = STATE_NUM

AXIS_NAMES = [
    'Viable Cell Density (cell $L^{-1}$)',
    'mAb Titre (mg $L^{-1}$)',
    'Glucose Concentration (mM)',
    'Ammonia Concentration (mM)',
    'Glutamine Concentration (mM)',
    'Lactate Concentration (mM)',
    'Glutamate Concentration (mM)',
    'Asparagine Concentration (mM)',
]

RMSE_NAMES = [
    'RMSE-Viable Cell Density (cell $L^{-1}$)',
    'RMSE-mAb Titre (mg $L^{-1}$)',
    'RMSE-Glucose Concentration (mM)',
    'RMSE-Ammonia Concentration (mM)',
    'RMSE-Glutamine Concentration (mM)',
    'RMSE-Lactate Concentration (mM)',
    'RMSE-Glutamate Concentration (mM)',
    'RMSE-Asparagine Concentration (mM)',
]

# ─── Model parameters ────────────────────────────────────────────────────────
PARAMETER_KEYS = [
    'mu_max', 'mu_d_max', 'Kglc', 'Kasn',
    'KIlac', 'KIamm', 'Kd_amm', 'm_lac', 'm_glc', 'Lac_max_1', 'Lac_max_2',
    'Yx_glc', 'Yx_gln', 'Yx_glu', 'Yx_lac', 'Yx_amm', 'Yx_asn', 'Yx_asp',
    'Ygln_amm', 'Ylac_glc', 'Yasp_asn', 'Yasn_asp', 'm_mAb', 'YmAb_mu',
]

LATEX_LABELS = {
    "mu_max":    r"$\mu_{\max}$",
    "mu_d_max":  r"$\mu_{d,\max}$",
    "Kglc":      r"$K_{\text{Glc}}$",
    "Kasn":      r"$K_{\text{Asn}}$",
    "KIlac":     r"$K_{I,\text{Lac}}$",
    "KIamm":     r"$K_{I,\text{Amm}}$",
    "Kd_amm":    r"$K_{d,\text{Amm}}$",
    "m_lac":     r"$m_{\text{Lac}}$",
    "m_glc":     r"$m_{\text{Glc}}$",
    "Lac_max_1": r"$Lac_{\max 1}$",
    "Lac_max_2": r"$Lac_{\max 2}$",
    "Yx_glc":    r"$Y_{X,\text{Glc}}$",
    "Yx_gln":    r"$Y_{X,\text{Gln}}$",
    "Yx_glu":    r"$Y_{X,\text{Glu}}$",
    "Yx_lac":    r"$Y_{X,\text{Lac}}$",
    "Yx_amm":    r"$Y_{X,\text{Amm}}$",
    "Yx_asn":    r"$Y_{X,\text{Asn}}$",
    "Yx_asp":    r"$Y_{X,\text{Asp}}$",
    "Ygln_amm":  r"$Y_{\text{Gln,Amm}}$",
    "Ylac_glc":  r"$Y_{\text{Lac,Glc}}$",
    "Yasp_asn":  r"$Y_{\text{Asp,Asn}}$",
    "Yasn_asp":  r"$Y_{\text{Asn,Asp}}$",
    "m_mAb":     r"$m_{\text{mAb}}$",
    "YmAb_mu":   r"$Y_{\text{mAb},\mu}$",
}

# Nominal model parameters — CHO-T127 shake flask, Kotidis et al. 2019
MEAN_PARAMETERS = {
    'mu_max':    0.065,
    'mu_d_max':  0.015,
    'Kglc':      14.0378,
    'Kasn':      2.62371,
    'KIlac':     1000,
    'KIamm':     3.16935,
    'Kd_amm':    14.2830,
    'm_lac':     1.87253e-10,
    'm_glc':     3.43293e-11,
    'Lac_max_1': 21.1983,
    'Lac_max_2': 16,
    'Yx_glc':    1.0115e9,
    'Yx_gln':    4.64127e9,
    'Yx_glu':    1.45647e10,
    'Yx_lac':    5.45539e7,
    'Yx_amm':    2.36299e9,
    'Yx_asn':    7.6824e8,
    'Yx_asp':    3.59e9,
    'Ygln_amm':  0.104524,
    'Ylac_glc':  1.56,
    'Yasp_asn':  0.126,
    'Yasn_asp':  0.1,
    'm_mAb':     4.12718e-10,
    'YmAb_mu':   3.38956e-9,
}

# Parameter reparametrization for GS46-F-C-Inv taken from Strategic Framework for Parameterization of Cell Culture Models (Kotidis et al. 2019)
CHO_GS46_F_C_Inv_PARAMETERS = {
    'mu_max':    6.96e-2,
    'mu_d_max':  0.015,
    'Kglc':      14.0378,
    'Kasn':      2.62371,
    'KIlac':     1000,
    'KIamm':     3.16935,
    'Kd_amm':    14.2830,
    'm_lac':     1.87253e-10,
    'm_glc':     3.35e-11,
    'Lac_max_1': 18.55,
    'Lac_max_2': 7.14,
    'Yx_glc':    1.0115e9,
    'Yx_gln':    1.85e10,
    'Yx_glu':    5.68e9,
    'Yx_lac':    5.45539e7,
    'Yx_amm':    4.66e9,
    'Yx_asn':    8.69e8,
    'Yx_asp':    1.06e9,
    'Ygln_amm':  0.104524,
    'Ylac_glc':  1.56,
    'Yasp_asn':  0.126,
    'Yasn_asp':  0.1,
    'm_mAb':     1.31e-9,   # mg/cell/h
    'YmAb_mu':   3.38956e-9,
}

# Parameter ensemble prior covariance (std-dev widths)
PARAMETERS_ENSEMBLE_COVARIANCE = {
    'mu_max':    0.02,
    'mu_d_max':  0.0035,
    'Kglc':      4.2,
    'Kasn':      0.5,
    'KIlac':     150,
    'KIamm':     0.69,
    'Kd_amm':    2.5,
    'm_lac':     2e-11,
    'm_glc':     3.1e-12,
    'Lac_max_1': 3.0,
    'Lac_max_2': 2.0,
    'Yx_glc':    1.774e8,
    'Yx_gln':    6.033e8,
    'Yx_glu':    3.297e9,
    'Yx_lac':    2.568e7,
    'Yx_amm':    6.020e8,
    'Yx_asn':    2.077e8,
    'Yx_asp':    8.072e8,
    'Ygln_amm':  0.030,
    'Ylac_glc':  0.15,
    'Yasp_asn':  0.01,
    'Yasn_asp':  0.01,
    'm_mAb':     9.05e-11,
    'YmAb_mu':   3.1e-10,
}

# ─── Noise variances (process Q and observation R) ───────────────────────────
DATASET_NOISE_VARIANCES = {
    "CHO_T127_flask_PMJ": {
        "process_var": {'Xv': 1e+18, 'mAb': 1.0e+4, 'Glc': 1.6e+01, 'Amm': 1.0,
                        'Gln': 2.0, 'Lac': 1.2, 'Glu': 0.2, 'Asn': 0.8},
        "obs_var":     {'Xv': 3e+17, 'mAb': 1.5e+3, 'Glc': 5.0, 'Amm': 0.1,
                        'Gln': 0.1, 'Lac': 1.2, 'Glu': 0.004, 'Asn': 0.25},
    },
    "CHO_T127_SNS_36.5": {
        "process_var": {'Xv': 1.0e+19, 'mAb': 1.0e+05, 'Glc': 1.6e+01, 'Amm': 1.0,
                        'Gln': 4.0, 'Lac': 1.2, 'Glu': 0.2, 'Asn': 0.8},
        "obs_var":     {'Xv': 5e+17, 'mAb': 1.5e+03, 'Glc': 5.0, 'Amm': 0.1,
                        'Gln': 0.1, 'Lac': 1.2, 'Glu': 0.004, 'Asn': 0.25},
    },
    "CHO_T127_SNS_32": {
        "process_var": {'Xv': 2.0e+19, 'mAb': 1.0e+05, 'Glc': 9.0e+02, 'Amm': 6.0,
                        'Gln': 4.0, 'Lac': 5.0, 'Glu': 1.6, 'Asn': 5.0},
        "obs_var":     {'Xv': 3e+17, 'mAb': 1.5e+03, 'Glc': 5.0, 'Amm': 0.2,
                        'Gln': 0.1, 'Lac': 0.5, 'Glu': 0.04, 'Asn': 0.25},
    },
    "CHO_GS46_F_C_Inv": {
        "process_var": {'Xv': 2.0e+19, 'mAb': 1.0e+05, 'Glc': 9.6e+01, 'Amm': 6.0,
                        'Gln': 4.0, 'Lac': 5.0, 'Glu': 1.6, 'Asn': 3.0},
        "obs_var":     {'Xv': 3e+17, 'mAb': 1.5e+03, 'Glc': 5.0, 'Amm': 0.1,
                        'Gln': 0.1, 'Lac': 0.2, 'Glu': 0.04, 'Asn': 0.025},
    },
    "CHO_GS46_F_all": {
        "process_var": {'Xv': 2.0e+19, 'mAb': 1.0e+05, 'Glc': 9.6e+01, 'Amm': 6.0,
                        'Gln': 4.0, 'Lac': 5.0, 'Glu': 1.6, 'Asn': 3.0},
        "obs_var":     {'Xv': 3e+17, 'mAb': 1.5e+03, 'Glc': 5.0, 'Amm': 0.1,
                        'Gln': 0.1, 'Lac': 0.2, 'Glu': 0.08, 'Asn': 0.025},
    },
    "CHO_GS46_F_all_pl40": {
        "process_var": {'Xv': 2.0e+19, 'mAb': 1.0e+05, 'Glc': 9.6e+01, 'Amm': 6.0,
                        'Gln': 4.0, 'Lac': 5.0, 'Glu': 1.4, 'Asn': 3.0},
        "obs_var":     {'Xv': 3e+17, 'mAb': 1.5e+03, 'Glc': 5.0, 'Amm': 0.5,
                        'Gln': 0.1, 'Lac': 0.2, 'Glu': 0.04, 'Asn': 0.025},
    },
}

# ─── Noise scaling factors ────────────────────────────────────────────────────
KQ_DICT = {ds: 1e-9 for ds in DATASET_NOISE_VARIANCES}
KR_DICT = {ds: 1.0  for ds in DATASET_NOISE_VARIANCES}

# ─── Initial volumes (L) ─────────────────────────────────────────────────────
INITIAL_VOLUMES = {
    "CHO_GS46_F_C_Inv":    0.05,
    "CHO_GS46_F_all":      0.05,
    "CHO_GS46_F_all_pl40": 0.05,
    "CHO_T127_SNS_36.5":   0.9,
    "CHO_T127_SNS_32":     0.9,
    "CHO_T127_flask_PMJ":  0.1,
}

# ─── Best ensemble sizes (determined from tuning) ─────────────────────────────
BEST_ENSEMBLE_SIZES = {
    'CHO_T127_flask_PMJ':  50,
    'CHO_T127_SNS_36.5':   50,
    'CHO_T127_SNS_32':     50,
    'CHO_GS46_F_C_Inv':    75,
    'CHO_GS46_F_all':      75,
    'CHO_GS46_F_all_pl40': 75,
}

# Ensemble sizes to sweep during tuning
TUNING_ENSEMBLE_SIZES = [10, 25, 50, 75]

# ─── Dataset display customisation ───────────────────────────────────────────
CUSTOM_TITLES = {
    'CHO_T127_flask_PMJ':  'Cell Line A - Shake Flask 36.5°C',
    'CHO_T127_SNS_36.5':   'Cell Line A - Bioreactor 36.5°C',
    'CHO_T127_SNS_32':     'Cell Line A - Bioreactor 32°C',
    'CHO_GS46_F_C_Inv':    'Cell Line B - Feed C',
    'CHO_GS46_F_all':      'Cell Line B - Feed U',
    'CHO_GS46_F_all_pl40': 'Cell Line B - Feed U plus 40%',
}

DATASET_COLOURS = {
    "CHO_GS46_F_C_Inv":    "darkorange",
    "CHO_GS46_F_all":      "seagreen",
    "CHO_GS46_F_all_pl40": "navy",
    "CHO_T127_flask_PMJ":  "dimgrey",
    "CHO_T127_SNS_36.5":   "purple",
    "CHO_T127_SNS_32":     "teal",
}

DATASET_MARKERS = {
    "CHO_GS46_F_C_Inv":    "X",
    "CHO_GS46_F_all":      "s",
    "CHO_GS46_F_all_pl40": "^",
    "CHO_T127_flask_PMJ":  "o",
    "CHO_T127_SNS_36.5":   "v",
    "CHO_T127_SNS_32":     "D",
}

# ─── Sensitivity analysis ────────────────────────────────────────────────────
PRIOR_WIDTH_SCALES    = [0.5, 1.0, 1.5, 2.0]
PARAM_SENS_PERTURBATIONS = [0.10, 0.20, 0.30]   # ±10%, ±20%, ±30%

# Long-term forecast indices (measurement update number from which to start forecast)
FORECAST_INDICES = {
    "CHO_T127_SNS_36.5":   [7, 9, 12],
    "CHO_T127_SNS_32":     [7, 9, 12],
    "CHO_GS46_F_C_Inv":    [7, 9, 12],
    "CHO_GS46_F_all":      [7, 9, 12],
    "CHO_GS46_F_all_pl40": [7, 9, 12],
    "CHO_T127_flask_PMJ":  [6, 8, 10],
}

# Irregular measurement schedule (hours)
IRREGULAR_PATTERN_HOURS = (48, 72)
