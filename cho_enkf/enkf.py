"""
enkf.py
=======
Ensemble Kalman Filter (EnKF) classes and high-level runners.

Classes
-------
EnKF_ParameterEstimation  — dual state + parameter estimation via EnKF
ObsReplicates             — noisy observation ensemble generation
Visualization             — parameter ensemble visualisation helper

Runners
-------
run_enkf_with_tuning                      — sweep ensemble sizes
enkf_long_pred_best_ensemble_size         — long-term prediction
enkf_irregular_updates_best_ensemble_size — irregular measurement schedule
run_enkf_with_mean_params                 — sensitivity (override mean params)
run_pipeline_irregular_48_72              — convenience wrapper for irregular run
"""

import time
import numpy as np
import pandas as pd
from numpy.linalg import inv
from tqdm import tqdm

from cho_enkf.model import model_step


# ─── Visualization helper ────────────────────────────────────────────────────

class Visualization:
    """Generate and optionally plot a parameter prior ensemble."""

    def __init__(self, datasets_para: dict):
        self.datasets_para = datasets_para
        self.PX      = {ds: [] for ds in datasets_para}
        self.size_PX = {ds: 0  for ds in datasets_para}

    def create_parameter_ensemble(self, ensemble_size: int, PX_Cov: dict):
        np.random.seed(42)
        for ds in self.datasets_para:
            self.size_PX[ds] = ensemble_size
            self.PX[ds] = []
            rng = {k: np.random.normal(0, PX_Cov[ds][k], ensemble_size)
                   for k in self.datasets_para[ds]}
            for i in range(ensemble_size):
                member = self.datasets_para[ds].copy()
                for k in member:
                    member[k] += rng[k][i]
                self.PX[ds].append(member)


# ─── Observation replicate generator ────────────────────────────────────────

class ObsReplicates:
    """Generate noisy observation ensembles for all datasets."""

    def __init__(self, datasets: dict, state_num: int,
                 dataset_noise_variances: dict, ensemble_sizes: dict):
        self.datasets               = datasets
        self.state_num              = state_num
        self.dataset_noise_variances = dataset_noise_variances
        self.ensemble_sizes         = ensemble_sizes
        self.Z                      = {}
        self.T_models               = self._generate_T_models()

    def _generate_T_models(self, dt_model: float = 0.01) -> dict:
        T_models = {}
        for ds, data in self.datasets.items():
            Fin     = data["schedule"]["Fin"].values
            t_total = len(Fin) / 100
            N_model = int(t_total / dt_model)
            T_models[ds] = np.linspace(0, t_total, N_model + 1)
        return T_models

    def create_noisy_observations(self):
        np.random.seed(42)
        state_cols = ['Xv', 'mAb', 'Glc', 'Amm', 'Gln', 'Lac', 'Glu', 'Asn']
        for ds, data in self.datasets.items():
            real_obs   = data["exp_meas"][state_cols].values
            N_meas     = real_obs.shape[0]
            obs_var    = np.array(list(self.dataset_noise_variances[ds]["obs_var"].values()))
            ens_size   = self.ensemble_sizes.get(ds, 50)
            noise      = np.random.normal(0, np.sqrt(obs_var),
                                          size=(ens_size, N_meas, self.state_num))
            self.Z[ds] = real_obs + noise


# ─── Core EnKF class ─────────────────────────────────────────────────────────

class EnKF_ParameterEstimation:
    """
    Dual state and parameter estimation via the Ensemble Kalman Filter.

    Attributes that must be set before calling forecast/update methods:
      Q, R, H, fx, dt, z, Z, PX, para, x, X
    (all set explicitly in the runner functions below).
    """

    def __init__(self, dataset_name: str, datasets: dict,
                 dataset_noise_variances: dict, dt_model: float,
                 mean_parameters: dict = None):
        np.random.seed(42)
        self.dataset_name           = dataset_name
        self.datasets               = datasets
        self.dataset_noise_variances = dataset_noise_variances
        self.dt                     = dt_model

        exp_meas  = datasets[dataset_name]["exp_meas"]
        state_cols = ['Xv', 'mAb', 'Glc', 'Amm', 'Gln', 'Lac', 'Glu', 'Asn']
        self.x    = exp_meas.iloc[0][state_cols].values.astype(float)
        self.z    = exp_meas[state_cols].values
        self.H    = np.eye(len(self.x))

        # Will be overridden by runner functions
        self.para = mean_parameters.copy() if mean_parameters else {}
        self.X    = None
        self.PX   = None
        self.Z    = None
        self.Q    = None
        self.R    = None
        self.Kp   = None
        self.Ks   = None
        self.fx   = model_step

        # Records
        self.X_forecast_for_parameters_records = []
        self.X_forecast_for_states_records     = []
        self.X_posterior_records               = []
        self.PX_records                        = []
        self.para_records                      = []

    # ── Ensemble initialisation ──────────────────────────────────────────────

    def create_parameters_ensemble(self, ensemble_size: int, PX_Cov: dict):
        self.size_PX = ensemble_size
        np.random.seed(42)
        rng = {k: np.random.normal(0, PX_Cov[k], ensemble_size) for k in self.para}
        self.PX = []
        for i in range(ensemble_size):
            member = self.para.copy()
            for k in self.para:
                member[k] = max(1e-12, member[k] + rng[k][i])
            self.PX.append(member)

    def create_states_ensemble(self, ensemble_size: int):
        self.size_X = ensemble_size
        np.random.seed(42)
        self.X = np.random.multivariate_normal(self.x, self.Q, ensemble_size)
        self.X = np.where(self.X < 0, 1e-12, self.X)

    def create_noisy_observations(self, ensemble_size: int):
        np.random.seed(42)
        noise = np.random.multivariate_normal(np.zeros(len(self.x)), self.R,
                                              (ensemble_size, len(self.z)))
        self.Z = [np.where(self.z + n < 0, 1e-12, self.z + n) for n in noise]

    # ── Forecast steps ───────────────────────────────────────────────────────

    def forecast_for_parameters(self, controls):
        Fin_t, Fout_t, Fin_glc_t, V_t, feed = controls
        X_new = [self.fx(self.PX[n], x, 0.0, Fin_t, Fout_t, Fin_glc_t, V_t, self.dt, feed)
                 for n, x in enumerate(self.X)]
        np.random.seed(42)
        self.X = (np.array(X_new)
                  + np.random.multivariate_normal(np.zeros(len(self.x)), self.Q, self.size_PX))
        self.X = np.where(self.X < 0, 1e-12, self.X)
        self.x = np.mean(self.X, axis=0)
        for k in self.PX[0]:
            self.para[k] = np.mean([d[k] for d in self.PX])
        self.X_forecast_for_parameters_records.append(self.X.copy().tolist())

    def forecast_for_states(self, controls):
        Fin_t, Fout_t, Fin_glc_t, V_t, feed = controls
        X_new = [self.fx(self.PX[n], x, 0.0, Fin_t, Fout_t, Fin_glc_t, V_t, self.dt, feed)
                 for n, x in enumerate(self.X)]
        np.random.seed(42)
        self.X = (np.array(X_new)
                  + np.random.multivariate_normal(np.zeros(len(self.x)), self.Q, self.size_X))
        self.X = np.where(self.X < 0, 1e-12, self.X)
        self.x = np.mean(self.X, axis=0)
        for k in self.PX[0]:
            self.para[k] = np.mean([d[k] for d in self.PX])
        self.X_forecast_for_states_records.append(self.X.copy().tolist())

    def forecast_long(self, controls, current_state: np.ndarray,
                      current_para: dict) -> np.ndarray:
        Fin_t, Fout_t, Fin_glc_t, V_t, feed = controls
        new_state = self.fx(current_para, current_state, 0.0,
                            Fin_t, Fout_t, Fin_glc_t, V_t, self.dt, feed)
        np.random.seed(42)
        new_state += np.random.multivariate_normal(np.zeros(len(self.x)), self.Q)
        return new_state

    # ── Update steps ─────────────────────────────────────────────────────────

    def parameters_update(self, time_index: int):
        E          = self.X - self.x
        P          = E.T @ E / (self.size_PX - 1)
        PX_mat     = np.array([[d[k] for k in d] for d in self.PX])
        para_vec   = np.array(list(self.para.values()))
        theta      = (PX_mat - para_vec).T @ (self.X - self.x) / (self.size_PX - 1)
        Kp         = theta @ self.H.T @ inv(self.H @ P @ self.H.T + self.R)
        updated    = []
        for param, z_row, x_row in zip(self.PX, self.Z, self.X):
            raw = np.array(list(param.values())) + Kp @ (z_row[time_index] - self.H @ x_row)
            updated.append({k: max(1e-12, v) for k, v in zip(self.para.keys(), raw)})
        self.PX = updated
        self.PX_records.append(self.PX)
        for k in self.PX[0]:
            self.para[k] = np.mean([d[k] for d in self.PX])
        self.para_records.append(self.para.copy())

    def states_update(self, time_index: int):
        mean_x = np.mean(self.X, axis=0)
        E      = self.X - mean_x
        P      = E.T @ E / (self.size_X - 1)
        Ks     = P @ self.H.T @ inv(self.H @ P @ self.H.T + self.R)
        self.X = np.array([x + Ks @ (z[time_index] - self.H @ x)
                           for x, z in zip(self.X, self.Z)])
        self.X = np.where(self.X < 0, 1e-12, self.X)
        self.X_posterior_records.append(self.X.copy().tolist())
        self.x = np.mean(self.X, axis=0)


# ─── Runner: ensemble-size tuning ────────────────────────────────────────────

def run_enkf_with_tuning(datasets: dict, dataset_noise_variances: dict,
                          volume_results: dict, ensemble_sizes: list,
                          mean_parameters: dict, parameters_ensemble_covariance: dict,
                          kQ_dict: dict, kR_dict: dict):
    """
    Run the EnKF across all ensemble sizes for all datasets.

    Returns
    -------
    ensemble_tuning, mae_results, computation_times,
    PX_records_all, para_records_all,
    X_forecast_for_parameters_records_all,
    X_forecast_for_states_records_all,
    X_posterior_records_all, Z_all
    """
    ensemble_tuning = {}
    mae_results     = {}
    computation_times = {}
    PX_records_all  = {}
    para_records_all = {}
    Xf_para_all     = {}
    Xf_state_all    = {}
    Xpost_all       = {}
    Z_all           = {}

    for dataset_name, data in datasets.items():
        print(f"\nDataset: {dataset_name}")
        Fin     = data["schedule"]["Fin"].values
        Fout    = data["schedule"]["Fout"].values
        Fin_glc = data["schedule"]["Fin_glc"].values
        V       = volume_results[dataset_name]
        feed    = data["feed"].set_index("Metabolite")["Concentration (mM)"].to_dict()

        time_steps_A = [round(i * 0.01, 2) for i in range(len(Fin))]
        time_steps_B = [round(t, 2) for t in data["exp_meas"]["Time (hours)"].values]

        state_init  = data["exp_meas"].iloc[0][
            ['Xv', 'mAb', 'Glc', 'Amm', 'Gln', 'Lac', 'Glu', 'Asn']].values
        time_exp    = data["exp_meas"]["Time (hours)"].values
        time_indices = np.clip((time_exp * 100).astype(int), 0, len(Fin) - 1)

        for key in ['ensemble_tuning', 'mae_results', 'computation_times',
                    'PX_records_all', 'para_records_all', 'Xf_para_all',
                    'Xf_state_all', 'Xpost_all', 'Z_all']:
            locals()[key].setdefault(dataset_name, {})

        ensemble_tuning[dataset_name]  = {}
        mae_results[dataset_name]      = {}
        computation_times[dataset_name] = {}
        PX_records_all[dataset_name]   = {}
        para_records_all[dataset_name] = {}
        Xf_para_all[dataset_name]      = {}
        Xf_state_all[dataset_name]     = {}
        Xpost_all[dataset_name]        = {}
        Z_all[dataset_name]            = {}

        for ens_size in ensemble_sizes:
            print(f"  Ensemble size {ens_size} ...")
            t0 = time.time()

            kQ = kQ_dict[dataset_name]
            kR = kR_dict[dataset_name]
            Q  = kQ * np.diag(list(dataset_noise_variances[dataset_name]["process_var"].values()))
            R  = kR * np.diag(list(dataset_noise_variances[dataset_name]["obs_var"].values()))
            H  = np.eye(len(state_init))

            enkf             = EnKF_ParameterEstimation(dataset_name, datasets,
                                                         dataset_noise_variances,
                                                         dt_model=0.01,
                                                         mean_parameters=mean_parameters)
            enkf.PX          = []
            enkf.PX_records  = []
            enkf.para        = mean_parameters.copy()
            enkf.para_records = [mean_parameters.copy()]
            enkf.x           = state_init.copy()
            enkf.Q           = Q.copy()
            enkf.H           = H.copy()
            enkf.R           = R.copy()
            enkf.z           = data["exp_meas"].iloc[:, 1:9].values.copy()
            enkf.Z           = []
            enkf.fx          = model_step
            enkf.X_forecast_for_parameters_records = []
            enkf.X_forecast_for_states_records     = []
            enkf.X_posterior_records               = []

            enkf.create_parameters_ensemble(ens_size, parameters_ensemble_covariance)
            enkf.create_states_ensemble(ens_size)
            enkf.create_noisy_observations(ens_size)
            enkf.PX_records.append(enkf.PX)

            set_EnKF = [state_init.copy()]

            for idx_A, step_A in enumerate(tqdm(time_steps_A,
                                                desc=f"EnKF N={ens_size}")):
                if step_A in time_steps_B:
                    idx_B = int(np.searchsorted(time_steps_B, step_A))
                    enkf.forecast_for_parameters(
                        (Fin[idx_A], Fout[idx_A], Fin_glc[idx_A], V[idx_A], feed))
                    enkf.parameters_update(idx_B)

                enkf.forecast_for_states(
                    (Fin[idx_A], Fout[idx_A], Fin_glc[idx_A], V[idx_A], feed))

                for idx_B, step_B in enumerate(time_steps_B):
                    if step_A == step_B:
                        enkf.states_update(idx_B)

                set_EnKF.append(enkf.x.copy())

            set_EnKF = np.array(set_EnKF)
            ensemble_tuning[dataset_name][ens_size]  = set_EnKF
            mae_results[dataset_name][ens_size]       = np.abs(
                set_EnKF[time_indices] - data["exp_meas"].iloc[:, 1:9].values)
            computation_times[dataset_name][ens_size] = time.time() - t0
            para_records_all[dataset_name][ens_size]  = enkf.para_records
            Xf_para_all[dataset_name][ens_size]       = enkf.X_forecast_for_parameters_records
            Xf_state_all[dataset_name][ens_size]      = enkf.X_forecast_for_states_records
            Xpost_all[dataset_name][ens_size]         = enkf.X_posterior_records
            Z_all[dataset_name][ens_size]             = enkf.Z
            PX_records_all[dataset_name][ens_size]    = enkf.PX_records

    return (ensemble_tuning, mae_results, computation_times,
            PX_records_all, para_records_all,
            Xf_para_all, Xf_state_all, Xpost_all, Z_all)


# ─── Runner: long-term prediction ────────────────────────────────────────────

def enkf_long_pred_best_ensemble_size(datasets: dict, dataset_noise_variances: dict,
                                       volume_results: dict, dataset_ensemble_sizes: dict,
                                       mean_parameters: dict,
                                       parameters_ensemble_covariance: dict,
                                       kQ_dict: dict, kR_dict: dict):
    """
    Run the EnKF with the best ensemble size per dataset and generate
    long-term forecasts from each measurement update point.

    Returns a tuple of 8 dicts matching the notebook's unpacking pattern.
    """
    PX_records_best  = {}
    para_records_best = {}
    Xf_para_best     = {}
    Xf_state_best    = {}
    Xpost_best       = {}
    Z_best           = {}
    sim_best         = {}
    long_term_best   = {}

    for dataset_name, data in datasets.items():
        print(f"\nLong-term prediction: {dataset_name}")
        Fin     = data["schedule"]["Fin"].values
        Fout    = data["schedule"]["Fout"].values
        Fin_glc = data["schedule"]["Fin_glc"].values
        V       = volume_results[dataset_name]
        exp_meas = data["exp_meas"]
        state_init = exp_meas.iloc[0][
            ['Xv', 'mAb', 'Glc', 'Amm', 'Gln', 'Lac', 'Glu', 'Asn']].values
        feed    = data["feed"].set_index("Metabolite")["Concentration (mM)"].to_dict()

        time_steps_A = [round(i * 0.01, 2) for i in range(len(Fin))]
        time_steps_B = [round(t, 2) for t in exp_meas["Time (hours)"].values]

        ens_size = dataset_ensemble_sizes.get(dataset_name, 50)
        kQ = kQ_dict[dataset_name]
        kR = kR_dict[dataset_name]
        Q  = kQ * np.diag(list(dataset_noise_variances[dataset_name]["process_var"].values()))
        R  = kR * np.diag(list(dataset_noise_variances[dataset_name]["obs_var"].values()))
        H  = np.eye(len(state_init))

        enkf             = EnKF_ParameterEstimation(dataset_name, datasets,
                                                     dataset_noise_variances,
                                                     dt_model=0.01,
                                                     mean_parameters=mean_parameters)
        enkf.PX          = []
        enkf.PX_records  = []
        enkf.para        = mean_parameters.copy()
        enkf.para_records = [mean_parameters.copy()]
        enkf.x           = state_init.copy()
        enkf.Q           = Q.copy()
        enkf.H           = H.copy()
        enkf.R           = R.copy()
        enkf.z           = exp_meas.iloc[:, 1:9].values
        enkf.Z           = []
        enkf.fx          = model_step
        enkf.X_forecast_for_parameters_records = []
        enkf.X_forecast_for_states_records     = []
        enkf.X_posterior_records               = []

        set_EnKF      = [state_init.copy()]
        long_forecasts = []

        enkf.create_parameters_ensemble(ens_size, parameters_ensemble_covariance)
        enkf.create_states_ensemble(ens_size)
        enkf.create_noisy_observations(ens_size)
        enkf.PX_records.append(enkf.PX)

        for idx_A, step_A in enumerate(tqdm(time_steps_A, desc=dataset_name)):
            if step_A in time_steps_B:
                idx_B = int(np.searchsorted(time_steps_B, step_A))
                enkf.forecast_for_parameters(
                    (Fin[idx_A], Fout[idx_A], Fin_glc[idx_A], V[idx_A], feed))
                enkf.parameters_update(idx_B)

            enkf.forecast_for_states(
                (Fin[idx_A], Fout[idx_A], Fin_glc[idx_A], V[idx_A], feed))

            for idx_B, step_B in enumerate(time_steps_B):
                if step_A == step_B:
                    enkf.states_update(idx_B)

            set_EnKF.append(enkf.x.copy())

            if step_A in time_steps_B:
                cur_state = enkf.x.copy()
                cur_para  = {k: np.mean([d[k] for d in enkf.PX]) for k in enkf.PX[0]}
                traj = [cur_state.copy()]
                for fi in range(idx_A + 1, len(time_steps_A)):
                    ctrl = (Fin[fi], Fout[fi], Fin_glc[fi], V[fi], feed)
                    cur_state = enkf.forecast_long(ctrl, cur_state, cur_para)
                    traj.append(cur_state.copy())
                long_forecasts.append(np.array(traj))

        sim_best[dataset_name]   = np.array(set_EnKF)
        long_term_best[dataset_name] = long_forecasts
        PX_records_best[dataset_name]  = {ens_size: enkf.PX_records}
        para_records_best[dataset_name] = {ens_size: enkf.para_records}
        Xf_para_best[dataset_name]     = {ens_size: enkf.X_forecast_for_parameters_records}
        Xf_state_best[dataset_name]    = {ens_size: enkf.X_forecast_for_states_records}
        Xpost_best[dataset_name]       = {ens_size: enkf.X_posterior_records}
        Z_best[dataset_name]           = {ens_size: enkf.Z}

    return (PX_records_best, para_records_best,
            Xf_para_best, Xf_state_best, Xpost_best,
            Z_best, sim_best, long_term_best)


# ─── Irregular measurement helpers ───────────────────────────────────────────

def make_incomplete_exp_meas_48_72(exp_meas: pd.DataFrame,
                                    state_cols=('Xv', 'mAb', 'Glc', 'Amm',
                                                'Gln', 'Lac', 'Glu', 'Asn'),
                                    pattern_hours=(48, 72),
                                    decimal_places=2,
                                    keep_nearest=True) -> pd.DataFrame:
    """
    Return a copy of exp_meas where only rows on an irregular 48/72 h
    schedule retain state measurements; all other rows are set to NaN.
    """
    t        = exp_meas["Time (hours)"].astype(float).values
    t_round  = np.round(t, decimal_places)
    start_t  = float(t_round.min())
    end_t    = float(t_round.max())

    schedule = [round(start_t, decimal_places)]
    k = 0
    while schedule[-1] < end_t:
        schedule.append(round(schedule[-1] + pattern_hours[k % len(pattern_hours)],
                              decimal_places))
        k += 1
    schedule = np.array([s for s in schedule if s <= end_t], dtype=float)

    out = exp_meas.copy()
    for c in state_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    out.loc[:, list(state_cols)] = np.nan

    if keep_nearest:
        idx_keep = np.unique([int(np.argmin(np.abs(t_round - s))) for s in schedule])
    else:
        idx_keep = np.where(np.isin(t_round, schedule))[0]

    kept = exp_meas.loc[idx_keep, list(state_cols)].apply(
        pd.to_numeric, errors="coerce").astype(float).values
    out.loc[idx_keep, list(state_cols)] = kept
    return out


def attach_incomplete_measurements(datasets: dict,
                                    state_cols=('Xv', 'mAb', 'Glc', 'Amm',
                                                'Gln', 'Lac', 'Glu', 'Asn'),
                                    pattern_hours=(48, 72),
                                    decimal_places=2,
                                    keep_nearest=True,
                                    new_key="exp_meas_incomplete_48_72") -> dict:
    datasets_out = {}
    for name, data in datasets.items():
        d = data.copy()
        d[new_key] = make_incomplete_exp_meas_48_72(
            data["exp_meas"], state_cols=state_cols,
            pattern_hours=pattern_hours, decimal_places=decimal_places,
            keep_nearest=keep_nearest)
        datasets_out[name] = d
    return datasets_out


def enkf_irregular_updates_best_ensemble_size(
        datasets: dict, dataset_noise_variances: dict,
        volume_results: dict, dataset_ensemble_sizes: dict,
        mean_parameters: dict, parameters_ensemble_covariance: dict,
        kQ_dict: dict, kR_dict: dict,
        exp_meas_key="exp_meas_incomplete_48_72",
        state_cols=('Xv', 'mAb', 'Glc', 'Amm', 'Gln', 'Lac', 'Glu', 'Asn'),
        dt_kf=0.01, decimal_places=2):
    """EnKF where updates happen only at non-NaN measurement rows."""

    PX_records_irr  = {}
    para_records_irr = {}
    Xf_para_irr     = {}
    Xf_state_irr    = {}
    Xpost_irr       = {}
    Z_irr           = {}
    sim_irr         = {}
    runtime_irr     = {}
    upd_times_irr   = {}

    for dataset_name, data in datasets.items():
        print(f"\nIrregular EnKF: {dataset_name}")
        Fin     = data["schedule"]["Fin"].values
        Fout    = data["schedule"]["Fout"].values
        Fin_glc = data["schedule"]["Fin_glc"].values
        V       = volume_results[dataset_name]
        feed    = data["feed"].set_index("Metabolite")["Concentration (mM)"].to_dict()

        exp_used   = data[exp_meas_key]
        t_obs      = exp_used["Time (hours)"].astype(float).values
        state_init = (data["exp_meas"].iloc[0][list(state_cols)]
                      .apply(pd.to_numeric, errors="coerce").astype(float).values)

        time_steps_A = np.round(np.arange(0, len(Fin) * dt_kf, dt_kf), decimal_places)
        time_steps_B = np.round(t_obs, decimal_places)
        time_to_idxB = {}
        for iB, tB in enumerate(time_steps_B):
            time_to_idxB.setdefault(tB, iB)

        ens_size = dataset_ensemble_sizes.get(dataset_name, 50)
        kQ = kQ_dict[dataset_name]
        kR = kR_dict[dataset_name]
        Q  = kQ * np.diag(list(dataset_noise_variances[dataset_name]["process_var"].values()))
        R  = kR * np.diag(list(dataset_noise_variances[dataset_name]["obs_var"].values()))
        H  = np.eye(len(state_init))

        enkf             = EnKF_ParameterEstimation(dataset_name, datasets,
                                                     dataset_noise_variances,
                                                     dt_model=dt_kf,
                                                     mean_parameters=mean_parameters)
        enkf.PX          = []
        enkf.PX_records  = []
        enkf.para        = mean_parameters.copy()
        enkf.para_records = [mean_parameters.copy()]
        enkf.x           = state_init.copy()
        enkf.Q           = Q.copy()
        enkf.R           = R.copy()
        enkf.H           = H.copy()
        enkf.fx          = model_step
        enkf.dt          = dt_kf
        enkf.z           = (exp_used.loc[:, list(state_cols)]
                            .apply(pd.to_numeric, errors="coerce").astype(float).values)
        enkf.Z           = []
        enkf.X_forecast_for_parameters_records = []
        enkf.X_forecast_for_states_records     = []
        enkf.X_posterior_records               = []

        enkf.create_parameters_ensemble(ens_size, parameters_ensemble_covariance)
        enkf.create_states_ensemble(ens_size)
        enkf.create_noisy_observations(ens_size)
        enkf.PX_records.append(enkf.PX)

        set_EnKF  = [enkf.x.copy()]
        upd_times = []
        t_start   = time.time()

        for iA, tA in enumerate(tqdm(time_steps_A, desc=dataset_name)):
            ctrl = (Fin[iA], Fout[iA], Fin_glc[iA], V[iA], feed)
            enkf.forecast_for_states(ctrl)

            if tA in time_to_idxB:
                iB    = time_to_idxB[tA]
                z_row = enkf.z[iB, :]
                if not np.all(np.isnan(z_row)):
                    enkf.forecast_for_parameters(ctrl)
                    enkf.parameters_update(iB)
                    enkf.states_update(iB)
                    upd_times.append(float(tA))

            set_EnKF.append(enkf.x.copy())

        runtime_irr[dataset_name]   = time.time() - t_start
        upd_times_irr[dataset_name] = np.array(upd_times, dtype=float)
        sim_irr[dataset_name]       = np.asarray(set_EnKF)
        PX_records_irr[dataset_name]  = {ens_size: enkf.PX_records}
        para_records_irr[dataset_name] = {ens_size: enkf.para_records}
        Xf_para_irr[dataset_name]     = {ens_size: enkf.X_forecast_for_parameters_records}
        Xf_state_irr[dataset_name]    = {ens_size: enkf.X_forecast_for_states_records}
        Xpost_irr[dataset_name]       = {ens_size: enkf.X_posterior_records}
        Z_irr[dataset_name]           = {ens_size: enkf.Z}

    return (PX_records_irr, para_records_irr, Xf_para_irr, Xf_state_irr,
            Xpost_irr, Z_irr, sim_irr, runtime_irr, upd_times_irr)


def run_pipeline_irregular_48_72(datasets, dataset_noise_variances, volume_results,
                                  dataset_ensemble_sizes, mean_parameters,
                                  parameters_ensemble_covariance, kQ_dict, kR_dict,
                                  state_cols=('Xv', 'mAb', 'Glc', 'Amm',
                                              'Gln', 'Lac', 'Glu', 'Asn'),
                                  pattern_hours=(48, 72), dt_kf=0.01,
                                  decimal_places=2, keep_nearest=True,
                                  exp_meas_key="exp_meas_incomplete_48_72"):
    datasets_irr = attach_incomplete_measurements(
        datasets, state_cols=state_cols, pattern_hours=pattern_hours,
        decimal_places=decimal_places, keep_nearest=keep_nearest, new_key=exp_meas_key)
    results = enkf_irregular_updates_best_ensemble_size(
        datasets_irr, dataset_noise_variances, volume_results, dataset_ensemble_sizes,
        mean_parameters, parameters_ensemble_covariance, kQ_dict, kR_dict,
        exp_meas_key=exp_meas_key, state_cols=state_cols, dt_kf=dt_kf,
        decimal_places=decimal_places)
    return datasets_irr, results


# ─── Runner: parameter mean sensitivity ──────────────────────────────────────

def run_enkf_with_mean_params(mean_params_override: dict, datasets: dict,
                               dataset_noise_variances: dict, volume_results: dict,
                               dataset_ensemble_sizes: dict,
                               parameters_ensemble_covariance: dict,
                               kQ_dict: dict, kR_dict: dict):
    """Run EnKF with an overridden mean_parameters dict (for ±20% sensitivity)."""
    sim_out   = {}
    para_out  = {}
    dt_kf     = 0.01
    dec_places = 2

    for dataset_name, data in datasets.items():
        print(f"  [{dataset_name}]")
        Fin     = data["schedule"]["Fin"].values
        Fout    = data["schedule"]["Fout"].values
        Fin_glc = data["schedule"]["Fin_glc"].values
        V       = volume_results[dataset_name]
        exp_meas = data["exp_meas"]
        state_init = exp_meas.iloc[0][
            ['Xv', 'mAb', 'Glc', 'Amm', 'Gln', 'Lac', 'Glu', 'Asn']].values
        feed = data["feed"].set_index("Metabolite")["Concentration (mM)"].to_dict()

        time_steps_A = [round(i * dt_kf, dec_places) for i in range(len(Fin))]
        time_steps_B = [round(t, dec_places) for t in exp_meas["Time (hours)"].values]
        ens_size = dataset_ensemble_sizes.get(dataset_name, 50)

        kQ = kQ_dict[dataset_name]
        kR = kR_dict[dataset_name]
        Q  = kQ * np.diag(list(dataset_noise_variances[dataset_name]["process_var"].values()))
        R  = kR * np.diag(list(dataset_noise_variances[dataset_name]["obs_var"].values()))
        H  = np.eye(len(state_init))

        enkf             = EnKF_ParameterEstimation(dataset_name, datasets,
                                                     dataset_noise_variances,
                                                     dt_model=dt_kf,
                                                     mean_parameters=mean_params_override)
        enkf.PX          = []
        enkf.PX_records  = []
        enkf.para        = mean_params_override.copy()
        enkf.para_records = [mean_params_override.copy()]
        enkf.x           = state_init.copy()
        enkf.Q           = Q.copy()
        enkf.H           = H.copy()
        enkf.R           = R.copy()
        enkf.z           = exp_meas.iloc[:, 1:9].values
        enkf.Z           = []
        enkf.fx          = model_step
        enkf.X_forecast_for_parameters_records = []
        enkf.X_forecast_for_states_records     = []
        enkf.X_posterior_records               = []

        set_EnKF = [state_init.copy()]
        enkf.create_parameters_ensemble(ens_size, parameters_ensemble_covariance)
        enkf.create_states_ensemble(ens_size)
        enkf.create_noisy_observations(ens_size)
        enkf.PX_records.append(enkf.PX)

        for idx_A, step_A in enumerate(time_steps_A):
            if step_A in time_steps_B:
                idx_B = int(np.searchsorted(time_steps_B, step_A))
                enkf.forecast_for_parameters(
                    (Fin[idx_A], Fout[idx_A], Fin_glc[idx_A], V[idx_A], feed))
                enkf.parameters_update(idx_B)
            enkf.forecast_for_states(
                (Fin[idx_A], Fout[idx_A], Fin_glc[idx_A], V[idx_A], feed))
            for idx_B, step_B in enumerate(time_steps_B):
                if step_A == step_B:
                    enkf.states_update(idx_B)
            set_EnKF.append(enkf.x.copy())

        sim_out[dataset_name]  = np.array(set_EnKF)
        para_out[dataset_name] = {ens_size: enkf.para_records}

    return sim_out, para_out
