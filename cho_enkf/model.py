"""
model.py
========
CHO bioprocess model: volume integration and ODE-based model step (RK4/LSODA).
"""

import numpy as np
import scipy.integrate as scp
from tqdm import tqdm


# ─── Volume integration ──────────────────────────────────────────────────────

def volume_integration(init_volume: float, Fin: np.ndarray,
                       Fout: np.ndarray, step_len: np.ndarray) -> np.ndarray:
    """
    Integrate bioreactor volume over time using dV/dt = Fin - Fout.

    Parameters
    ----------
    init_volume : float
        Initial volume (L).
    Fin : np.ndarray
        Inlet flow rate at each step (L/hr).
    Fout : np.ndarray
        Outlet flow rate at each step (L/hr).
    step_len : np.ndarray
        Duration of each step (hr).

    Returns
    -------
    np.ndarray
        Volume at each time point (length = len(Fin) + 1).
    """
    def _dV(t, state, fin, fout):
        return np.array([fin - fout], dtype='float64')

    current_volume = init_volume
    volumes = [current_volume]

    for i in tqdm(range(len(Fin)), desc="Volume integration"):
        ode = scp.ode(_dV).set_integrator('lsoda', nsteps=3000)
        ode.set_initial_value(current_volume, 0.0).set_f_params(Fin[i], Fout[i])
        current_volume = float(ode.integrate(ode.t + step_len[i])[0])
        volumes.append(current_volume)

    return np.array(volumes)


def compute_volume_results(datasets: dict, initial_volumes: dict) -> dict:
    """
    Compute volume profiles for all datasets.

    Parameters
    ----------
    datasets : dict
    initial_volumes : dict  {dataset_name: float}

    Returns
    -------
    dict  {dataset_name: np.ndarray}
    """
    volume_results = {}
    for name, data in datasets.items():
        if name not in initial_volumes:
            continue
        Fin  = data["schedule"]["Fin"].values
        Fout = data["schedule"]["Fout"].values
        step = np.full(len(Fin), 0.01)
        volume_results[name] = volume_integration(initial_volumes[name], Fin, Fout, step)
        print(f"Volume integration complete: {name}")
    return volume_results


# ─── Process model ───────────────────────────────────────────────────────────

def model_params(state: np.ndarray, model_parameters: dict) -> tuple:
    """
    Compute kinetic rates from the current state and parameters.

    Returns
    -------
    (mu, mu_d, Qglc, Qamm, Qgln, Qlac, Qglu, Qasn, QmAb)
    """
    Xv, mAb, Glc, Amm, Gln, Lac, Glu, Asn = state
    p = model_parameters

    flim = (Glc / (p['Kglc'] + Glc)) * (Asn / (p['Kasn'] + Asn))
    finh = (p['KIamm'] / (p['KIamm'] + Amm)) * (p['KIlac'] / (p['KIlac'] + Lac))

    mu   = p['mu_max']   * flim * finh
    mu_d = p['mu_d_max'] * (Amm / (p['Kd_amm'] + Amm))

    Qglc = -mu / p['Yx_glc'] - p['m_glc']
    Qamm =  mu / p['Yx_amm']
    Qlac = ((mu / p['Yx_lac'] - p['Ylac_glc'] * Qglc) *
            (p['Lac_max_1'] - Lac) / p['Lac_max_1'] +
            p['m_lac'] * (p['Lac_max_2'] - Lac) / p['Lac_max_2'])
    Qglu = -mu / p['Yx_glu']
    Qasn = ((mu * (p['Yx_asp'] - p['Yx_asn'] * p['Yasn_asp'])) /
            (p['Yx_asn'] * p['Yx_asp']) *
            (p['Yasn_asp'] * p['Yasp_asn'] - 1))
    Qgln =  mu / p['Yx_gln'] + p['Ygln_amm'] * Qamm
    QmAb =  p['YmAb_mu'] * mu + p['m_mAb']

    return mu, mu_d, Qglc, Qamm, Qgln, Qlac, Qglu, Qasn, QmAb


def model_step(model_parameters: dict, current_state: np.ndarray,
               time: float, Fin_t: float, Fout_t: float,
               Fin_glc_t: float, V_t: float,
               step_len: float, feed: dict) -> np.ndarray:
    """
    Advance the model by one time step using LSODA integration.

    Parameters
    ----------
    model_parameters : dict
    current_state    : np.ndarray  shape (8,)
    time             : float       current simulation time (hr)
    Fin_t            : float       inlet flow rate (L/hr)
    Fout_t           : float       outlet flow rate (L/hr)
    Fin_glc_t        : float       glucose feed flow rate (L/hr)
    V_t              : float       current volume (L)
    step_len         : float       integration step (hr)
    feed             : dict        feed concentrations {metabolite: mM}

    Returns
    -------
    np.ndarray  updated state, shape (8,)
    """
    def _odes(t, state):
        Xv, mAb, Glc, Amm, Gln, Lac, Glu, Asn = state
        mu, mu_d, Qglc, Qamm, Qgln, Qlac, Qglu, Qasn, QmAb = model_params(state, model_parameters)

        dXv  = Xv * ((mu - mu_d) * V_t - Fin_t) / V_t
        dmAb = (QmAb * V_t * Xv - Fin_t * mAb) / V_t
        dGlc = (Fin_t * (feed["Glc"] - Glc) + Qglc * V_t * Xv
                + Fin_glc_t * feed["glc_conc_in_glc"]) / V_t
        dAmm = (Fin_t * (feed["Amm"] - Amm) + Qamm * V_t * Xv) / V_t
        dGln = (Fin_t * (feed["Gln"] - Gln) + Qgln * V_t * Xv) / V_t
        dLac = (Fin_t * (feed["Lac"] - Lac) + Qlac * V_t * Xv) / V_t
        dGlu = (Fin_t * (feed["Glu"] - Glu) + Qglu * V_t * Xv) / V_t
        dAsn = (Fin_t * (feed["Asn"] - Asn) + Qasn * V_t * Xv) / V_t

        return np.array([dXv, dmAb, dGlc, dAmm, dGln, dLac, dGlu, dAsn], dtype='float64')

    current_state = np.where(current_state <= 0, 1e-12, current_state)
    rtol = 1e-3
    atol_vector = np.maximum(np.abs(current_state) * rtol, 1e-8)

    ode = scp.ode(_odes).set_integrator('lsoda', nsteps=3000,
                                         atol=atol_vector, rtol=rtol)
    ode.set_initial_value(current_state, time)
    try:
        new_state = list(ode.integrate(ode.t + step_len))
    except Exception as exc:
        print(f"LSODA failed: {exc}")
        new_state = list(current_state)

    return np.array(new_state)


# ─── Nominal simulation ──────────────────────────────────────────────────────

def simulate_all_datasets(datasets: dict, volume_results: dict,
                           mean_parameters: dict) -> dict:
    """
    Run a forward simulation with nominal parameters for all datasets.

    Returns
    -------
    dict  {dataset_name: {"full_simulation": ndarray, "daily_simulation": ndarray}}
    """
    simulation_results = {}
    state_cols = ['Xv', 'mAb', 'Glc', 'Amm', 'Gln', 'Lac', 'Glu', 'Asn']

    for name, data in datasets.items():
        print(f"Simulating {name} ...")
        exp_meas = data["exp_meas"]
        init     = exp_meas.iloc[0][state_cols].to_dict()
        feed     = data["feed"].set_index("Metabolite")["Concentration (mM)"].to_dict()
        Fin      = data["schedule"]["Fin"].values
        Fout     = data["schedule"]["Fout"].values
        Fin_glc  = data["schedule"]["Fin_glc"].values
        V        = volume_results[name]

        state = np.array(list(init.values()))
        step_len = 0.01
        rows = [state.copy()]
        t = 0.0

        for i in tqdm(range(len(Fin)), desc=name):
            state = model_step(mean_parameters, state, t,
                               Fin[i], Fout[i], Fin_glc[i], V[i], step_len, feed)
            rows.append(state.copy())
            t += step_len

        full_sim = np.vstack(rows)

        time_exp = exp_meas["Time (hours)"].values
        idx      = np.clip((time_exp * 100).astype(int), 0, len(full_sim) - 1)

        simulation_results[name] = {
            "full_simulation":  full_sim,
            "daily_simulation": full_sim[idx],
        }
        print(f"Done: {name}")

    return simulation_results
