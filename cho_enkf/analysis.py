"""
analysis.py
===========
Post-processing and statistical analysis of EnKF results.
"""

import numpy as np
import pandas as pd


def compute_r2(y_obs: np.ndarray, y_sim: np.ndarray) -> float:
    """
    R² = 1 - SS_res / SS_tot.
    NaN observations are ignored. Returns NaN when there are fewer than 2 valid points.
    """
    mask = ~np.isnan(y_obs)
    if mask.sum() < 2:
        return np.nan
    y_o, y_s = y_obs[mask], y_sim[mask]
    ss_res = np.sum((y_o - y_s) ** 2)
    ss_tot = np.sum((y_o - np.mean(y_o)) ** 2)
    return np.nan if ss_tot <= 0 else 1.0 - ss_res / ss_tot


def compute_r2_table(datasets: dict, ensemble_tuning: dict,
                     best_ensemble_sizes: dict, state_names: list) -> pd.DataFrame:
    """
    Compute an R² summary table for all datasets at their best ensemble size.

    Returns
    -------
    pd.DataFrame  indexed by dataset name
    """
    rows = []
    for ds_name, best_n in best_ensemble_sizes.items():
        exp_meas    = datasets[ds_name]['exp_meas']
        time_exp    = exp_meas['Time (hours)'].values
        time_indices = np.clip((time_exp * 100).astype(int), 0,
                               len(ensemble_tuning[ds_name][best_n]) - 1)
        sim_at_obs  = ensemble_tuning[ds_name][best_n][time_indices]

        r2_row = {'Dataset': ds_name, 'N': best_n}
        r2_vals = []
        for j, state in enumerate(state_names):
            y_obs = (exp_meas[state].values.astype(float)
                     if state in exp_meas.columns
                     else exp_meas.iloc[:, j + 1].values.astype(float))
            r2_v = compute_r2(y_obs, sim_at_obs[:, j])
            r2_row[state] = round(r2_v, 3) if not np.isnan(r2_v) else 'N/A'
            if not np.isnan(r2_v):
                r2_vals.append(r2_v)
        r2_row['Mean R²'] = round(np.mean(r2_vals), 3) if r2_vals else np.nan
        rows.append(r2_row)

    return pd.DataFrame(rows).set_index('Dataset')


def compute_parameter_convergence_table(dataset_name: str, ensemble_size: int,
                                         datasets: dict,
                                         PX_records_all: dict) -> pd.DataFrame | None:
    """
    Compute initial spread, final spread, and convergence (%) for each parameter.

    Returns pd.DataFrame or None if no valid records exist.
    """
    raw_records  = PX_records_all.get(dataset_name, {}).get(ensemble_size, [])
    valid_records = [r for r in raw_records if r]
    if not valid_records:
        print(f"No valid records for {dataset_name}, N={ensemble_size}")
        return None

    keys    = list(valid_records[0][0].keys())
    results = []
    for key in keys:
        spreads = [np.std([m[key] for m in rec]) for rec in valid_records]
        s0, sf  = spreads[0], spreads[-1]
        conv    = 100.0 * (s0 - sf) / s0 if s0 != 0 else 0.0
        results.append({"Parameter": key, "Initial Spread": s0,
                         "Final Spread": sf, "Convergence (%)": conv})
    df = pd.DataFrame(results)
    top10 = df.nlargest(10, "Convergence (%)")["Parameter"].tolist()
    print(f"{dataset_name}: top converging params → {top10}")
    return df


def compute_overall_convergence_table(dataset_info: list, datasets: dict,
                                       PX_records_all: dict) -> pd.DataFrame:
    """
    Aggregate convergence tables across datasets.

    Parameters
    ----------
    dataset_info : list of (dataset_name, ensemble_size)

    Returns
    -------
    pd.DataFrame  rows = parameters, columns = dataset names
    """
    overall = {}
    for ds_name, ens_size in dataset_info:
        table = compute_parameter_convergence_table(ds_name, ens_size,
                                                    datasets, PX_records_all)
        if table is not None:
            overall[ds_name] = table.set_index("Parameter")["Convergence (%)"]
    return pd.DataFrame(overall).round(1)


def get_posterior_param_matrix(dataset_name: str, ensemble_size: int,
                                PX_records: dict, parameter_keys: list) -> np.ndarray:
    """
    Extract the parameter ensemble at the final assimilation step.

    Returns
    -------
    np.ndarray  shape (N_ensemble, N_params)
    """
    records   = PX_records[dataset_name][ensemble_size]
    valid     = [r for r in records if r]
    final_step = valid[-1]
    return np.array([[member[p] for p in parameter_keys] for member in final_step])
