"""
plotting.py
===========
All publication-quality plotting functions.

Every function accepts an optional *save_path* (Path or str).
Pass None to display without saving; pass a path to save and suppress display.
"""

import colorsys
import math
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from pathlib import Path


# ─── Helper ─────────────────────────────────────────────────────────────────

def _savefig(fig, save_path, dpi=600):
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    plt.close(fig)



def _shade(base_color, n, l_lo=0.25, l_hi=0.78):
    """Return n shades of base_color, from dark (l_lo) to light (l_hi)."""
    h, l, s = colorsys.rgb_to_hls(*mcolors.to_rgb(base_color))
    return [colorsys.hls_to_rgb(h, float(lv), min(s, 0.90))
            for lv in np.linspace(l_lo, l_hi, n)]


def _compute_state_scales(rmse_dict, eps=1e-12):
    all_rmse = np.concatenate(list(rmse_dict.values()), axis=0)
    return np.maximum(np.median(all_rmse, axis=0), eps)


# ─── Ensemble tuning ────────────────────────────────────────────────────────

def plot_rmse_variance_and_computation_time_all(rmse_results, computation_times,
                                                 datasets_to_include=None,
                                                 exclude_ensemble_sizes=None,
                                                 custom_titles=None,
                                                 weights=None,
                                                 colours=None,
                                                 save_path=None):
    """Plot normalised RMSE and runtime vs ensemble size for each dataset.

    colours : dict with keys 'rmse', 'std', 'runtime' (optional).
    """
    if exclude_ensemble_sizes is None:
        exclude_ensemble_sizes = []
    if datasets_to_include is None:
        datasets_to_include = list(rmse_results.keys())
    if custom_titles is None:
        custom_titles = {ds: ds for ds in datasets_to_include}
    if colours is None:
        colours = {"rmse": "maroon", "std": "midnightblue", "runtime": "darkgreen"}

    n_plots = len(datasets_to_include)
    ncols   = min(n_plots, 3)
    nrows   = math.ceil(n_plots / ncols)
    labels  = [f"({chr(97 + i)})" for i in range(n_plots)]

    sns.set_style("white")
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 10))
    axes = np.array(axes).flatten() if n_plots > 1 else [axes]

    for idx, ds_name in enumerate(datasets_to_include):
        ax1   = axes[idx]
        scales = _compute_state_scales(rmse_results[ds_name])
        if weights is not None:
            w = np.asarray(weights, dtype=float)
            w = w / w.sum()
        else:
            w = None

        ens_sizes, scores, stds, times = [], [], [], []
        for ens, rmse in rmse_results[ds_name].items():
            if ens in exclude_ensemble_sizes:
                continue
            rmse_norm = rmse / scales
            per_run   = (np.mean(rmse_norm, axis=1) if w is None
                         else np.sum(rmse_norm * w[None, :], axis=1))
            ens_sizes.append(ens)
            scores.append(float(np.mean(per_run)))
            stds.append(float(np.std(per_run)))
            times.append(computation_times[ds_name][ens])

        ax1.plot(ens_sizes, scores, marker="o", color=colours["rmse"],
                 markersize=10, linewidth=3, label="Overall normalised RMSE")
        ax1.plot(ens_sizes, stds, marker="s", linestyle="--", color=colours["std"],
                 markersize=10, linewidth=3, label="Std of normalised RMSE")
        ax1.set_xticks(ens_sizes)
        ax1.set_xlabel("Ensemble Size", fontsize=16, fontweight="bold", labelpad=15)
        ax1.set_ylabel("Normalised error metrics", fontsize=16, fontweight="bold", labelpad=15)
        ax1.set_title(custom_titles.get(ds_name, ds_name), fontsize=18, fontweight="bold", pad=25)
        ax1.text(-0.25, 1.03, labels[idx], transform=ax1.transAxes,
                 fontsize=16, fontweight="bold", va="center", ha="center")
        ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
        ax1.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        for lab in ax1.get_xticklabels() + ax1.get_yticklabels():
            lab.set_fontsize(16); lab.set_fontweight("bold")
        ax1.spines["bottom"].set_linewidth(2)
        ax1.spines["left"].set_linewidth(2)

        ax2 = ax1.twinx()
        ax2.plot(ens_sizes, times, marker="d", color=colours["runtime"],
                 markersize=10, linewidth=3, label="Runtime")
        ax2.set_ylabel("Runtime (s)", color=colours["runtime"], fontsize=16,
                        fontweight="bold", labelpad=15)
        ax2.tick_params(axis="y", labelcolor=colours["runtime"])
        for lab in ax2.get_yticklabels():
            lab.set_fontsize(16); lab.set_fontweight("bold")
        ax2.spines["right"].set_linewidth(2)
        ax2.grid(False)

    for j in range(n_plots, len(axes)):
        fig.delaxes(axes[j])

    legend_elements = [
        Line2D([0], [0], marker="o", color=colours["rmse"], lw=4, markersize=10,
               label="Overall normalised MAE (all states)"),
        Line2D([0], [0], marker="s", color=colours["std"], linestyle="--",
               lw=4, markersize=10, label="Std of normalised MAE"),
        Line2D([0], [0], marker="d", color=colours["runtime"], lw=4, markersize=10,
               label="Entire culture runtime"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), prop={"size": 20, "weight": "bold"},
               frameon=False)
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    plt.subplots_adjust(wspace=0.7, hspace=0.6)
    _savefig(fig, save_path)


# ─── T127 overlay ────────────────────────────────────────────────────────────

def overlay_T127_subplots_with_errorbars(simulation_results, datasets,
                                          ensemble_tuning, dataset_ensemble_sizes,
                                          state_names, axis_name, save_path=None):
    sns.set(style="white", context="talk")
    T127 = ["CHO_T127_flask_PMJ", "CHO_T127_SNS_36.5", "CHO_T127_SNS_32"]
    colours = {"CHO_T127_flask_PMJ": "dimgray",
               "CHO_T127_SNS_36.5": "purple",
               "CHO_T127_SNS_32":  "teal"}
    markers = {"CHO_T127_flask_PMJ": "o",
               "CHO_T127_SNS_36.5": "v",
               "CHO_T127_SNS_32":   "D"}
    num_states = simulation_results[T127[0]]["full_simulation"].shape[1]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']

    for i in range(num_states):
        ax = axes[i]
        ax.set_title(labels[i], fontsize=16, fontweight='bold', loc='left')
        for ds in T127:
            ens  = dataset_ensemble_sizes[ds]
            col  = colours[ds]
            mk   = markers[ds]
            enkf = ensemble_tuning[ds][ens]
            em   = datasets[ds]["exp_meas"]
            T    = np.linspace(0, len(enkf) * 0.01, len(enkf)) / 24.0
            t    = em["Time (hours)"].values / 24.0
            ax.plot(T, enkf[:, i], linestyle='-', linewidth=3, color=col)
            state = state_names[i]
            if f"{state}_std" in em.columns:
                ax.errorbar(t, em[state].values, yerr=em[f"{state}_std"].values,
                            fmt=mk, color=col, ecolor='black', elinewidth=2,
                            capsize=4, markersize=8, markeredgecolor='black')
            else:
                ax.scatter(t, em[state].values, color=col, s=80,
                           edgecolor='black', marker=mk)
        ax.set_xlabel('Time (Days)', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_ylabel(axis_name[i], fontsize=14, fontweight='bold', labelpad=10)
        ax.set_xticks(np.arange(0, T.max() + 1, 2))
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_fontsize(16); lab.set_fontweight('bold')
        ax.tick_params(axis='x', which='both', bottom=True, top=False,
                       direction='in', length=3, width=2)
        ax.tick_params(axis='y', which='both', left=True, right=False,
                       direction='in', length=3, width=2)
        for spine in ax.spines.values():
            spine.set_linewidth(2)

    for j in range(num_states, len(axes)):
        fig.delaxes(axes[j])

    legend_elements = [
        Line2D([0], [0], color=colours["CHO_T127_flask_PMJ"], lw=3,
               label="Base Cell Line A Shake Flask - EnKF"),
        Line2D([0], [0], marker="o", color=colours["CHO_T127_flask_PMJ"],
               markersize=8, linestyle='None', markeredgecolor='black',
               label="Base Cell Line A Shake Flask - Experimental"),
        Line2D([0], [0], color=colours["CHO_T127_SNS_36.5"], lw=3,
               label="Cell Line A Bioreactor 36.5°C - EnKF"),
        Line2D([0], [0], marker="v", color=colours["CHO_T127_SNS_36.5"],
               markersize=8, linestyle='None', markeredgecolor='black',
               label="Cell Line A Bioreactor 36.5°C - Experimental"),
        Line2D([0], [0], color=colours["CHO_T127_SNS_32"], lw=3,
               label="Cell Line A Bioreactor 32°C - EnKF"),
        Line2D([0], [0], marker="D", color=colours["CHO_T127_SNS_32"],
               markersize=8, linestyle='None', markeredgecolor='black',
               label="Cell Line A Bioreactor 32°C - Experimental"),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, -0.02), fontsize=14, frameon=False,
               prop={'weight': 'bold'})
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    plt.subplots_adjust(bottom=0.15)
    _savefig(fig, save_path)


# ─── GS46 overlay ────────────────────────────────────────────────────────────

def overlay_gs46_subplots_with_errorbars(simulation_results, datasets,
                                          ensemble_tuning, dataset_ensemble_sizes,
                                          state_names, axis_name, save_path=None):
    sns.set(style="white", context="talk")
    GS46 = ["CHO_GS46_F_C_Inv", "CHO_GS46_F_all", "CHO_GS46_F_all_pl40"]
    colours = {"CHO_GS46_F_C_Inv":    "darkorange",
               "CHO_GS46_F_all":      "seagreen",
               "CHO_GS46_F_all_pl40": "navy"}
    markers = {"CHO_GS46_F_C_Inv":    "X",
               "CHO_GS46_F_all":      "s",
               "CHO_GS46_F_all_pl40": "^"}
    num_states = simulation_results[GS46[0]]["full_simulation"].shape[1]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']

    for i in range(num_states):
        ax = axes[i]
        ax.set_title(labels[i], fontsize=16, fontweight='bold', loc='left')
        for ds in GS46:
            ens  = dataset_ensemble_sizes[ds]
            col  = colours[ds]
            mk   = markers[ds]
            enkf = ensemble_tuning[ds][ens]
            em   = datasets[ds]["exp_meas"]
            T    = np.linspace(0, len(enkf) * 0.01, len(enkf)) / 24.0
            t    = em["Time (hours)"].values / 24.0
            ax.plot(T, enkf[:, i], linestyle='-', linewidth=3, color=col)
            state = state_names[i]
            if f"{state}_std" in em.columns:
                ax.errorbar(t, em[state].values, yerr=em[f"{state}_std"].values,
                            fmt=mk, color=col, ecolor='black', elinewidth=2,
                            capsize=4, markersize=8, markeredgecolor='black')
            else:
                ax.scatter(t, em[state].values, color=col, s=80,
                           marker=mk, edgecolor='black')
        ax.set_xlabel('Time (Days)', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_ylabel(axis_name[i], fontsize=14, fontweight='bold', labelpad=10)
        ax.set_xticks(np.arange(0, T.max() + 1, 2))
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_fontsize(16); lab.set_fontweight('bold')
        ax.tick_params(axis='x', which='both', bottom=True, top=False,
                       direction='in', length=3, width=2)
        ax.tick_params(axis='y', which='both', left=True, right=False,
                       direction='in', length=3, width=2)
        for spine in ax.spines.values():
            spine.set_linewidth(2)

    for j in range(num_states, len(axes)):
        fig.delaxes(axes[j])

    legend_elements = [
        Line2D([0], [0], color=colours["CHO_GS46_F_C_Inv"], lw=3,
               label="Cell Line B Feed C - EnKF"),
        Line2D([0], [0], marker='X', color=colours["CHO_GS46_F_C_Inv"],
               markersize=8, linestyle='None', markeredgecolor='black',
               label="Cell Line B Feed C - Experimental"),
        Line2D([0], [0], color=colours["CHO_GS46_F_all"], lw=3,
               label="Cell Line B Feed U - EnKF"),
        Line2D([0], [0], marker='s', color=colours["CHO_GS46_F_all"],
               markersize=8, linestyle='None', markeredgecolor='black',
               label="Cell Line B Feed U - Experimental"),
        Line2D([0], [0], color=colours["CHO_GS46_F_all_pl40"], lw=3,
               label="Cell Line B Feed U+40% - EnKF"),
        Line2D([0], [0], marker='^', color=colours["CHO_GS46_F_all_pl40"],
               markersize=8, linestyle='None', markeredgecolor='black',
               label="Cell Line B Feed U+40% - Experimental"),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, -0.02), fontsize=14, frameon=False,
               prop={'weight': 'bold'})
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    plt.subplots_adjust(bottom=0.15)
    _savefig(fig, save_path)


# ─── Long-term prediction ─────────────────────────────────────────────────────

def plot_longterm_pred_ensemble_simulation_errorbar(
        simulation_results, datasets, ensemble_tuning, dataset_ensemble_sizes,
        long_term_forecasts_all, axis_name, state_names,
        dataset_colours, dataset_markers,
        selected_forecast_indices=None, save_dir=None):
    sns.set(style="white", context="talk")
    dt = 0.01
    forecast_ls = ["--", ":", "-."]

    for ds_name, ens_size in dataset_ensemble_sizes.items():
        if ds_name not in simulation_results:
            continue
        set_model = simulation_results[ds_name]["full_simulation"]
        T_model   = np.linspace(0, len(set_model) * dt, len(set_model)) / 24.0
        exp_meas  = datasets[ds_name]["exp_meas"]
        t_obs     = exp_meas["Time (hours)"].values / 24.0
        set_EnKF  = ensemble_tuning[ds_name][ens_size]
        num_states = set_EnKF.shape[1]

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']

        forecasts = long_term_forecasts_all.get(ds_name)
        if selected_forecast_indices and ds_name in selected_forecast_indices:
            fidx = sorted(selected_forecast_indices[ds_name])
        else:
            fidx = list(range(len(forecasts))) if forecasts else []

        for i in range(num_states):
            ax = axes[i]
            ax.set_title(labels[i], fontsize=16, fontweight='bold', loc='left')
            sns.lineplot(x=T_model, y=set_model[:, i], color='red', linewidth=3, ax=ax)
            sns.lineplot(x=T_model, y=set_EnKF[:, i],
                         color=dataset_colours[ds_name], linewidth=3, ax=ax)

            if forecasts is not None:
                for k, j in enumerate(fidx[:3]):
                    fc  = forecasts[j]
                    upd = len(T_model) - fc.shape[0]
                    if upd + fc.shape[0] != len(T_model):
                        continue
                    full_fc = np.concatenate((set_EnKF[:upd, i], fc[:, i]))
                    ax.plot(T_model, full_fc, linestyle=forecast_ls[k],
                            linewidth=3, color=dataset_colours[ds_name])

            state_col = state_names[i]
            if f"{state_col}_std" in exp_meas.columns:
                ax.errorbar(t_obs, exp_meas[state_col].values,
                            yerr=exp_meas[f"{state_col}_std"].values,
                            fmt=dataset_markers[ds_name],
                            markerfacecolor=dataset_colours[ds_name],
                            markeredgecolor='black', ecolor='black',
                            elinewidth=2, capsize=3, markersize=8)
            else:
                ax.scatter(t_obs, exp_meas[state_col].values,
                           s=80, color=dataset_colours[ds_name],
                           edgecolor='black', marker=dataset_markers[ds_name])

            ax.set_xlabel('Time (Days)', fontsize=14, fontweight='bold', labelpad=10)
            ax.set_ylabel(axis_name[i], fontsize=14, fontweight='bold', labelpad=10)
            ax.set_xticks(np.arange(0, T_model.max() + 1, 2))
            for lab in ax.get_xticklabels() + ax.get_yticklabels():
                lab.set_fontsize(16); lab.set_fontweight('bold')
            ax.tick_params(axis='x', which='both', bottom=True, top=False,
                           direction='in', length=3, width=2)
            ax.tick_params(axis='y', which='both', left=True, right=False,
                           direction='in', length=3, width=2)
            for spine in ax.spines.values():
                spine.set_linewidth(2)
            if ax.get_legend() is not None:
                ax.get_legend().remove()

        fig.tight_layout(rect=[0, 0, 1, 1])
        plt.subplots_adjust(bottom=0.12)

        static_leg = [
            Line2D([0], [0], color='red', lw=3, label='Kotidis 2019 Model'),
            Line2D([0], [0], marker=dataset_markers[ds_name],
                   markerfacecolor=dataset_colours[ds_name],
                   markeredgecolor='black', markersize=8, linestyle='None',
                   label='Experimental Data'),
            Line2D([0], [0], color=dataset_colours[ds_name], lw=3,
                   label='EnKF Full Trajectory'),
        ]
        leg1 = fig.legend(handles=static_leg, loc='lower center', ncol=3,
                          bbox_to_anchor=(0.5, -0.01), fontsize=14, frameon=False,
                          prop={'weight': 'bold'})
        fig.add_artist(leg1)

        if forecasts and fidx:
            fc_leg = [
                Line2D([0], [0], color=dataset_colours[ds_name],
                       lw=2, linestyle=forecast_ls[k % 3],
                       label=f'Prediction from Day {j}')
                for k, j in enumerate(fidx[:3])
            ]
            leg2 = fig.legend(handles=fc_leg, loc='lower center', ncol=3,
                              bbox_to_anchor=(0.5, -0.07), fontsize=14, frameon=False,
                              prop={'weight': 'bold'})
            fig.add_artist(leg2)

        sp = (Path(save_dir) / f"long_term_pred_{ds_name}.png") if save_dir else None
        _savefig(fig, sp)


# ─── Parameter comparison across datasets ───────────────────────────────────

def plot_parameter_comparison_across_datasets(dataset_names, ensemble_sizes,
                                               datasets, PX_records_all,
                                               latex_labels,
                                               selected_keys=None,
                                               colours=None,
                                               custom_legends=None,
                                               save_path=None):
    dash_styles = ['-', '--', '-.', ':']
    avail = list(latex_labels.keys())
    keys  = [k for k in (selected_keys or avail) if k in avail]
    if not keys:
        print("No valid parameters."); return

    num_params = len(keys)
    num_rows   = (num_params + 1) // 2
    sns.set(style="white", context="talk")
    fig, axs = plt.subplots(num_rows, 2, figsize=(16, 2.5 * num_rows))
    axs = axs.flatten()
    sub_labels = [f"({chr(97 + i)})" for i in range(num_params)]

    for i, key in enumerate(keys):
        for di, ds in enumerate(dataset_names):
            ens  = ensemble_sizes[di]
            px   = PX_records_all[ds][ens]
            t    = datasets[ds]["exp_meas"]["Time (hours)"].values / 24.0
            mins, maxs, means = [], [], []
            for rec in px:
                vals = [m[key] for m in rec]
                mins.append(min(vals)); maxs.append(max(vals))
                means.append(np.mean(vals))
            col  = colours[ds] if colours else sns.color_palette("husl", len(dataset_names))[di]
            lbl  = custom_legends[di] if custom_legends else ds
            axs[i].fill_between(t, mins, maxs, color=col, alpha=0.2,
                                 edgecolor='black', label=f'Uncertainty ({lbl})')
            axs[i].plot(t, means, linestyle=dash_styles[di % 4], color=col,
                        linewidth=2.5, label=f'Mean ({lbl})')

        axs[i].set_ylabel(latex_labels[key], fontsize=20, fontweight='bold', labelpad=10)

        axs[i].text(-0.15, 1.05, sub_labels[i], transform=axs[i].transAxes,
                    fontsize=18, fontweight='bold', va='top', ha='left')

        axs[i].tick_params(axis='both', which='major', labelsize=16, width=2)
        axs[i].ticklabel_format(style='scientific', scilimits=(-3, 3))

        axs[i].tick_params(axis='x', which='both', bottom=True, top=False,
                           direction='in', length=3, width=2, labelsize=16)
        axs[i].tick_params(axis='y', which='both', left=True, right=False,
                           direction='in', length=3, width=2, labelsize=16)

        for label in axs[i].get_xticklabels() + axs[i].get_yticklabels():
            label.set_fontsize(16)
            label.set_fontweight('bold')

        for spine in axs[i].spines.values():
            spine.set_linewidth(2.5)

    if num_params % 2 != 0:
        axs[-1].axis('off')

    for ax in axs:
        ax.set_xlabel("Time (Days)", fontsize=16, fontweight='bold', labelpad=2)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.subplots_adjust(wspace=0.3, hspace=0.6)

    handles, lbls = axs[0].get_legend_handles_labels()
    by_label = dict(zip(lbls, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='lower center',
               ncol=len(dataset_names), frameon=False,
               bbox_to_anchor=(0.5, -0.1), prop={'size': 16, 'weight': 'bold'})
    _savefig(fig, save_path, dpi=600)


# ─── Posterior correlation heatmap ───────────────────────────────────────────

def plot_posterior_param_correlation(corr_matrices, best_ensemble_sizes,
                                      parameter_keys, latex_labels,
                                      dataset_titles=None, save_path=None):
    short_labels = [latex_labels[p] for p in parameter_keys]
    if dataset_titles is None:
        dataset_titles = {ds: ds for ds in best_ensemble_sizes}

    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    axes = axes.flatten()

    for ax, (ds_name, _) in zip(axes, best_ensemble_sizes.items()):
        corr = corr_matrices[ds_name]
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, mask=mask, ax=ax, cmap='RdBu_r',
                    vmin=-1, vmax=1, center=0,
                    xticklabels=short_labels, yticklabels=short_labels,
                    square=True, linewidths=0.3,
                    cbar_kws={'shrink': 0.7, 'label': 'Pearson r'})
        ax.set_title(dataset_titles.get(ds_name, ds_name), fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', labelsize=7, rotation=90)
        ax.tick_params(axis='y', labelsize=7, rotation=0)

    plt.tight_layout()
    _savefig(fig, save_path, dpi=600)


# ─── Prior width sensitivity ─────────────────────────────────────────────────

def plot_prior_width_sensitivity_rmse(prior_width_rmse, prior_width_scales,
                                       best_ensemble_sizes, save_path=None,
                                       custom_titles=None):
    sns.set(style="white", context="talk")
    datasets_list = list(best_ensemble_sizes.keys())
    x = np.arange(len(datasets_list))
    width = 0.25
    colors = ["#4C72B0", "#55A868", "#C44E52", "#DD8452"]
    x_labels = [custom_titles.get(ds, ds) if custom_titles else ds
                for ds in datasets_list]

    fig, ax = plt.subplots(figsize=(14, 7))
    for i, scale in enumerate(prior_width_scales):
        vals = []
        for ds in datasets_list:
            v = prior_width_rmse[scale][ds]
            vals.append(sum(v.values()) / len(v) if isinstance(v, dict)
                        else float(np.mean(v)))
        ax.bar(x + (i - len(prior_width_scales) // 2) * width, vals, width,
               label=f"{scale}× prior width",
               color=colors[i % len(colors)], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=25, ha="right")
    ax.set_ylabel("Mean RMSE", fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title("Prior Width Sensitivity: Effect on EnKF RMSE",
                 fontsize=16, fontweight='bold')
    ax.legend(title="Prior width scale", fontsize=14, frameon=False,
              prop={'weight': 'bold'})
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontsize(16); lab.set_fontweight('bold')
    ax.tick_params(axis='x', which='both', bottom=True, top=False,
                   direction='in', length=3, width=2)
    ax.tick_params(axis='y', which='both', left=True, right=False,
                   direction='in', length=3, width=2)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    plt.tight_layout()
    _savefig(fig, save_path, dpi=600)


def plot_prior_width_state_profiles(prior_width_sim, prior_width_scales,
                                     best_ensemble_sizes, datasets,
                                     state_names, axis_names,
                                     dataset_colours=None, dataset_markers=None,
                                     scale_labels=None,
                                     custom_titles=None, save_dir=None,
                                     ylim_scale_overrides=None):
    sns.set(style="white", context="talk")
    if scale_labels is None:
        scale_labels = {s: f"{s}× prior width" for s in prior_width_scales}

    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']

    for ds_name in best_ensemble_sizes:
        ds_color  = dataset_colours[ds_name] if dataset_colours else 'steelblue'
        ds_marker = dataset_markers[ds_name] if dataset_markers else 'o'
        # shades from dark → light corresponding to scales in order
        shades = _shade(ds_color, len(prior_width_scales))
        scale_shade = dict(zip(prior_width_scales, shades))
        # solid for 0.5× and 1.0×; dashed for 1.5×; dotted for 2.0×
        _ls_cycle = ['-', '-', '--', ':']
        scale_ls = {s: _ls_cycle[i] for i, s in enumerate(prior_width_scales)}

        exp_meas  = datasets[ds_name]["exp_meas"]
        t_exp     = exp_meas["Time (hours)"].values / 24.0
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.ravel()

        for i, (state, ylabel) in enumerate(zip(state_names, axis_names)):
            ax = axes[i]
            ax.set_title(labels[i], fontsize=16, fontweight='bold', loc='left')

            for scale in prior_width_scales:
                sim   = prior_width_sim[scale][ds_name]
                t_sim = np.linspace(0, len(sim) * 0.01, len(sim)) / 24.0
                ax.plot(t_sim, sim[:, i], color=scale_shade[scale],
                        linestyle=scale_ls[scale], linewidth=3, zorder=3)
            std_col = f"{state}_std"
            if std_col in exp_meas.columns:
                ax.errorbar(t_exp, exp_meas[state].values,
                            yerr=exp_meas[std_col].values,
                            fmt=ds_marker,
                            color=ds_color, ecolor='black', capsize=3,
                            markeredgecolor='black',
                            markersize=8, elinewidth=2, zorder=4)
            else:
                ax.scatter(t_exp, exp_meas[state].values,
                           color=ds_color, s=80, edgecolor='black',
                           marker=ds_marker, zorder=4)
            ax.set_xlabel('Time (Days)', fontsize=14, fontweight='bold', labelpad=10)
            ax.set_ylabel(ylabel, fontsize=14, fontweight='bold', labelpad=10)
            ax.set_xticks(np.arange(0, t_exp.max() + 1, 2))
            for lab in ax.get_xticklabels() + ax.get_yticklabels():
                lab.set_fontsize(16); lab.set_fontweight('bold')
            ax.tick_params(axis='x', which='both', bottom=True, top=False,
                           direction='in', length=3, width=2)
            ax.tick_params(axis='y', which='both', left=True, right=False,
                           direction='in', length=3, width=2)
            for spine in ax.spines.values():
                spine.set_linewidth(2)

            # Optional per-(dataset, state) limit: derived from experimental data only
            if (ylim_scale_overrides and
                    ds_name in ylim_scale_overrides and
                    state in ylim_scale_overrides[ds_name]):
                exp_v = exp_meas[state].values.astype(float)
                std_c = f"{state}_std"
                if std_c in exp_meas.columns:
                    err_v = exp_meas[std_c].values.astype(float)
                    exp_lo = float(np.nanmin(exp_v - err_v))
                    exp_hi = float(np.nanmax(exp_v + err_v))
                else:
                    exp_lo = float(np.nanmin(exp_v))
                    exp_hi = float(np.nanmax(exp_v))
                span = exp_hi - exp_lo if exp_hi != exp_lo else abs(exp_hi)
                ax.set_ylim(bottom=exp_lo - span * 0.1, top=exp_hi * 1.5)

        fig.tight_layout(rect=[0, 0, 1, 1])
        plt.subplots_adjust(bottom=0.12)

        legend_handles = [
            Line2D([0], [0], color=scale_shade[s], lw=3, ls=scale_ls[s],
                   label=scale_labels[s])
            for s in prior_width_scales
        ]
        legend_handles.append(
            Line2D([0], [0], marker=ds_marker, color=ds_color,
                   markeredgecolor='black', markersize=8, linestyle='None',
                   label='Experimental Data')
        )
        fig.legend(handles=legend_handles, loc="lower center",
                   ncol=len(prior_width_scales) + 1,
                   fontsize=14, frameon=False, bbox_to_anchor=(0.5, -0.01),
                   prop={'weight': 'bold'})
        sp = (Path(save_dir) / f"prior_width_profiles_{ds_name}.png") if save_dir else None
        _savefig(fig, sp, dpi=600)


# ─── Prior covariance stability heatmap ──────────────────────────────────────

def plot_stability_heatmap(stability_flags, prior_width_scales,
                           best_ensemble_sizes, state_names, custom_titles,
                           save_path=None, x_labels=None):
    """
    Plot a 2×3 grid of binary stability heatmaps (green=stable, red=unstable).

    Parameters
    ----------
    stability_flags : {scale: {ds_name: ndarray(n_states, dtype=bool)}}
                      True = unstable, False = stable
    prior_width_scales : list of scale values (e.g. [0.5, 1.0, 1.5, 2.0])
    best_ensemble_sizes : dict mapping ds_name → ensemble size (used for ordering)
    state_names : list of str, length n_states
    custom_titles : dict mapping ds_name → display title
    save_path : Path or str or None
    """
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch

    ds_order  = list(best_ensemble_sizes.keys())
    n_ds      = len(ds_order)
    n_states  = len(state_names)
    n_scales  = len(prior_width_scales)
    scale_labels = x_labels if x_labels is not None else [f"{s}×" for s in prior_width_scales]

    # Build data array: shape (n_ds, n_states, n_scales)
    data = np.zeros((n_ds, n_states, n_scales), dtype=float)
    for si, scale in enumerate(prior_width_scales):
        for di, ds in enumerate(ds_order):
            if scale in stability_flags and ds in stability_flags[scale]:
                flags = stability_flags[scale][ds]
                data[di, :len(flags), si] = flags.astype(float)

    cmap = mcolors.ListedColormap(['#2ca02c', '#d62728'])   # green / red

    ncols = 3
    nrows = math.ceil(n_ds / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.8, nrows * n_states * 0.42 + 1.2))
    axes = np.array(axes).flatten()

    for di, ds in enumerate(ds_order):
        ax = axes[di]
        mat = data[di]                      # (n_states, n_scales)
        ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect='auto',
                  interpolation='none')

        # Grid lines
        ax.set_xticks(np.arange(-0.5, n_scales, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_states, 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=1.5)
        ax.tick_params(which='minor', length=0)

        ax.set_xticks(np.arange(n_scales))
        ax.set_xticklabels(scale_labels, fontsize=11, fontweight='bold')
        ax.set_yticks(np.arange(n_states))
        ax.set_yticklabels(state_names, fontsize=11, fontweight='bold')
        ax.set_title(custom_titles.get(ds, ds), fontsize=12,
                     fontweight='bold', pad=6)
        ax.tick_params(axis='both', which='major', length=0)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    for di in range(n_ds, len(axes)):
        axes[di].set_visible(False)

    legend_elements = [Patch(facecolor='#2ca02c', edgecolor='black', label='Stable'),
                       Patch(facecolor='#d62728', edgecolor='black', label='Unstable')]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               fontsize=12, frameon=True, bbox_to_anchor=(0.5, -0.02),
               prop={'weight': 'bold'})

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _savefig(fig, save_path, dpi=600)


# ─── Parameter mean sensitivity (±%) ─────────────────────────────────────────



def plot_param_sensitivity_comparison(datasets, perturb_sims, sim_baseline,
                                       perturbations, state_names, axis_names,
                                       dataset_ensemble_sizes,
                                       dataset_colours, dataset_markers,
                                       save_dir=None, dt_kf=0.01,
                                       ylim_scale_overrides=None):
    """
    perturb_sims : {p: {'plus': sim_dict, 'minus': sim_dict}}
                   where p is the fractional perturbation (e.g. 0.10, 0.20, 0.30)
    perturbations: ordered list of p values (same order used for colour mapping)
    """
    sns.set(style='white', context='talk')
    sub_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']

    pct_labels = {p: f"{int(p * 100)}%" for p in perturbations}

    for ds_name in dataset_ensemble_sizes:
        if ds_name not in sim_baseline:
            continue
        exp_meas = datasets[ds_name]['exp_meas']
        t_meas   = exp_meas['Time (hours)'].values / 24.0

        all_n = [sim_baseline[ds_name].shape[0]] + [
            perturb_sims[p][sign][ds_name].shape[0]
            for p in perturbations for sign in ('plus', 'minus')
        ]
        t_sim     = np.arange(max(all_n)) * dt_kf / 24.0
        t_max     = max(t_sim[-1], t_meas.max() if t_meas.size else 0)
        day_ticks = np.arange(0, int(np.ceil(t_max)) + 1, 2)

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.ravel()

        # Build per-condition shades: darkest(−max) → … → base → … → lightest(+max)
        # n_conditions = 2*k+1; _shade returns dark→light
        k = len(perturbations)
        all_shades = _shade(dataset_colours[ds_name], 2 * k + 1)
        # order: [-p_max, …, -p_min, baseline, +p_min, …, +p_max]
        minus_shade = {p: all_shades[i]     for i, p in enumerate(reversed(perturbations))}
        baseline_shade = all_shades[k]
        plus_shade  = {p: all_shades[k+1+i] for i, p in enumerate(perturbations)}
        # solid for inner perturbations; -- for largest negative, : for largest positive
        p_max = perturbations[-1]
        minus_ls = {p: ('--' if p == p_max else '-') for p in perturbations}
        plus_ls  = {p: (':' if p == p_max else '-')  for p in perturbations}

        baseline_traj = sim_baseline[ds_name]
        for i, sname in enumerate(state_names):
            ax = axes[i]
            ax.set_title(sub_labels[i], fontsize=16, fontweight='bold', loc='left')

            # Positive perturbations
            for p in perturbations:
                traj = perturb_sims[p]['plus'][ds_name]
                tax  = np.arange(traj.shape[0]) * dt_kf / 24.0
                ax.plot(tax, traj[:, i], color=plus_shade[p],
                        linestyle=plus_ls[p], linewidth=3)

            # Baseline
            traj = sim_baseline[ds_name]
            tax  = np.arange(traj.shape[0]) * dt_kf / 24.0
            ax.plot(tax, traj[:, i], color=baseline_shade,
                    linestyle='-', linewidth=3)

            # Negative perturbations
            for p in perturbations:
                traj = perturb_sims[p]['minus'][ds_name]
                tax  = np.arange(traj.shape[0]) * dt_kf / 24.0
                ax.plot(tax, traj[:, i], color=minus_shade[p],
                        linestyle=minus_ls[p], linewidth=3)

            col = sname
            if f'{col}_std' in exp_meas.columns:
                ax.errorbar(t_meas, exp_meas[col].values,
                            yerr=exp_meas[f'{col}_std'].values,
                            fmt=dataset_markers[ds_name],
                            color=dataset_colours[ds_name],
                            markeredgecolor='black', ecolor='black',
                            elinewidth=2, capsize=3, markersize=8)
            else:
                ax.scatter(t_meas, exp_meas[col].values,
                           s=80, color=dataset_colours[ds_name],
                           edgecolor='black', marker=dataset_markers[ds_name],
                           zorder=5)

            ax.set_xlabel('Time (Days)', fontsize=14, fontweight='bold', labelpad=10)
            ax.set_ylabel(axis_names[i], fontsize=14, fontweight='bold', labelpad=10)
            ax.set_xticks(day_ticks)
            for lab in ax.get_xticklabels() + ax.get_yticklabels():
                lab.set_fontsize(16); lab.set_fontweight('bold')
            ax.tick_params(axis='x', which='both', bottom=True, top=False,
                           direction='in', length=3, width=2)
            ax.tick_params(axis='y', which='both', left=True, right=False,
                           direction='in', length=3, width=2)
            for spine in ax.spines.values():
                spine.set_linewidth(2)

            # Optional per-(dataset, state) limit: derived from experimental data only
            if (ylim_scale_overrides and
                    ds_name in ylim_scale_overrides and
                    sname in ylim_scale_overrides[ds_name]):
                exp_v = exp_meas[sname].values.astype(float)
                std_c = f"{sname}_std"
                if std_c in exp_meas.columns:
                    err_v = exp_meas[std_c].values.astype(float)
                    exp_lo = float(np.nanmin(exp_v - err_v))
                    exp_hi = float(np.nanmax(exp_v + err_v))
                else:
                    exp_lo = float(np.nanmin(exp_v))
                    exp_hi = float(np.nanmax(exp_v))
                span = exp_hi - exp_lo if exp_hi != exp_lo else abs(exp_hi)
                ax.set_ylim(bottom=exp_lo - span * 0.1, top=exp_hi * 1.5)

        ds_marker = dataset_markers[ds_name]
        legend_handles = []
        for p in reversed(perturbations):
            legend_handles.append(
                Line2D([0], [0], color=minus_shade[p], lw=3, ls=minus_ls[p],
                       label=f'−{pct_labels[p]} prior mean'))
        legend_handles.append(
            Line2D([0], [0], color=baseline_shade, lw=3, ls='-',
                   label='Baseline prior mean'))
        for p in perturbations:
            legend_handles.append(
                Line2D([0], [0], color=plus_shade[p], lw=3, ls=plus_ls[p],
                       label=f'+{pct_labels[p]} prior mean'))
        legend_handles.append(
            Line2D([0], [0], marker=ds_marker, color=dataset_colours[ds_name],
                   markeredgecolor='black', markersize=8, linestyle='None',
                   label='Experimental Data'))

        ncol = len(perturbations) * 2 + 2   # +percents, baseline, -percents, experiment
        fig.tight_layout(rect=[0, 0, 1, 1])
        plt.subplots_adjust(bottom=0.12)
        fig.legend(handles=legend_handles, loc='lower center', ncol=ncol,
                   bbox_to_anchor=(0.5, -0.01), fontsize=14, frameon=False,
                   prop={'weight': 'bold'})
        sp = (Path(save_dir) / f"param_sensitivity_{ds_name}.png") if save_dir else None
        _savefig(fig, sp, dpi=600)


# ─── EnKF vs reparametrised model comparison ─────────────────────────────────

def plot_enkf_vs_reparametrised(dataset, enkf_traj, reparam_sim, nominal_sim,
                                 state_names, axis_names,
                                 dataset_colour='darkorange',
                                 dataset_marker='X',
                                 custom_title=None,
                                 save_path=None, dt_kf=0.01):
    """
    Compare EnKF state estimate against:
      - open-loop simulation with reparametrised (dataset-specific) parameters
      - open-loop nominal (Kotidis) simulation

    Parameters
    ----------
    dataset      : dict with key 'exp_meas'
    enkf_traj    : ndarray (n_steps, n_states)  — EnKF filtered trajectory
    reparam_sim  : ndarray (n_steps, n_states)  — reparametrised open-loop
    nominal_sim  : ndarray (n_steps, n_states)  — nominal Kotidis open-loop
    """
    sns.set(style='white', context='talk')
    sub_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']

    exp_meas  = dataset['exp_meas']
    t_meas    = exp_meas['Time (hours)'].values / 24.0
    n_steps   = max(enkf_traj.shape[0], reparam_sim.shape[0], nominal_sim.shape[0])
    t_sim     = np.arange(n_steps) * dt_kf / 24.0
    t_max     = max(t_sim[-1], t_meas.max() if t_meas.size else 0)
    day_ticks = np.arange(0, int(np.ceil(t_max)) + 1, 2)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()

    for i, (sname, ylabel) in enumerate(zip(state_names, axis_names)):
        ax = axes[i]
        ax.set_title(sub_labels[i], fontsize=16, fontweight='bold', loc='left')

        # Nominal simulation
        t_nom = np.arange(nominal_sim.shape[0]) * dt_kf / 24.0
        ax.plot(t_nom, nominal_sim[:, i], color='grey', linestyle=':',
                linewidth=3, label='Nominal model (Kotidis 2019)')

        # Reparametrised open-loop
        t_rep = np.arange(reparam_sim.shape[0]) * dt_kf / 24.0
        ax.plot(t_rep, reparam_sim[:, i], color='royalblue', linestyle='--',
                linewidth=3, label='Reparametrised model')

        # EnKF filtered estimate
        t_enkf = np.arange(enkf_traj.shape[0]) * dt_kf / 24.0
        ax.plot(t_enkf, enkf_traj[:, i], color='seagreen', linestyle='-',
                linewidth=3, label='EnKF estimate')

        # Experimental data
        std_col = f'{sname}_std'
        if std_col in exp_meas.columns:
            ax.errorbar(t_meas, exp_meas[sname].values,
                        yerr=exp_meas[std_col].values,
                        fmt=dataset_marker, color=dataset_colour,
                        markeredgecolor='black', ecolor='black',
                        elinewidth=2, capsize=3, markersize=8,
                        label='Experimental Data')
        else:
            ax.scatter(t_meas, exp_meas[sname].values,
                       s=80, color=dataset_colour,
                       edgecolor='black', marker=dataset_marker,
                       label='Experimental Data', zorder=5)

        ax.set_xlabel('Time (Days)', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold', labelpad=10)
        ax.set_xticks(day_ticks)
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_fontsize(16); lab.set_fontweight('bold')
        ax.tick_params(axis='x', which='both', bottom=True, top=False,
                       direction='in', length=3, width=2)
        ax.tick_params(axis='y', which='both', left=True, right=False,
                       direction='in', length=3, width=2)
        for spine in ax.spines.values():
            spine.set_linewidth(2)

    legend_handles = [
        Line2D([0], [0], color='grey',      lw=3, ls=':', label='Nominal model (Kotidis 2019)'),
        Line2D([0], [0], color='royalblue', lw=3, ls='--', label='Reparametrised model'),
        Line2D([0], [0], color='seagreen',  lw=3, ls='-',  label='EnKF estimate'),
        Line2D([0], [0], marker=dataset_marker, color=dataset_colour,
               markeredgecolor='black', markersize=8, linestyle='None',
               label='Experimental Data'),
    ]
    fig.tight_layout(rect=[0, 0, 1, 1])
    plt.subplots_adjust(bottom=0.12)
    fig.legend(handles=legend_handles, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.01), fontsize=14, frameon=False,
               prop={'weight': 'bold'})
    _savefig(fig, save_path, dpi=600)


# ─── Irregular measurement plots ─────────────────────────────────────────────

def plot_all_datasets_state_profiles(datasets, simulation_trajectories_irregular,
                                      axis_name, state_names,
                                      dataset_colours, dataset_markers,
                                      dataset_ensemble_sizes,
                                      exp_meas_key="exp_meas_incomplete_48_72",
                                      dt_kf=0.01, save_dir=None):
    """Per-dataset 2×4 irregular-measurement profiles, styled identically to
    the long-term prediction plots (same ticks, fonts, error bars, legend)."""
    import pandas as pd
    sns.set(style="white", context="talk")
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
    num_states = len(state_names)

    figs = {}
    for ds_name in dataset_ensemble_sizes:
        if ds_name not in simulation_trajectories_irregular:
            continue
        if exp_meas_key not in datasets[ds_name]:
            continue

        sim    = np.asarray(simulation_trajectories_irregular[ds_name])
        t_sim  = np.arange(sim.shape[0]) * dt_kf / 24.0
        em     = datasets[ds_name][exp_meas_key]
        t_meas = em["Time (hours)"].astype(float).values / 24.0

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()

        for i in range(num_states):
            ax = axes[i]
            ax.set_title(labels[i], fontsize=16, fontweight='bold', loc='left')

            # EnKF irregular trajectory
            ax.plot(t_sim, sim[:, i], linestyle='-', linewidth=3,
                    color=dataset_colours[ds_name])

            # Experimental data with error bars
            state_col = state_names[i]
            yval = pd.to_numeric(em[state_col], errors="coerce").values
            if f"{state_col}_std" in em.columns:
                yerr = pd.to_numeric(em[f"{state_col}_std"], errors="coerce").values
                ax.errorbar(t_meas, yval, yerr=yerr,
                            fmt=dataset_markers[ds_name],
                            markerfacecolor=dataset_colours[ds_name], markeredgecolor='black',
                            ecolor='black', elinewidth=2, capsize=3, markersize=8)
            else:
                ax.scatter(t_meas, yval, s=80, color=dataset_colours[ds_name],
                           edgecolor='black', marker=dataset_markers[ds_name])

            ax.set_xlabel('Time (Days)', fontsize=14, fontweight='bold', labelpad=10)
            ax.set_ylabel(axis_name[i], fontsize=14, fontweight='bold', labelpad=10)
            ax.set_xticks(np.arange(0, max(t_sim.max(), t_meas.max()) + 1, 2))
            for lab in ax.get_xticklabels() + ax.get_yticklabels():
                lab.set_fontsize(16); lab.set_fontweight('bold')
            ax.tick_params(axis='x', which='both', bottom=True, top=False,
                           direction='in', length=3, width=2)
            ax.tick_params(axis='y', which='both', left=True, right=False,
                           direction='in', length=3, width=2)
            for spine in ax.spines.values():
                spine.set_linewidth(2)

        for j in range(num_states, len(axes)):
            fig.delaxes(axes[j])

        fig.tight_layout(rect=[0, 0, 1, 1])
        plt.subplots_adjust(bottom=0.12)

        static_leg = [
            Line2D([0], [0], marker=dataset_markers[ds_name],
                   markerfacecolor=dataset_colours[ds_name], markeredgecolor='black',
                   markersize=8, linestyle='None',
                   label='Irregular Interval Experimental Data'),
            Line2D([0], [0], color=dataset_colours[ds_name], lw=3,
                   label='EnKF'),
        ]
        fig.legend(handles=static_leg, loc='lower center',
                   ncol=len(static_leg), bbox_to_anchor=(0.5, -0.01),
                   fontsize=14, frameon=False, prop={'weight': 'bold'})

        figs[ds_name] = fig
        sp = (Path(save_dir) / f"irregular_{ds_name}.png") if save_dir else None
        _savefig(fig, sp)
    return figs


# ─── Irregular: all six datasets overlaid on one figure ──────────────────────

def overlay_all_datasets_irregular(datasets_irregular, sim_irregular,
                                    axis_name, state_names,
                                    dataset_colours, dataset_markers,
                                    custom_titles,
                                    ds_list=None,
                                    exp_meas_key="exp_meas_incomplete_48_72",
                                    dt_kf=0.01, legend_ncol=3,
                                    save_path=None):
    """Plot a subset of datasets on a single 2×4 figure with the style of the
    overlay_T127/GS46 errorbars functions, using the 48/72 h irregular
    measurement schedule results from script 03.

    ds_list : list of dataset keys to include (default: all six).
    """
    import pandas as pd

    ALL_DS = [
        "CHO_T127_flask_PMJ",
        "CHO_T127_SNS_36.5",
        "CHO_T127_SNS_32",
        "CHO_GS46_F_C_Inv",
        "CHO_GS46_F_all",
        "CHO_GS46_F_all_pl40",
    ]

    candidates = ds_list if ds_list is not None else ALL_DS

    # Only include datasets that are present in both dicts
    datasets_present = [ds for ds in candidates
                        if ds in datasets_irregular
                        and ds in sim_irregular
                        and exp_meas_key in datasets_irregular[ds]]

    sns.set(style="white", context="talk")
    num_states = len(state_names)
    labels = [f'({chr(97 + i)})' for i in range(num_states)]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, state in enumerate(state_names):
        ax = axes[i]
        ax.set_title(labels[i], fontsize=16, fontweight='bold', loc='left')

        for ds in datasets_present:
            col = dataset_colours[ds]
            mk  = dataset_markers[ds]
            sim = np.asarray(sim_irregular[ds])
            t_sim = np.arange(sim.shape[0]) * dt_kf / 24.0   # days
            ax.plot(t_sim, sim[:, i], linestyle='-', linewidth=3, color=col)

            em     = datasets_irregular[ds][exp_meas_key]
            t_meas = em["Time (hours)"].astype(float).values / 24.0
            z_col  = pd.to_numeric(em[state], errors="coerce").values
            std_col = f"{state}_std"
            if std_col in em.columns:
                yerr = pd.to_numeric(em[std_col], errors="coerce").values
                ax.errorbar(t_meas, z_col, yerr=yerr,
                            fmt=mk, color=col, ecolor='black', elinewidth=2,
                            capsize=4, markersize=8, markeredgecolor='black')
            else:
                ax.scatter(t_meas, z_col, color=col, s=80,
                           marker=mk, edgecolor='black')

        ax.set_xlabel('Time (Days)', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_ylabel(axis_name[i], fontsize=14, fontweight='bold', labelpad=10)
        ax.set_xticks(np.arange(0, int(np.ceil(t_sim.max())) + 1, 2))
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_fontsize(16)
            lab.set_fontweight('bold')
        ax.tick_params(axis='x', which='both', bottom=True, top=False,
                       direction='in', length=3, width=2)
        ax.tick_params(axis='y', which='both', left=True, right=False,
                       direction='in', length=3, width=2)
        for spine in ax.spines.values():
            spine.set_linewidth(2)

    for j in range(num_states, len(axes)):
        fig.delaxes(axes[j])

    legend_elements = []
    for ds in datasets_present:
        col   = dataset_colours[ds]
        mk    = dataset_markers[ds]
        label = custom_titles.get(ds, ds)
        legend_elements.append(
            Line2D([0], [0], color=col, lw=3, label=f"{label} — EnKF"))
        legend_elements.append(
            Line2D([0], [0], marker=mk, color=col, markersize=8,
                   linestyle='None', markeredgecolor='black',
                   label=f"{label} — Experimental"))

    fig.legend(handles=legend_elements, loc='lower center', ncol=legend_ncol,
               bbox_to_anchor=(0.5, -0.02), fontsize=14, frameon=False,
               prop={'weight': 'bold'})
    fig.tight_layout(rect=[0, 0.14, 1, 1])
    plt.subplots_adjust(bottom=0.18)
    _savefig(fig, save_path)


# ─── Irregular: T127 + GS46 combined into one large figure ───────────────────

def overlay_irregular_combined(datasets_irregular, sim_irregular,
                                axis_name, state_names,
                                dataset_colours, dataset_markers,
                                custom_titles,
                                t127_ds, gs46_ds,
                                exp_meas_key="exp_meas_incomplete_48_72",
                                dt_kf=0.01, save_path=None):
    """Two stacked 2×4 panels in a single figure: top = Cell Line A (t127_ds),
    bottom = Cell Line B (gs46_ds).  Style matches overlay_T127/GS46 errorbars."""
    import pandas as pd

    sns.set(style="white", context="talk")
    num_states = len(state_names)
    top_labels = [f'({chr(97 + i)})' for i in range(num_states)]
    bot_labels = [f'({chr(97 + num_states + i)})' for i in range(num_states)]

    fig = plt.figure(figsize=(20, 22))
    subfigs = fig.subfigures(2, 1, hspace=0.08)

    for subfig, ds_list, panel_labels, group_title in [
        (subfigs[0], t127_ds, top_labels,  "Cell Line A"),
        (subfigs[1], gs46_ds, bot_labels, "Cell Line B"),
    ]:
        datasets_present = [ds for ds in ds_list
                            if ds in datasets_irregular
                            and ds in sim_irregular
                            and exp_meas_key in datasets_irregular[ds]]

        axes = subfig.subplots(2, 4)
        axes = axes.flatten()

        subfig.suptitle(group_title, fontsize=18, fontweight='bold', y=1.01)

        for i, state in enumerate(state_names):
            ax = axes[i]
            ax.set_title(panel_labels[i], fontsize=16, fontweight='bold', loc='left')

            for ds in datasets_present:
                col = dataset_colours[ds]
                mk  = dataset_markers[ds]
                sim = np.asarray(sim_irregular[ds])
                t_sim = np.arange(sim.shape[0]) * dt_kf / 24.0
                ax.plot(t_sim, sim[:, i], linestyle='-', linewidth=3, color=col)

                em     = datasets_irregular[ds][exp_meas_key]
                t_meas = em["Time (hours)"].astype(float).values / 24.0
                z_col  = pd.to_numeric(em[state], errors="coerce").values
                std_col = f"{state}_std"
                if std_col in em.columns:
                    yerr = pd.to_numeric(em[std_col], errors="coerce").values
                    ax.errorbar(t_meas, z_col, yerr=yerr,
                                fmt=mk, color=col, ecolor='black', elinewidth=2,
                                capsize=4, markersize=8, markeredgecolor='black')
                else:
                    ax.scatter(t_meas, z_col, color=col, s=80,
                               marker=mk, edgecolor='black')

            ax.set_xlabel('Time (Days)', fontsize=14, fontweight='bold', labelpad=10)
            ax.set_ylabel(axis_name[i], fontsize=14, fontweight='bold', labelpad=10)
            ax.set_xticks(np.arange(0, int(np.ceil(t_sim.max())) + 1, 2))
            for lab in ax.get_xticklabels() + ax.get_yticklabels():
                lab.set_fontsize(16)
                lab.set_fontweight('bold')
            ax.tick_params(axis='x', which='both', bottom=True, top=False,
                           direction='in', length=3, width=2)
            ax.tick_params(axis='y', which='both', left=True, right=False,
                           direction='in', length=3, width=2)
            for spine in ax.spines.values():
                spine.set_linewidth(2)

        legend_elements = []
        for ds in datasets_present:
            col   = dataset_colours[ds]
            mk    = dataset_markers[ds]
            label = custom_titles.get(ds, ds)
            legend_elements.append(
                Line2D([0], [0], color=col, lw=3, label=f"{label} — EnKF"))
            legend_elements.append(
                Line2D([0], [0], marker=mk, color=col, markersize=8,
                       linestyle='None', markeredgecolor='black',
                       label=f"{label} — Experimental"))

        subfig.legend(handles=legend_elements, loc='lower center', ncol=3,
                      bbox_to_anchor=(0.5, -0.04), fontsize=14, frameon=False,
                      prop={'weight': 'bold'})
        subfig.subplots_adjust(left=0.07, right=0.97, top=0.92,
                               bottom=0.18, wspace=0.45, hspace=0.55)

    _savefig(fig, save_path)
