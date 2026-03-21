"""
plotting.py
===========
All publication-quality plotting functions.

Every function accepts an optional *save_path* (Path or str).
Pass None to display without saving; pass a path to save and suppress display.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from pathlib import Path


# ─── Helper ─────────────────────────────────────────────────────────────────

def _savefig(fig, save_path, dpi=300):
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def _compute_state_scales(rmse_dict, eps=1e-12):
    all_rmse = np.concatenate(list(rmse_dict.values()), axis=0)
    return np.maximum(np.median(all_rmse, axis=0), eps)


# ─── Ensemble tuning ────────────────────────────────────────────────────────

def plot_rmse_variance_and_computation_time_all(rmse_results, computation_times,
                                                 datasets_to_include=None,
                                                 exclude_ensemble_sizes=None,
                                                 custom_titles=None,
                                                 weights=None,
                                                 save_path=None):
    """Plot normalised RMSE and runtime vs ensemble size for each dataset."""
    if exclude_ensemble_sizes is None:
        exclude_ensemble_sizes = []
    if datasets_to_include is None:
        datasets_to_include = list(rmse_results.keys())
    if custom_titles is None:
        custom_titles = {ds: ds for ds in datasets_to_include}

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

        ax1.plot(ens_sizes, scores, marker="o", color="maroon",
                 markersize=10, linewidth=3, label="Overall normalised RMSE")
        ax1.plot(ens_sizes, stds, marker="s", linestyle="--", color="midnightblue",
                 markersize=10, linewidth=3, label="Std of normalised RMSE")
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
        ax2.plot(ens_sizes, times, marker="d", color="darkgreen",
                 markersize=10, linewidth=3, label="Runtime")
        ax2.set_ylabel("Runtime (s)", color="darkgreen", fontsize=16,
                        fontweight="bold", labelpad=15)
        ax2.tick_params(axis="y", labelcolor="darkgreen")
        for lab in ax2.get_yticklabels():
            lab.set_fontsize(16); lab.set_fontweight("bold")
        ax2.spines["right"].set_linewidth(2)
        ax2.grid(False)

    for j in range(n_plots, len(axes)):
        fig.delaxes(axes[j])

    legend_elements = [
        Line2D([0], [0], marker="o", color="maroon", lw=4, markersize=10,
               label="Overall normalised RMSE (all states)"),
        Line2D([0], [0], marker="s", color="midnightblue", linestyle="--",
               lw=4, markersize=10, label="Std of normalised RMSE"),
        Line2D([0], [0], marker="d", color="darkgreen", lw=4, markersize=10,
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
            T    = np.linspace(0, len(enkf) * 0.01, len(enkf))
            t    = em["Time (hours)"].values
            ax.plot(T, enkf[:, i], linestyle='-', linewidth=3, color=col)
            state = state_names[i]
            if f"{state}_std" in em.columns:
                ax.errorbar(t, em[state].values, yerr=em[f"{state}_std"].values,
                            fmt=mk, color=col, ecolor='black', elinewidth=2,
                            capsize=4, markersize=8, markeredgecolor='black')
            else:
                ax.scatter(t, em[state].values, color=col, s=80,
                           edgecolor='black', marker=mk)
        ax.set_xlabel('Time (hours)', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_ylabel(axis_name[i], fontsize=14, fontweight='bold', labelpad=10)
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
            T    = np.linspace(0, len(enkf) * 0.01, len(enkf))
            t    = em["Time (hours)"].values
            ax.plot(T, enkf[:, i], linestyle='-', linewidth=3, color=col)
            state = state_names[i]
            if f"{state}_std" in em.columns:
                ax.errorbar(t, em[state].values, yerr=em[f"{state}_std"].values,
                            fmt=mk, color=col, ecolor='black', elinewidth=2,
                            capsize=4, markersize=8, markeredgecolor='black')
            else:
                ax.scatter(t, em[state].values, color=col, s=80,
                           marker=mk, edgecolor='black')
        ax.set_xlabel('Time (hours)', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_ylabel(axis_name[i], fontsize=14, fontweight='bold', labelpad=10)
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
        T_model   = np.linspace(0, len(set_model) * dt, len(set_model))
        exp_meas  = datasets[ds_name]["exp_meas"]
        t_obs     = exp_meas["Time (hours)"].values
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

            ax.set_xlabel('Time (hours)', fontsize=14, fontweight='bold', labelpad=10)
            ax.set_ylabel(axis_name[i], fontsize=14, fontweight='bold', labelpad=10)
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
                       label=f'Prediction from update {j}')
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
            t    = datasets[ds]["exp_meas"]["Time (hours)"].values
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
        for lab in axs[i].get_xticklabels() + axs[i].get_yticklabels():
            lab.set_fontsize(16); lab.set_fontweight('bold')
        for spine in axs[i].spines.values():
            spine.set_linewidth(2.5)

    if num_params % 2 != 0:
        axs[-1].axis('off')
    for ax in axs:
        ax.set_xlabel("Time (hours)", fontsize=16, fontweight='bold', labelpad=10)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.subplots_adjust(wspace=0.3, hspace=0.6)
    handles, lbls = axs[0].get_legend_handles_labels()
    by_label = dict(zip(lbls, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='lower center',
               ncol=len(dataset_names), frameon=False,
               bbox_to_anchor=(0.5, -0.1), prop={'size': 16, 'weight': 'bold'})
    _savefig(fig, save_path, dpi=300)


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

    plt.suptitle('Posterior Parameter Ensemble Correlation (final assimilation step)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    _savefig(fig, save_path, dpi=600)


# ─── Prior width sensitivity ─────────────────────────────────────────────────

def plot_prior_width_sensitivity_rmse(prior_width_rmse, prior_width_scales,
                                       best_ensemble_sizes, save_path=None):
    datasets_list = list(best_ensemble_sizes.keys())
    x = np.arange(len(datasets_list))
    width = 0.25
    colors = ["#4C72B0", "#55A868", "#C44E52", "#DD8452"]

    fig, ax = plt.subplots(figsize=(12, 5))
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
    ax.set_xticklabels(datasets_list, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Mean RMSE")
    ax.set_title("Prior Width Sensitivity: Effect on EnKF RMSE")
    ax.legend(title="Prior width scale")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    _savefig(fig, save_path, dpi=150)


def plot_prior_width_state_profiles(prior_width_sim, prior_width_scales,
                                     best_ensemble_sizes, datasets,
                                     state_names, axis_names,
                                     scale_colors=None, scale_labels=None,
                                     save_dir=None):
    if scale_colors is None:
        scale_colors = {s: c for s, c in zip(prior_width_scales,
                        ["#4C72B0", "#55A868", "#C44E52", "#DD8452"])}
    if scale_labels is None:
        scale_labels = {s: f"{s}× prior width" for s in prior_width_scales}

    for ds_name in best_ensemble_sizes:
        exp_meas = datasets[ds_name]["exp_meas"]
        t_exp    = exp_meas["Time (hours)"].values
        fig, axes = plt.subplots(2, 4, figsize=(18, 7))
        axes = axes.ravel()

        for i, (state, ylabel) in enumerate(zip(state_names, axis_names)):
            ax = axes[i]
            for scale in prior_width_scales:
                sim     = prior_width_sim[scale][ds_name]
                t_sim   = np.linspace(0, len(sim) * 0.01, len(sim))
                ax.plot(t_sim, sim[:, i], color=scale_colors[scale],
                        linewidth=2, label=scale_labels[scale], zorder=3)
            std_col = f"{state}_std"
            if std_col in exp_meas.columns:
                ax.errorbar(t_exp, exp_meas[state].values,
                            yerr=exp_meas[std_col].values, fmt="o",
                            color="black", ecolor="grey", capsize=3,
                            markersize=4, linewidth=1.2,
                            label="Experimental (±std)", zorder=4)
            else:
                ax.scatter(t_exp, exp_meas[state].values,
                           color="black", s=20, label="Experimental", zorder=4)
            ax.set_xlabel("Time (hours)", fontsize=9)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.tick_params(labelsize=8)
            ax.grid(linestyle="--", alpha=0.4)

        handles, lbls = axes[0].get_legend_handles_labels()
        fig.legend(handles, lbls, loc="lower center", ncol=4, fontsize=10,
                   frameon=True, bbox_to_anchor=(0.5, -0.04))
        fig.suptitle(f"Prior Width Sensitivity — {ds_name}", fontsize=13,
                     fontweight="bold")
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        sp = (Path(save_dir) / f"prior_width_profiles_{ds_name}.png") if save_dir else None
        _savefig(fig, sp, dpi=150)


# ─── Parameter mean sensitivity (±20%) ───────────────────────────────────────

SENS_COLOURS = {'+20%': 'royalblue', 'Baseline': 'seagreen', '-20%': 'tomato'}
SENS_LS      = {'+20%': '--',        'Baseline': '-',         '-20%': ':'}
SENS_LW      = 2.5


def plot_param_sensitivity_comparison(datasets, sim_plus20, sim_baseline, sim_minus20,
                                       state_names, axis_names, dataset_ensemble_sizes,
                                       dataset_colours, dataset_markers,
                                       save_dir=None, dt_kf=0.01):
    sns.set(style='white', context='talk')
    sub_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
    cases = [('+20%', sim_plus20), ('Baseline', sim_baseline), ('-20%', sim_minus20)]

    for ds_name in dataset_ensemble_sizes:
        if ds_name not in sim_baseline:
            continue
        exp_meas = datasets[ds_name]['exp_meas']
        t_meas   = exp_meas['Time (hours)'].values / 24.0
        n_steps  = max(sim_plus20[ds_name].shape[0],
                       sim_baseline[ds_name].shape[0],
                       sim_minus20[ds_name].shape[0])
        t_sim    = np.arange(n_steps) * dt_kf / 24.0
        t_max    = max(t_sim[-1], t_meas.max() if t_meas.size else 0)
        day_ticks = np.arange(0, int(np.ceil(t_max)) + 1, 2)

        fig, axes = plt.subplots(2, 4, figsize=(22, 11))
        axes = axes.ravel()

        for i, sname in enumerate(state_names):
            ax = axes[i]
            ax.set_title(sub_labels[i], fontsize=14, fontweight='bold', loc='left')
            for label, sim_dict in cases:
                traj = sim_dict[ds_name]
                tax  = np.arange(traj.shape[0]) * dt_kf / 24.0
                ax.plot(tax, traj[:, i], color=SENS_COLOURS[label],
                        linestyle=SENS_LS[label], linewidth=SENS_LW, label=label)
            col = sname
            if f'{col}_std' in exp_meas.columns:
                ax.errorbar(t_meas, exp_meas[col].values,
                            yerr=exp_meas[f'{col}_std'].values,
                            fmt=dataset_markers[ds_name],
                            color=dataset_colours[ds_name],
                            markeredgecolor='black', ecolor='black',
                            elinewidth=1.5, capsize=3, markersize=7,
                            label='Experiment')
            else:
                ax.scatter(t_meas, exp_meas[col].values,
                           s=60, color=dataset_colours[ds_name],
                           edgecolor='black', marker=dataset_markers[ds_name],
                           label='Experiment', zorder=5)
            ax.set_xlabel('Time (days)', fontsize=12, fontweight='bold')
            ax.set_ylabel(axis_names[i], fontsize=11, fontweight='bold')
            ax.set_xticks(day_ticks)
            ax.tick_params(axis='both', direction='in', length=4, width=1.5)
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
            ax.grid(True, alpha=0.25)
            if ax.get_legend() is not None:
                ax.get_legend().remove()

        legend_handles = [
            Line2D([0], [0], color=SENS_COLOURS['+20%'], lw=SENS_LW, ls='--',
                   label='EnKF +20% μ'),
            Line2D([0], [0], color=SENS_COLOURS['Baseline'], lw=SENS_LW, ls='-',
                   label='EnKF Baseline'),
            Line2D([0], [0], color=SENS_COLOURS['-20%'], lw=SENS_LW, ls=':',
                   label='EnKF −20% μ'),
            Line2D([0], [0], marker=dataset_markers[ds_name],
                   color=dataset_colours[ds_name], markeredgecolor='black',
                   markersize=8, linestyle='None', label='Experiment'),
        ]
        fig.legend(handles=legend_handles, loc='lower center', ncol=4,
                   bbox_to_anchor=(0.5, -0.02), fontsize=12, frameon=False,
                   prop={'weight': 'bold'})
        fig.suptitle(f'{ds_name}: EnKF sensitivity to ±20% mean parameter perturbation',
                     fontsize=14, fontweight='bold', y=1.01)
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        sp = (Path(save_dir) / f"param_sensitivity_{ds_name}.png") if save_dir else None
        _savefig(fig, sp, dpi=300)


# ─── Irregular measurement plots ─────────────────────────────────────────────

def plot_all_datasets_state_profiles(datasets, simulation_trajectories_irregular,
                                      exp_meas_key="exp_meas_incomplete_48_72",
                                      state_cols=('Xv', 'mAb', 'Glc', 'Amm',
                                                  'Gln', 'Lac', 'Glu', 'Asn'),
                                      dt_kf=0.01, max_cols=4,
                                      save_dir=None):
    import pandas as pd
    figs = {}
    for ds_name in datasets:
        if ds_name not in simulation_trajectories_irregular:
            continue
        if exp_meas_key not in datasets[ds_name]:
            continue
        sim    = np.asarray(simulation_trajectories_irregular[ds_name])
        t_sim  = np.arange(sim.shape[0]) * dt_kf / 24.0
        em     = datasets[ds_name][exp_meas_key]
        t_meas = em["Time (hours)"].astype(float).values / 24.0
        z      = em.loc[:, list(state_cols)].apply(pd.to_numeric, errors="coerce").astype(float).values

        n = len(state_cols)
        ncols = min(max_cols, n)
        nrows = int(np.ceil(n / ncols))
        t_max = max(t_sim.max(), t_meas.max() if len(t_meas) else 0)
        day_ticks = np.arange(0, int(np.ceil(t_max)) + 1, 2)

        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(16, 3.8 * nrows))
        axes = np.atleast_1d(axes).ravel()
        for i, col in enumerate(state_cols):
            ax = axes[i]
            ax.plot(t_sim, sim[:, i], label="EnKF state")
            ax.plot(t_meas, z[:, i], linestyle="None", marker="o", label="Measurement")
            ax.set_title(col); ax.set_xlabel("Time (days)"); ax.set_ylabel(col)
            ax.set_xticks(day_ticks); ax.grid(True, alpha=0.3)
            if i == 0: ax.legend()
        for j in range(n, len(axes)):
            axes[j].axis("off")
        fig.suptitle(f"{ds_name}: EnKF vs incomplete measurements ({exp_meas_key})",
                     y=1.01)
        fig.tight_layout()
        figs[ds_name] = fig
        sp = (Path(save_dir) / f"irregular_{ds_name}.png") if save_dir else None
        _savefig(fig, sp)
    return figs
