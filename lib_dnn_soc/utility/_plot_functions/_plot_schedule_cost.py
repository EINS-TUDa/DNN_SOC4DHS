"""
Author: Andreas Bott
year: 2024
mail: andreas.bott@eins.tu-darmstadt.de or andi-bott@web.de

_plot_schedule_cost shows 1D slices of the cost function for each power plant over variation of its inputs.
The plots did not make it to the final paper and are not fully refined.
"""

import matplotlib.lines as mLines
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def _plot_schedule_costs(cost_fn, schedule_orig, schedule_adj, schedule_adj_sl, power_plants, pp_params=None,
                         axq=None, axT=None, title=None, show_boundaries=False):
    """
    Plots the cost functions and the schedule for each power plant
    :param cost_fn: list of cost functions for each power plant
    :param schedule_orig: original schedule used to calculate the cost functions
    :param schedule_adj: adjusted schedule
    :param schedule_adj_sl: adjusted schedule with corrected slack power
    :param power_plants: list of power plant names
    :param pp_params: dict of power plant parameters (optional, but needed if show_boundaries is True)
    :param axq: optional, axis for the power plot
    :param axT: optional, axis for the temperature plot
    :param title: title of the plot
    :param show_boundaries: if True, plots the boundaries of the power plants
    :return: (axQ, axT)
    """

    assert pp_params is not None if show_boundaries else True, 'if show_boundaries is True, pp_params must be provided'
    if axq is None or axT is None:
        new_fig = True
        fig, (axq, axT) = plt.subplots(1, 2, figsize=(10, 3))
    else:
        new_fig = False
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colours = prop_cycle.by_key()['color']

    plot_width_q = (-np.inf, np.inf)
    plot_width_T = (np.inf, -np.inf)
    # plot for each power plant:
    for i, pp in enumerate(power_plants):
        if pp_params is not None:
            q_min = pp_params[pp]['q_min'].numpy()
            q_max = pp_params[pp]['q_max'].numpy()
            T_min = pp_params[pp]['T_min']
            T_max = pp_params[pp]['T_max']
        else:
            q_min = schedule_adj[0, i] * 0.75
            q_max = schedule_adj[0, i] * 1.25
            T_min = schedule_adj[1, i] * 0.75
            T_max = schedule_adj[1, i] * 1.25

        # plot cost over power deviation:
        q_range = np.linspace(q_min, q_max, int(1e4))
        cost = cost_fn[i](Q_is=q_range, Q_sched=schedule_orig[0][:, i], T_is=schedule_adj[1][:, i],
                          T_sched=schedule_orig[1][:, i])
        sns.lineplot(ax=axq, x=q_range, y=cost, label=f'power plant {pp}', legend=False)
        axq.axvline(x=schedule_adj[0][:, i], linestyle='-', alpha=0.5, color=colours[i])
        axq.axvline(x=schedule_adj_sl[0][:, i], linestyle='--', alpha=0.5, color=colours[i])

        # plot cost over temperature deviation:
        T_range = np.linspace(T_min, T_max, int(1e4))
        cost = cost_fn[i](Q_is=schedule_adj[0][:, i], Q_sched=schedule_orig[0][:, i], T_is=T_range,
                          T_sched=schedule_orig[1][:, i])
        sns.lineplot(ax=axT, x=T_range, y=cost, label=f'power plant {pp}', legend=False)
        # axT.axvline(x=T_schedule[:, i], linestyle='--', alpha=0.5, color=colours[i])
        axT.axvline(x=schedule_adj[1][:, i], linestyle='-', alpha=0.5, color=colours[i])

        plot_width_q = (max(plot_width_q[0], q_min), min(plot_width_q[1], q_max))
        plot_width_T = (min(plot_width_T[0], T_min), max(plot_width_T[1], T_max))

        if show_boundaries:
            axq.axvline(x=q_min, linestyle='-', alpha=0.5, color=colours[i])
            axq.axvspan(xmin=0, xmax=q_min, alpha=0.1, color=colours[i])
            axq.axvline(x=q_max, linestyle='-', alpha=0.5, color=colours[i])
            axq.axvspan(xmin=q_max, xmax=5 * q_max, alpha=0.1, color=colours[i])
            axT.axvline(x=T_min, linestyle='-', alpha=0.5, color=colours[i])
            axT.axvspan(xmin=0, xmax=T_min, alpha=0.1, color=colours[i])
            axT.axvline(x=T_max, linestyle='-', alpha=0.5, color=colours[i])
            axT.axvspan(xmin=T_max, xmax=5 * T_max, alpha=0.1, color=colours[i])

    space = (0.9, 1.1)
    spacing = lambda boundaries: [x * y for x, y in zip(boundaries, space)]
    axq.set_xlim(spacing(plot_width_q))
    axq.set_ylabel('Cost [€]')
    axq.set_title('Cost function over power')
    axT.set_xlim(spacing(plot_width_T))
    axT.set_title('Cost function over temperature')
    if new_fig:
        dashed = mLines.Line2D([0], [0], color='black', linestyle='--', label=f'Schedule')
        dotted = mLines.Line2D([0], [0], color='black', linestyle=':', label=f'realised')
        fig.legend(handles=axq.get_legend_handles_labels()[0] + [dashed] + [dotted], loc='upper right')
        # if title is not None:
        fig.suptitle(title)
        plt.tight_layout()
        plt.show(block=False)
    return axq, axT

