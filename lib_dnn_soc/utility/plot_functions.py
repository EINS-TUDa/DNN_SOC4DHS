"""
Author: Andreas Bott
year: 2024
mail: andreas.bott@eins.tu-darmstadt.de or andi-bott@web.de

This file defines different plotting functions. It utilises some functions from the _plot_functions file and serves as
an interface to the user for these.
"""
import matplotlib
import matplotlib.lines as mLines
import matplotlib.pyplot as plt
import lib_dnn_soc.utility._plot_functions as pf
import numpy as np
import tensorflow as tf


def comparisonplot(cost_fn, schedule_orig, schedules_adj, schedules_adj_sl, power_plants, pp_params=None,
                   title=None, subtitles=None, show_boundaries=False, save=False, path=None, name=None):
    """
    This function creates a series of subplots to compare different scheduling results.
    each subplot shows the cost function for the power plants heat power and temperature as well as the
    adjusted schedules. The layout of the subplots is defined in _plot_schedule_costs.

    :param cost_fn: list of cost functions for each power plant
    :param schedules_adj: adjusted schedule
    :param schedules_adj_sl: adjusted schedule with corrected slack power
    :param power_plants: list of power plant names
    :param pp_params: dict of power plant parameters (optional, but needed if show_boundaries is True)
    :param title: title of the plot
    :param subtitles: list of subtitles for each row
    :param show_boundaries: if True, plots the boundaries of the power plants
    :param cost_sched_ind: index for the initial schedules_adj used to calculate the cost functions
    :param save: if True, saves the plot to a file
    :param path: path to the file
    :param name: name of the file
    """

    n_schedules = len(schedules_adj)
    if n_schedules == 1:
        raise Exception('deprecated function, single line plotting is currently not supported.')
    else:
        # prepare canvas and data
        fig, axs = plt.subplots(n_schedules, 2, figsize=(10, 3 * n_schedules))
        for i in range(n_schedules):
            # this function draws a single row of the plot:
            axq, axT = pf._plot_schedule_costs(cost_fn=cost_fn,
                                               schedule_orig=schedule_orig,
                                               schedule_adj=schedules_adj[i],
                                               schedule_adj_sl=schedules_adj_sl[i],
                                               power_plants=power_plants, pp_params=pp_params,
                                               axq=axs[i, 0], axT=axs[i, 1], show_boundaries=show_boundaries)
            if subtitles is not None:
                fig.text(0.01, 1-(2*i+1)/(2*n_schedules), subtitles[i], rotation='vertical', va='center', ha='center',
                         fontsize=14,)
        dashed = mLines.Line2D([0], [0], color='black', linestyle='--', label=f'Schedule')
        dotted = mLines.Line2D([0], [0], color='black', linestyle=':', label=f'realised')
        fig.legend(handles=axq.get_legend_handles_labels()[0] + [dashed] + [dotted], loc='upper right')
        # if title is not None:
        fig.suptitle(title)
        plt.tight_layout()
        plt.subplots_adjust(left=0.095)
        plt.show(block=False)
    if save:
        pf.save_plot(fig, path+name)


def plot_mf_T(models, SE, prior, T_nodes, d_modified='all', mf_edge='slack', lables_mfT=None, **kwargs):
    """
    plots the temperature at demands over the mass flow at PP0 for different thermal demands
    :param models: list of state models used to compute the values
    :param SE: StateEstimator object
    :param prior: prior schedules
    :param T_nodes: list of nodes to be plotted
    :param d_modified: list of demands to be modified
    :param kwargs:
    :return: fig, axs
    """

    def _compute_values(schedules, d_modifier, d_range, t_ind, m_ind, model):
        # computes the arrays for T and mf at spec positions for demands received by plugging d_range in d_modifier
        T_vals = np.zeros((len(d_range), len(t_ind)))
        mf_vals = np.zeros((len(d_range), len(m_ind)))
        for i, d in enumerate(d_range):
            print(f'{i}/{len(d_range)}')
            state = model(**{**schedules, **{'d': d_modifier(d)}})
            T_vals[i, :] = tf.squeeze(tf.gather(state, t_ind))
            mf_vals[i, :] = tf.squeeze(tf.gather(state, m_ind))
        return T_vals, mf_vals

    # setup:
    d_modified = SE.demands if d_modified == 'all' else d_modified
    lables_mfT = ['' for _ in range(len(d_modified))] if lables_mfT is None else lables_mfT

    def d_modifier(demand, update):
        index = SE.dem_ind[demand]
        d = prior['d']
        d = tf.tensor_scatter_nd_update(d, [[0, index]], [update])
        return d

    d_mods = [lambda x: d_modifier(d, x) for d in d_modified]

    mf_edge = list(SE.heatings.keys())[0] if mf_edge == 'slack' else mf_edge
    T_ind = [SE.find_node[n] for n in T_nodes]
    m_ind = [SE.find_edge[mf_edge] + SE.n_nodes]
    fig, axs = plt.subplots(len(T_ind), 1, sharex=True, figsize=(2 * len(T_ind) + 1, 4))

    # compute values:
    for model in models:
        first = True
        for j, d_mod in enumerate(d_mods):
            legend = True
            T_vals, mf_vals = _compute_values(prior, d_mod, np.linspace(-300, 300, 100), T_ind, m_ind, model)
            for i, ax in enumerate(axs):
                if first:
                    if legend:
                        ax.plot(mf_vals[:, 0], T_vals[:, i], linewidth=2.5, legend=lables_mfT[j])
                        legend = False
                    else:
                        ax.plot(mf_vals[:, 0], T_vals[:, i], linewidth=2.5, legend=lables_mfT[j])
                else:
                    ax.plot(mf_vals[:, 0], T_vals[:, i], linewidth=2.5, linestyle='--')
            first = False

    # add marker for the schedule:ax
    state = models[0](**prior)
    mf_0 = tf.squeeze(tf.gather(state, m_ind))
    for ax in axs:
        ax.axvline(mf_0, color='black', linestyle='--', linewidth=2.5)

    # plot formatting:
    axs[-1].set_xlabel('mass flow [kg/s]')
    axs[0].set_xlim(1.4, 2.1)
    axs[0].set_ylim(83, 85)
    axs[0].set_yticklabels([f'{int(x)}' for x in axs[0].get_yticks()])
    axs[0].set_ylabel('B [°C]')
    axs[1].set_ylim(75, 90)
    axs[0].set_ylabel('C [°C]')
    axs[2].set_ylim(90, 95)
    axs[0].set_ylabel('D [°C]')
    fig.legend()
    return fig, axs


def feasibilty_plot(est_d, cost_fn, q_min, q_max, state_model, SE, OPT, td, tq, fig, ax, marker=None, add_axis_label=True,
                    tpe='kde', save=False, titles=None, add_legend=True, add_colors=True, add_colourbar=False, **kwargs):
    """
    This function creates a plot showing the feasible set of the optimisaion task. The setpoints for both power plants
    are not independent but depend on the demands. This is indicated as a kde plot. The color indicates, whether the
    resulting grid-state is feasible or not.
    The cost function is added to the plot as a contour plot.
    The power plant constraints are also shown in the plot.
    """
    # plot the feasible set:
    space = 10    # space around the feasible set
    pf._plot_valid_set(est_d, q_min, q_max, state_model, SE, OPT, td, tq, tpe=tpe, fig=fig, axs=ax, space=space)
    axs = [ax] if type(ax) is not list else ax
    for ax in axs:
        # plot the power plant constraints:
        pf._plot_q_constraints(ax, q_min, q_max)
        # add cost function to the plot:
        pf._add_cost_to_plot(ax, cost_fn, q_min, q_max, **kwargs)
        # add marker to plot:
        if marker is not None:
            pf._add_marker_to_plot(ax, *marker, color='yellow')
            add_marker = True
        else:
            add_marker = False
        # format the plot:
        pf._format_plot(q_min, q_max, space=space, fig=fig, ax=ax, T4=tq[0, 0].numpy(), T0=tq[0, 1].numpy(), label=add_axis_label)

    if add_legend:
        pf._add_legend(fig, add_colors=add_colors, add_marker=add_marker)

    if add_colourbar == 'below':
        # add axis below the right plot:
        # fig.subplots_adjust(right=1)
        # fig.tight_layout()
        # fig, ax = plt.subplots()
        pos_axs11 = fig.axes[-1].get_position().p0
        width_axs11 = fig.axes[-1].get_position().p1[0] - pos_axs11[0]
        # cbar_ax = fig.add_axes([*(pos_axs11 + np.array([0., -.05])), width_axs11, 0.02])  #
        cbar_ax = fig.add_axes([*(pos_axs11 + np.array([0., -.17])), width_axs11, 0.04])  #
        size = cbar_ax.get_position()
        # fig.colorbar(plt.cm.ScalarMappable(norm=None, cmap='winter'), label='probability of feasible state', cax=cbar_ax, orientation='horizontal', boundaries=[0, 1])
        cbar_ax.imshow(np.vstack([np.linspace(0, 1, 250) for _ in range(10)]), cmap='winter')
        cbar_ax.set_xticks([i for i in np.linspace(0, 250, 6)])
        cbar_ax.set_xticklabels([f'{int(i*100):d}%' for i in np.linspace(0, 1, 6)])
        cbar_ax.set_yticks([])
        cbar_ax.set_position([*(pos_axs11 + np.array([0., -.17])), width_axs11, 0.1])

    # add titles to the plot:
    if titles is not None:
        for i, title in enumerate(titles):
            axs[i].set_title(title)
    # save the plot:
    if save:
        pf.save_plot(fig, 'results/figures/feasibility_plot.png')