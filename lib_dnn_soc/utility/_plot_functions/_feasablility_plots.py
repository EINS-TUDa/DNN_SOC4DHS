"""
Author: Andreas Bott
year: 2024
mail: andreas.bott@eins.tu-darmstadt.de or andi-bott@web.de

main functionality: _plot_valid_set
    The feasibility plot shows a 2D plot of the heat power of the two power plants. With different encodings, the plot
    shows the likelihood of any combination to appear and the likelihood of the combination to be valid.

The other functions are helper functions or used to add additional information to the plot, i.e. cost functions,
constraints, and legends.

The paper uses tpe == 'imshow'
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import scipy.stats as stats
import warnings


def _plot_valid_set(est_d, q_min, q_max, state_model, SE, OPT, td, tq, tpe, fig, axs, bw_adjust=1., space=0, **kwargs):
    slack_power, validity, q = None, None, None
    q_dist = tfd.Uniform(q_min[0, 0] + space, q_max[0, 0] - space)
    for i, d in enumerate(est_d):
        # print(i)
        # _q = tf.range(q_min[0, 0], q_max[0, 0], -0.5, dtype=tf.float64)
        _q = q_dist.sample(50)
        _slack_power, _validity = _evaluate_d_sample(_q, tf.expand_dims(d, axis=0), state_model, SE, OPT, td, tq,
                                                     **kwargs)
        slack_power = tf.concat([slack_power, _slack_power], axis=0) if slack_power is not None else _slack_power
        validity = tf.concat([validity, _validity], axis=0) if validity is not None else _validity
        q = tf.concat([q, _q], axis=0) if q is not None else _q

    if tpe == 'scatter':
        """ plots data as scatter plot """
        sns.scatterplot(x=q, y=slack_power, ax=axs, hue=validity, hue_order=[True, False], palette=['green', 'red'],
                        alpha=0.05, markers='o', s=30)

    if tpe == 'postprod':
        """ post data for appearance likelihood and validity likelihood in different axes for postprocessing """
        # code is copied from https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
        dataset_valid = np.array([q[validity], slack_power[validity]])
        KDE_valid = stats.gaussian_kde(dataset_valid)
        dataset_invalid = np.array([q[~validity], slack_power[~validity]])
        KDE_invalid = stats.gaussian_kde(dataset_invalid)
        # grid:
        X, Y = np.mgrid[q_min[0, 0].numpy() + space:q_max[0, 0].numpy() - space:100j,
               q_min[0, 1].numpy() + space:q_max[0, 1].numpy() - space:100j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z_valid = np.reshape(KDE_valid(positions).T, X.shape)
        Z_invalid = np.reshape(KDE_invalid(positions).T, X.shape)
        Z_sum = Z_valid + Z_invalid
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Z_fraction = np.nan_to_num(Z_valid / Z_sum)
            Z_fraction[Z_fraction < 0.1 * Z_sum.max()] = 0
        levels_sum = np.linspace(0, np.quantile(Z_sum, 0.99), 100)
        levels_fraction = np.linspace(0, 1, 40)
        axs[0].contourf(X, Y, Z_sum, alpha=1, cmap='gist_gray', levels=levels_sum, antialiased=True)
        axs[1].contourf(X, Y, Z_fraction, alpha=1, cmap='winter', levels=levels_fraction, antialiased=True)
        for axs in axs:
            for c in axs.collections:
                c.set_edgecolor("face")

    elif tpe == 'imshow':
        """ uses imshow to create the plot - allows to encode appearance likelihood as transparency. """
        dataset_valid = np.array([q[validity], slack_power[validity]])
        KDE_valid = stats.gaussian_kde(dataset_valid)
        dataset_invalid = np.array([q[~validity], slack_power[~validity]])
        KDE_invalid = stats.gaussian_kde(dataset_invalid)
        # grid:
        X, Y = np.mgrid[q_min[0, 0].numpy() + space:q_max[0, 0].numpy() - space:100j,
               q_min[0, 1].numpy() + space:q_max[0, 1].numpy() - space:100j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z_valid = np.reshape(KDE_valid(positions).T, X.shape)
        Z_invalid = np.reshape(KDE_invalid(positions).T, X.shape)
        Z_sum = Z_valid + Z_invalid
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Z_fraction = np.nan_to_num(Z_valid / Z_sum)
            Z_fraction[Z_fraction < 0.1 * Z_sum.max()] = 0
        axs[0].imshow(Z_fraction, extent=(q_min[0, 0].numpy() + space, q_max[0, 0].numpy() - space,
                                          q_min[0, 1].numpy() + space, q_max[0, 1].numpy() - space),
                      origin='lower', cmap='winter', alpha=Z_sum / Z_sum.max(), interpolation='bilinear')

    elif tpe == 'kde':
        """ plot valid and invalid set independently in same plot using kde plots """
        sns.kdeplot(x=q, y=slack_power, hue=validity,
                    ax=axs, hue_order={True: 'valid', False: 'invalid'}, fill=True, cumulative=False,
                    palette=['green', 'red'],
                    common_norm=True, levels=50, thresh=0.001, alpha=0.7, bw_adjust=bw_adjust)
    else:
        raise ValueError('tpe must be either "scatter" or "kde"')

def _evaluate_d_sample(q, d, state_model, SE, OPT, t_d0, t_q0, slack='PP0', add_slack_constraint=False):
    n = len(q)
    # compute states and slack power:
    state = state_model([tf.repeat(d, n, axis=0), q, tf.repeat(t_d0, n, axis=0), tf.repeat(t_q0, n, axis=0)])[0]
    sl_ind = SE.find_edge[slack]
    mf = state[:, sl_ind + SE.n_nodes]
    T_in = state[:, SE.find_node[SE.edges[sl_ind]['from']]]
    T_out = state[:, SE.find_node[SE.edges[sl_ind]['to']]]
    power_sl = mf * (T_in - T_out) * 4.18

    # evaluate state w.r.t. grid constrains and pp constraints: (temperatures are not checked, as they are constant)
    # concat the power and the slack power as expected for the boundary checks
    power = tf.transpose(tf.concat([[q], [power_sl]], axis=0))
    validity = tf.reduce_all(OPT.APC.B_state_low(state) == 0, axis=1)
    if add_slack_constraint:
        validity = tf.logical_and(validity, tf.reduce_all(OPT.APC.B_min_sup(power) == 0, axis=1))
        validity = tf.logical_and(validity, tf.reduce_all(OPT.APC.B_max_sup(power) == 0, axis=1))
    return power_sl, validity


def _add_cost_to_plot(ax, cost_fn, q_min, q_max, **kwargs):
    points_x = np.linspace(q_min[0, 0], q_max[0, 0], 50)
    points_y = np.linspace(q_min[0, 1], q_max[0, 1], 50)
    cost = np.zeros((len(points_x), len(points_y)))
    for i, x in enumerate(points_x):
        for j, y in enumerate(points_y):
            cost[i, j] = cost_fn(x, y)
    ax.contour(points_x, points_y, np.transpose(cost), colors='grey', **kwargs)


def _plot_q_constraints(ax, q_min, q_max):
    ax.plot([q_min[0, 0], q_max[0, 0]], [q_max[0, 1], q_max[0, 1]], 'k--', linewidth=1.5)
    ax.plot([q_min[0, 0], q_max[0, 0]], [q_min[0, 1], q_min[0, 1]], 'k--', linewidth=1.5)
    ax.plot([q_min[0, 0], q_min[0, 0]], [q_min[0, 1], q_max[0, 1]], 'k--', linewidth=1.5)
    ax.plot([q_max[0, 0], q_max[0, 0]], [q_min[0, 1], q_max[0, 1]], 'k--', linewidth=1.5)

def _format_plot(q_min, q_max, space, fig, ax, T4, T0, label=True):
    ax.set_xlim([q_min[0, 0] + space, q_max[0, 0] - space])
    ax.set_ylim([q_min[0, 1] + space, q_max[0, 1] - space])
    if label:
        ax.set_xlabel(f'heat power plant E [kW] at {T4:.2f}°C')
        ax.set_ylabel(f'heat power plant A [kW] at {T0:.2f}°C')

def _add_legend(fig, add_colors=True, add_marker=False):
    handles = []
    if add_colors:
        handles.append(mpatches.Patch(facecolor='green', label='valid'))
        handles.append(mpatches.Patch(facecolor='red', label='invalid'))

    handles.append(mlines.Line2D([], [], color='grey', label='Cost function', linestyle='-'))
    handles.append(mlines.Line2D([], [], color='black', label='Power plant limits', linestyle='--'))
    if add_marker:
        handles.append(mlines.Line2D([], [], color='yellow', marker='*', linestyle='None',
                                     markersize=10, label='Optimised powers'))
    fig.legend(handles=handles, title='Optimisation Parameter:', loc='upper right')

def _add_marker_to_plot(ax, marker, x_lim=None, y_lim=None, color='yellow'):
    x_lim = np.array([x_lim]).T
    y_lim = np.array([y_lim]).T
    ax.errorbar(*marker, xerr=x_lim, yerr=y_lim, color=color)
    ax.plot(*marker, marker='*', markersize=10, color=color)
