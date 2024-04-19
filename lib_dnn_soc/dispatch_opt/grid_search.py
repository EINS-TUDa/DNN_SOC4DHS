"""
Author: Andreas Bott
year: 2024
mail: andreas.bott@eins.tu-darmstadt.de or andi-bott@web.de

This module contains functions for the grid search baseline -> see grid_search.py for the main function
"""

import numpy as np
import tensorflow as tf

def setup(q_res, T_res, OPT, Scenario):
    """
    :param q_res: resolution for the power set points
    :param T_res: resolution for the temperature set points
    :param OPT: Optimiser object
    :param Scenario: Scenario object
    :return:
        q_4, T_0, T_4: grid of potential set points
        i0, j0, k0: indices of the schedule points in the grid
        cost_fn: cost function for a given set point and specified slack power q0
        cost_no_slack: cost function without the cost for the slack plant (lower bound for the cost of each set point)
        is_valid: function to check the validity of a set point (returns a boolean)
        analyse_validtiy: function to analyse the validity of a set point
            (returns a tuple of booleans for state, power up and power low violations)
    """

    # define grid of potential set points:
    pp_param = Scenario.settings['powerplant_params']
    q_4 = tf.range(pp_param['PP4']['q_min'], pp_param['PP4']['q_max'], -q_res, dtype=tf.float64)
    T_0 = tf.range(pp_param['PP0']['T_min'], pp_param['PP0']['T_max'], T_res, dtype=tf.float64)
    T_4 = tf.range(pp_param['PP4']['T_min'], pp_param['PP4']['T_max'], T_res, dtype=tf.float64)

    # schedule:
    sched = Scenario.get_schedules()
    q_sched = sched[0]
    T_sched = sched[1]

    # schedule indices:
    i0 = np.where(T_0 == T_sched[0, 1])[0][0]
    j0 = np.where(T_4 == T_sched[0, 0])[0][0]
    k0 = np.where(q_4 == q_sched[0, 0])[0][0]

    # cost function without the cost for the slack plant (lower bound for the cost of each set point)
    cost_fn_nosl = lambda t0, t4, q4: OPT.APC.cost_fn[0](q4, q_sched[0:1, 0], t4, T_sched[0:1, 0]) + \
                                      OPT.APC.cost_fn[1](q_sched[0:1, 1], q_sched[0:1, 1], t0, T_sched[0, 1])

    # compute the lower bound for all potential set points
    cost_no_slack = np.zeros((len(T_0), len(T_4), len(q_4)))
    for i, T0 in enumerate(T_0):
        for j, T4 in enumerate(T_4):
            for k, q4 in enumerate(q_4):
                cost_no_slack[i, j, k] = cost_fn_nosl(T0, T4, q4)

    # cost function for a given set point and specified slack power q0 (including the cost for the slack plant)
    cost_fn = lambda i, j, k, q0: cost_no_slack[i, j, k] + OPT.APC.cost_fn[1](q0, q_sched[0:1, 1], T_sched[0, 1],
                                                                              T_sched[0, 1])

    # define functions to check the validity of a set point dependent on the schedule
    def is_valid(state, q0):
        q_schedule = tf.tensor_scatter_nd_update(q_sched, [[0, 1]], [q0])
        *_, loss = OPT.test_boundaries(schedule=[q_schedule, T_sched], state=state)
        return loss == 0

    def analyse_validtiy(state, q0):
        q_schedule = tf.tensor_scatter_nd_update(q_sched, [[0, 1]], [q0])
        loss_state = OPT.APC.B_state_low(tf.transpose(state))
        loss_power_up = OPT.APC.B_max_sup(q_schedule)
        loss_power_low = OPT.APC.B_min_sup(q_schedule)
        return tf.reduce_any(loss_state != 0), tf.reduce_any(loss_power_up != 0), tf.reduce_any(loss_power_low != 0)


    return q_4, T_0, T_4, i0, j0, k0, cost_fn, cost_no_slack, is_valid, analyse_validtiy




class ValidSet(object):
    """
    class to collect and update the tensor of possible set points and the corresponding costs.
    The class provides methods to remove invalid points and to get the best point.

    attributes:
        cost_q_ub: tensor containing a lower bound for the costs for each set point
        valid_set: boolean tensor indicating whether a set point is valid
    """
    def __init__(self, cost_no_slack):
        self.cost_q_ub = cost_no_slack.copy()
        self.valid_set = np.ones_like(cost_no_slack, dtype=bool)

    def _update_cost(self):
        # set cost for invalid points to infinity (required for np.min to work properly in get_best_point)
        self.cost_q_ub[~self.valid_set] = np.inf

    def get_best_point(self):
        # returns the indices of the point with the lowest cost
        i, j, k = np.unravel_index(self.cost_q_ub.argmin(), self.cost_q_ub.shape)
        return i, j, k

    def set_upper_bound(self, cost_ub):
        # set points whose cost is higher than the upper bound to invalid, returns the number of valid points
        self.valid_set[self.cost_q_ub > cost_ub] = False
        self._update_cost()
        return self.get_n_valid()

    def remove_point(self, i, j, k):
        # remove a point from the valid set, returns the number of valid points
        self.valid_set[i, j, k] = False
        self._update_cost()
        return self.get_n_valid()

    def remove_slice(self, edge, dir, axis):
        # remove a slice from the valid set, returns the number of valid points
        # edge: tuple of indices for the edge of the slice
        # dir: list of strings indicating the direction of the slice removal ('lower' or 'upper')
        # axis: list of integers indicating the axis of the slice removal
        i, j, k = edge
        slices = [slice(i, i + 1), slice(j, j + 1), slice(k, k + 1)]
        for d, ax in zip(dir, axis):
            if d == 'lower':
                slices[ax] = slice(None, edge[ax] + 1, 1)
            else:
                slices[ax] = slice(edge[ax], None, 1)
        self.valid_set[slices[0], slices[1], slices[2]] = False
        self._update_cost()
        return self.get_n_valid()

    def get_n_valid(self):
        return np.sum(self.valid_set)
