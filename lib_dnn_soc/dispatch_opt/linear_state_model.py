"""
Author: Andreas Bott
year: 2024
mail: andreas.bott@eins.tu-darmstadt.de or andi-bott@web.de

This module introduces a linear state model which mimics the DNN state model and can be used in the optimisation
process.
"""

import numpy as np
import tensorflow as tf
from lib_dnn_soc.DNN_lib.se_NN_lib import get_demands

class LinearStateModel(object):
    """
    This class mimics the state model by and linearises the state model around the current state.

    external methods:
        __init__(StateSimulator, inputs): initialises the linear model, inputs are the grid state for linearisation
        set_jacobian(inputs): updates the jacobian of the state model around the given input
        __call__(inputs): evalues the linear model around the given input. Returns grid state and power values
    """

    def __init__(self, StateSimulator, inputs):
        self.StateSimulator = StateSimulator
        self._set_mapping_matrix(inputs)
        self.set_jacobian(inputs)

        self._get_power_values = get_demands(self.StateSimulator.SE, add_slack_output=True)

    def set_jacobian(self, inputs):
        self.state, self.jacobian = self._calculate_jacobian(inputs)
        self.mean_inputs = tf.concat([inputs['d'], inputs['q'][:, :-1], inputs['T_d'] , inputs['T_q']], axis=1)

    def _set_mapping_matrix(self, inputs):
        self.mapping_matrix = self._compute_mapping_matrix(inputs)

    def _calculate_jacobian(self, inputs):
        # computes the local solution and jacobian of the state model
        state = self.StateSimulator.get_state(**inputs)
        SE = self.StateSimulator.SE
        power_vals = SE.get_demand_from_grid(False)
        jac = SE.evaluate_state_equations('demand and temperature jacobian',  Q_heat=power_vals, use_heating=False)
        return state, jac

    def _compute_mapping_matrix(self, inputs):
        # computes the mapping matrix for the state model
        # order: demands according to Q_heat (SE), then temperatures according to SE.demands + SE.heating
        [q, T_q, d, T_d] = inputs.values()
        n_dem = tf.shape(d)[1].numpy()
        n_pp = tf.shape(q)[1].numpy()
        n_outputdims = 2 * (n_dem + n_pp) - 1
        mask_d = np.zeros((n_outputdims, n_dem))
        mask_q = np.zeros((n_outputdims, n_pp-1))
        mask_T_d = np.zeros((n_outputdims, n_dem))
        mask_T_q = np.zeros((n_outputdims, n_pp))
        SE = self.StateSimulator.SE
        for i, ind in enumerate(SE.dem_pos):
            mask_d[ind, i] = 1
            mask_T_d[ind + n_dem + n_pp, i] = 1
        for i, ind in enumerate(SE.pp_pos[:-1]):
            mask_q[ind, i] = 1
            mask_T_q[ind + n_dem + n_pp, i] = 1
        return tf.concat([mask_d, mask_q, mask_T_d, mask_T_q], axis=1)

    @tf.function
    def __call__(self, inputs):
        c_inputs = tf.concat([inputs[0], inputs[1], inputs[2], inputs[3]], axis=1)
        state_est = self.state + tf.matmul(self.jacobian, tf.matmul(self.mapping_matrix, c_inputs-self.mean_inputs, transpose_b=True))
        state_est = tf.transpose(state_est)
        power_est = self._get_power_values(state_est)
        return state_est, power_est