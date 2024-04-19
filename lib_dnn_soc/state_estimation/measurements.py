"""
Author: Andreas Bott
year: 2024
mail: andreas.bott@eins.tu-darmstadt.de or andi-bott@web.de

The measurement class mimics the measurement process by selecting the measured states and adding measurement noise.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

class Measurument(object):
    def __init__(self, installed_measurements, SE):
        self.installed_measure = installed_measurements
        measurement_indices = []
        measurement_noise = []
        for type in installed_measurements.keys():
            for ind in installed_measurements[type]:
                try:
                    jnd = SE.find_node[ind[0]]
                except KeyError:
                    jnd = SE.find_edge[ind[0]]
                if type == 'T':
                    offset = 0
                    val = SE.T[jnd]
                elif type == 'mf':
                    offset = SE.n_nodes
                    val = SE.mf[jnd]
                elif type == 'p':
                    offset = SE.n_nodes + SE.n_edges
                    val = SE.p[jnd]
                elif type == 'T_end':
                    offset = SE.n_nodes + SE.n_edges + SE.n_nodes
                    val = SE.T_end[jnd]
                else:
                    raise Exception('miss-specified measurement type')
                measurement_indices.append(jnd + offset)
                measurement_noise.append(tf.squeeze(ind[1] / 100 * val))
        self.measurement_indices = measurement_indices
        self.measurement_noise = measurement_noise
        self.measurement_dist = tfd.MultivariateNormalDiag(loc=tf.zeros(shape=(1), dtype=tf.float64),
                                                           scale_diag=measurement_noise)
    def _get_measured_states(self, state):
        return tf.gather_nd(state, indices=np.expand_dims(self.measurement_indices, axis=-1))

    def generate_measurement_values(self, state):
        m_vals_true = self._get_measured_states(state)
        m_noise = self.measurement_dist.sample(1)
        return m_vals_true + tf.transpose(m_noise)

    def weight_samples(self, states, measurement_values, normalise=True):
        # calculate the probability of the state given the measurement
        states_m_vals = tf.gather(states, self.measurement_indices, axis=1)
        sample_p_prob = self.measurement_dist.prob(states_m_vals - tf.transpose(measurement_values))
        weights = sample_p_prob
        if normalise:
            return weights / np.sum(weights)
        else:
            return weights