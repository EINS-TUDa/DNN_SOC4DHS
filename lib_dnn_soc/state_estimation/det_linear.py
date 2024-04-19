"""
Author: Andreas Bott
year: 2024
mail: andreas.bott@eins.tu-darmstadt.de or andi-bott@web.de

Deterministic State Estimator based on linear model and Bayes theorem.
"""

from .stateestimator import StateEstimator
import tensorflow as tf
import tensorflow_probability as tfp
import lib_dnn_soc.utility as util
tfd = tfp.distributions
import numpy as np


class DeterministicLinearStateEstimator(StateEstimator):
    def __init__(self, StateSimulator, compensate_invertibility='eye', n_samples=1, **kwargs):
        """
        Deterministic State Estimator based on linear model and Bayes theorem.
        :param compensate_invertibility: The prior state covariance matrix may not be invertible.
                                            This parameter specifies how to compensate for this.
                                            'eye' adds a small value to the diagonal of the covariance matrix
                                            'None' does not compensate for invertibility
                                            'EV' compensates by setting all negative eigenvalues to 1e-6
        """
        self.compensate_invertibility = compensate_invertibility
        self.StateSimulator = StateSimulator
        self.n_samples = n_samples
        super().__init__(**kwargs)


    def estimate_state(self, q, T_d, T_q, meas_vals, **kwargs):
        # compute prior state mean and covariance
        prior_state_mean = self.StateSimulator.get_state(q=q, T_d=T_d, T_q=T_q, d=tf.expand_dims(self.d_prior_dist.mean(), axis=0))
        T = tf.Variable(prior_state_mean[0:self.SE.n_nodes])
        mf = tf.Variable(prior_state_mean[self.SE.n_nodes:self.SE.n_nodes+self.SE.n_edges])
        p = tf.Variable(prior_state_mean[self.SE.n_nodes+self.SE.n_edges:self.SE.n_nodes+self.SE.n_edges+self.SE.n_nodes])
        T_end = tf.Variable(prior_state_mean[self.SE.n_nodes+self.SE.n_edges+self.SE.n_nodes:])
        Jac = self.SE.evaluate_state_equations('demand jacobian', T=T, mf=mf, p=p, T_end=T_end)
        # jacobian initially contains the derivatives for heat supplies as well
        dem_Jac = tf.gather(Jac, self.SE.dem_pos, axis=1)
        prior_state_cov = dem_Jac @ self.d_prior_dist.covariance() @ tf.transpose(dem_Jac)
        # compensate for invertibility
        prior_state_cov = self._compensate_invertibility(prior_state_cov)

        # compute the posterior state mean and covariance
        post_state_mean, post_state_cov = self._condition_on_measurement(meas_vals, prior_state_mean, prior_state_cov)

        # compute the posterior demands based on the posterior mean state
        dem_post = np.zeros(self.SE.n_demands)
        for i, dem in enumerate(self.SE.demands.keys()):
            e_ind = self.SE.find_edge[dem]
            Tsup = post_state_mean[self.SE.find_node[self.SE.edges[e_ind]['from']]]
            mf = post_state_mean[e_ind + self.SE.n_nodes]
            Tret = post_state_mean[e_ind + 2 * self.SE.n_nodes + self.SE.n_edges]
            dem_val = self.SE.eq_demand_Q(mf, 0, Tsup, Tret)
            dem_post[self.SE.dem_ind[dem]] = tf.squeeze(dem_val)

        def get_demand_from_state(states):
            demands = np.zeros((self.SE.n_demands, len(states)))
            for i, dem in enumerate(self.SE.demands.keys()):
                e_ind = self.SE.find_edge[dem]
                Tsup = tf.gather(states, self.SE.find_node[self.SE.edges[e_ind]['from']], axis=1)
                mf = tf.gather(states, e_ind + self.SE.n_nodes, axis=1)
                Tret = tf.gather(states, e_ind + 2 * self.SE.n_nodes + self.SE.n_edges, axis=1)
                dem_val = self.SE.eq_demand_Q(mf, 0, Tsup, Tret)
                demands[self.SE.dem_ind[dem]] = tf.squeeze(dem_val)
            return demands

        dem_post = dem_post[self.SE.dem_sign > 0]   # do not return the supply estimates
        mean_est = tf.expand_dims(dem_post, axis=0)
        if self.n_samples > 1:
            post_state_dist = util.BotchedNormalDist(mean=post_state_mean, botched_cov=post_state_cov, mask_matrix=self.SE.mask_matrix_full)
            post_state_samples = post_state_dist.sample(self.n_samples)
            post_power_samples = get_demand_from_state(post_state_samples)
            post_dem_samples = post_power_samples[self.SE.dem_sign > 0]
            post_dem_samples = tf.constant(post_dem_samples.T, dtype=tf.float64)
        else:
            post_dem_samples = mean_est
        return post_dem_samples, tf.repeat(tf.transpose(post_state_mean), self.n_samples, axis=0)

    def _compensate_invertibility(self, cov):
        if self.compensate_invertibility == 'eye':
            return cov + 1e-6 * tf.eye(cov.shape[0], dtype=tf.float64)
        elif self.compensate_invertibility == 'None':
            return cov
        elif self.compensate_invertibility == 'EV':
            # this was the way it was done in the AE Paper, however it does not work for this case
            w, v = np.linalg.eig(cov)
            w = tf.maximum(w, 1e-6)
            return v @ tf.linalg.diag(w) @ tf.transpose(v)
        else:
            raise ValueError('compensate_invertability must be "eye" or "zero"')

    def _condition_on_measurement(self, measurement_value, prior_mean, prior_cov):
        dim = tf.shape(prior_mean)[0]
        inv_prior_cov = tf.linalg.pinv(prior_cov)
        diag_entries = np.zeros(dim)
        diag_entries[self.Measurements.measurement_indices] = 1 / np.array(self.Measurements.measurement_noise)**2
        inv_meas_cov = tf.linalg.diag(diag_entries)

        # store measurement in vector of the same length as the state vector
        measurement_values_vec = np.zeros((dim, 1))
        if len(measurement_value) > 0:
            if tf.shape(measurement_value)[0] == 1:
                measurement_values_vec[self.Measurements.measurement_indices] = tf.transpose(measurement_value)
            else:
                measurement_values_vec[self.Measurements.measurement_indices] = measurement_value

        # some values are independent of the uncertain demand power, exclude them from the calculation
        # this is done via boolean masks, see lib_dnn_soc/steady_state_modelling/state_equations.py for details on masks
        M = tf.cast(tf.squeeze(tf.concat(self.SE.masks, axis=0)), tf.float64)
        prior_mean_red = tf.boolean_mask(prior_mean, M)
        inv_meas_cov = tf.boolean_mask(tf.boolean_mask(inv_meas_cov, M, axis=0), M, axis=1)
        measurement_values_vec = tf.boolean_mask(tf.cast(measurement_values_vec, tf.float64), M)

        # compute the posterior mean and covariance
        posterior_cov = tf.linalg.pinv(inv_prior_cov + inv_meas_cov)
        posterior_mean = posterior_cov @ (inv_prior_cov @ prior_mean_red + inv_meas_cov @ measurement_values_vec)

        # expand the posterior mean to the full state vector
        post_state_mean = tf.Variable(prior_mean)
        post_state_mean.assign_sub(tf.sparse.sparse_dense_matmul(tf.cast(self.SE.mask_matrix_full, tf.float64), prior_mean_red))
        post_state_mean.assign_add(tf.sparse.sparse_dense_matmul(tf.cast(self.SE.mask_matrix_full, tf.float64), posterior_mean))

        return post_state_mean, posterior_cov