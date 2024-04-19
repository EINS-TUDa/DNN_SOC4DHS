"""
Author: Andreas Bott
year: 2024
mail: andreas.bott@eins.tu-darmstadt.de or andi-bott@web.de

State Estimator using the Sampling Importance Resampling (SIR) algorithm.
"""

from .stateestimator import StateEstimator
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

class SIR(StateEstimator):
    def __init__(self, n_initial_samples, n_results, **kwargs):
        super().__init__(**kwargs)
        self.n_initial_samples = n_initial_samples
        self.n_results = n_results

    def estimate_state(self, q, T_d, T_q, meas_vals, **kwargs):
        state_model = self.get_state_model(q[:, :-1], T_d, T_q)

        # sample from the prior
        demands_prior = self.d_prior_dist.sample(self.n_initial_samples)
        # calculate the state prior:
        states_prior = state_model(demands_prior)
        # calculate the weights of the samples:
        weights = self.Measurements.weight_samples(states_prior, meas_vals)
        # resample for posterior (sample with replacement):
        sample_ind = tf.random.categorical(tf.math.log(tf.expand_dims(weights, axis=0)), self.n_results)
        demands_post = tf.gather(demands_prior, sample_ind, axis=0)
        states_post = tf.gather(states_prior, sample_ind, axis=0)
        return tf.squeeze(demands_post), tf.squeeze(states_post)
