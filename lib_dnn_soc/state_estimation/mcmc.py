"""
Author: Andreas Bott
year: 2024
mail: andreas.bott@eins.tu-darmstadt.de or andi-bott@web.de

State Estimator using the Hamiltonian Monte Carlo MCMC algorithm.

This Estimator is not used in the paper, as SIR is faster and sufficient for the given problem. In this sense, this is
a little bonus feature for the interested reader of the code base.
The MCMC estimator was developed and published in
    "Deep learning-enabled MCMC for probabilistic state estimation in district heating grids"
    by Andreas Bott, Mario Beykirch, and Florian Steinke, published in Applied Energy 2023
"""

from .stateestimator import StateEstimator
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

class LogProbDemand:
    def __init__(self, demand_prior, model, Measurement, measurement_values):
        """
        This class is used to evaluate each sample during the MCMC algorithm.
        __call__ returns the un-normalised log_prob for the posterior distribution over the demands.

        :param demand_prior:           tfp.distribution for prior demand, has to have a .logprob function
        :param model:                  callable mapping demand -> states
        :param Measurement:            Measurement class, has to have a .get_measurement_values function
        :param measurement_values:     measured values, same order as measurement_index
        """
        self.demand_prior = demand_prior
        self.model = model
        # measurement distribution is obtained by evaluating the measurement distribution around the measurement values
        self.meas_dist = Measurement.measurement_dist.copy(loc=tf.transpose(measurement_values))
        """ this function returns the measurement values corresponding to one demand """
        self.get_measurement_values = lambda demand: tf.gather(model(demand), Measurement.measurement_indices, axis=1)

    def __call__(self, demands):
        M = tf.map_fn(lambda d: tf.reduce_sum(
            self.meas_dist.log_prob(self.get_measurement_values(tf.expand_dims(d, axis=0))), axis=0), demands) \
            + self.demand_prior.log_prob(demands)
        return M


class MCMC(StateEstimator):
    def __init__(self, n_chains, n_burnin, n_results, **kwargs):
        super().__init__(**kwargs)
        self.n_chains = n_chains
        self.n_burnin = n_burnin
        self.n_results = n_results

    def estimate_state(self, q, T_d, T_q, meas_vals, **kwargs):
        tf.config.run_functions_eagerly(False)

        state_model = self.get_state_model(q[:, :-1], T_d, T_q)
        # initialise the logarithmic posterior probability function given demand prior and measurement values
        log_prob_demand = LogProbDemand(self.d_prior_dist, state_model, self.Measurements, meas_vals)

        ''' 
            large parts of the following code are taken from the tensorflow probability tutorial on MCMC
            https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/HamiltonianMonteCarlo
        '''
        # initialise the MCMC kernel
        adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
                tfp.mcmc.HamiltonianMonteCarlo(
                    # tfp.experimental.mcmc.PreconditionedHamiltonianMonteCarlo(
                    target_log_prob_fn=log_prob_demand,
                    num_leapfrog_steps=1,
                    step_size=0.01),
                num_adaptation_steps=int(self.n_burnin * 0.8))

        # initialise the initial state of the MCMC kernel
        initial_state = tf.Variable(self.d_prior_dist.sample(self.n_chains))
        # run the MCMC kernel
        @tf.function()
        def run_chain(n_burnin, n_results, initial_state):
            samples, kernel_results = tfp.mcmc.sample_chain(
                num_results=n_results,
                num_burnin_steps=n_burnin,
                current_state=initial_state,
                kernel=adaptive_hmc,
                trace_fn=lambda _, pkr: pkr)
            return samples, kernel_results

        samples, kernel_results = run_chain(self.n_burnin, self.n_results, initial_state)
        # return the samples - stack order: [sample (0-n_samples), chain (0-n_chains), demand (0-n_demands)]
        # stack order after sampling:
        # [chain 0 sample 0, chain 0, sample 1, ... chain 0 sample n_samples, chain 1 sample 0, ... ]
        samples = tf.concat(tf.unstack(samples, axis=1), axis=0)
        return samples, state_model(samples)



