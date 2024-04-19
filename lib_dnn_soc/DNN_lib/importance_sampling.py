"""
Author: Andreas Bott
year: 2024
mail: andreas.bott@eins.tu-darmstadt.de or andi-bott@web.de


This module contains objects and functions to generate training samples using an importance sampling method.

The Theory behind this approach is described in the paper:
    "Efficient Training of Learning-Based Thermal Power Flow for 4th Generation District Heating Grids"
    by Andreas Bott, Mario Beykirch, and Florian Steinke

    currently under review at the journal "Energy", preprint available at: https://arxiv.org/abs/2403.11877
"""

import os
import numpy as np
import tensorflow as tf
from lib_dnn_soc.steady_state_modelling.steady_state_solvers import my_fixpoint_iteratror, solve_SE
from lib_dnn_soc.utility import ZeroTruncatedMultivariateNormal
# setup solvers:
FI = my_fixpoint_iteratror()


def setup_mf_sampling(d_distribution, t_distribution, SE, cycles):
    """
    Computes the proxy distribution for the mass flows at active edges in the grid

    :param d_distribution: initial demand distribution
    :param t_distribution: initial temperature distribution
    :param SE:             steady state equations object
    :param cycles:         list of cycles in the grid required for the solver
    :return:               ZeroTruncatedMultivariateNormal object representing the proxy distribution
    """
    d_mean = d_distribution.mean()
    d_cov = d_distribution.covariance()
    d_cor = np.transpose(d_cov / np.sqrt(np.diag(d_cov))) / np.sqrt(np.diag(d_cov))
    T_mean = t_distribution.mean()

    # calculate mass flow at mean demand and temperature
    solve_SE(d_mean, T_mean, SE, cycles)
    mf_mean = np.zeros_like(d_mean)
    for d in SE.demands.keys():
        mf_mean[SE.dem_ind[d]] = SE.mf[SE.find_edge[d]].numpy()

    # calculate mf at demand mean + 1 sigma; mean temperature
    d_p1_std = d_mean + d_distribution.stddev()
    solve_SE(d_p1_std, T_mean, SE, cycles)
    mf_p1_std = np.zeros_like(d_p1_std)
    for d in SE.demands.keys():
        mf_p1_std[SE.dem_ind[d]] = SE.mf[SE.find_edge[d]].numpy()

    mf_std = mf_p1_std - mf_mean
    mf_cov = tf.transpose(d_cor * mf_std) * mf_std

    # construct mf-distribution
    mf_dist = ZeroTruncatedMultivariateNormal(loc=tf.constant(mf_mean, dtype=tf.float64), scale_tril=tf.linalg.cholesky(mf_cov),
                                              validate_args=True, name='prior_mass_flow_distribution')
    return mf_dist

class ImportanceSampler(object):
    """
    Class to handle the importance sampling process for the DNN training data generation

    public functions:
    - setup:                        setup the proxy distribution for the mass flows
    - generate_training_samples:    generate training samples for the DNN
    """
    def __init__(self, d_dist, T_dist, SE, cycles, grid, results_file=None):
        self.d_dist = d_dist
        self.T_dist = T_dist
        self.SE = SE
        self.cycles = cycles
        self.grid = grid
        if results_file is not None:
            results_path = '/'.join(results_file(0).split('/')[:-1])
            if not os.path.exists(results_path):
                os.makedirs(results_path)

    def setup(self):
        self.mf_dist = setup_mf_sampling(self.d_dist, self.T_dist, self.SE, self.cycles)

    def generate_training_samples(self, n_samples, include_slack, file_spec, calc_weights=False, verbose=False):
        """
        performs IS sampling to generate training samples for the DNN
        alias for self._sampling
        :param n_samples:       number of samples to draw
        :param include_slack:   if True, the heating power is included in the demand output
        :param results_file:    expects a function with one parameter returning the file to store the data in or None.
                                If None, no data is stored
        :param calc_weights:    if True, compute the sample weights
        :param verbose:         if True, print progress to console
        """
        # alias for self.sample
        _ = self._sampling(n_samples, include_slack, verbose=verbose, results_file=file_spec, calc_weights=calc_weights)

    def _solve_single_sample(self, SE, mf, T):
        # only consider 'valid' samples in which the total consumption is larger than the total supply
        while tf.reduce_sum(self.d_dist.signs * mf) < 0:
            mf = tf.squeeze(self.mf_dist.sample(1))

        SE.load_save_state()
        for dem in SE.demands.keys():
            SE.set_active_edge_temperature(dem, T[SE.dem_ind[dem]])
        # set temperature for heating - last entry in T_vector
        ind = [k for k in SE.heatings.keys()][0]
        SE.set_active_edge_temperature(ind, T[-1])

        # solve sample
        FI.solve_mf(SE, mf, self.cycles)
        FI.solve_p(SE, self.grid)
        FI.solve_temp(SE)

    def _sampling(self, n_samples, include_slack, calc_weights, results_file, verbose):
        # setup optional variables and alias:
        if calc_weights:
            weights = np.zeros((n_samples, 2))
        if verbose:
            print_frac = n_samples // 10
        store_results = results_file is not None

        # alias:
        SE = self.SE

        # run setup if not already done
        if not hasattr(self, 'mf_dist'):
            self.setup()

        ''' sampling process: '''
        # draw samples from distributions
        mf_samples = self.mf_dist.sample(n_samples)
        T_samples = self.T_dist.sample(n_samples)


        # solve each sample
        for i, (mf, T) in enumerate(zip(mf_samples, T_samples)):
            if verbose and i % print_frac == 0:
                print(f'calculating sample {i} of {n_samples} samples')

            self._solve_single_sample(SE, mf, T)

            if calc_weights:
                # calculate demand from gridstate:
                Q_vals = SE.get_demand_from_grid(heating=False)
                weights[i, :] = [self.mf_dist.prob(mf_samples[i, :]), self.d_dist.prob(Q_vals)]

            if store_results:
                # calculate demand from gridstate and save results:
                Q_vals = SE.get_demand_from_grid(heating=include_slack)
                state = tf.concat([SE.T, SE.mf, SE.p, SE.T_end], axis=0)
                tf.print(Q_vals, T_samples[i, :], tf.transpose(state), summarize=-1,
                         output_stream='file://' + f'{results_file(i)}')

        if calc_weights:
            w = weights[:, 1] / weights[:, 0]
            w = w / np.sum(w)

            n_eff = n_samples / (1 + np.var(w, ddof=1))
            print(f'effective samplesize: {n_eff}')
            print(f'effective samplerate: {100 / (1 + np.var(w, ddof=1))}%')

        return n_eff if calc_weights else np.nan
