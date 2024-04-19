"""
Author: Andreas Bott
year: 2024
mail: andreas.bott@eins.tu-darmstadt.de or andi-bott@web.de

setup functions for different parts of the code
"""


import tensorflow as tf
import lib_dnn_soc.dispatch_opt as opt
import lib_dnn_soc.state_estimation as est
from lib_dnn_soc.state_estimation.measurements import Measurument
from lib_dnn_soc.steady_state_modelling.steady_state_solvers import SteadyStateSimulator
from lib_dnn_soc.utility._setup_functions import _parse_gridfile, _setup_SE, _calculate_d_prior_dist, \
    _calculate_T_prior_dist


def setup_prior_settings(grid_identifier, grid_specification):
    """
    This function sets up the different objects needed for the grid operation and APC
    :param grid_identifier: string identifying the grid
    :param grid_specification: dictionary containing the grid specification
    :return:
        SE: State-equations Object
        StateSimulator: StateSimulator Object, calculates the state of the grid for given heat powers and temperatures
        Measurements: Measurement Object, generates measurement values for given state
        d_prior_dist: distribution over uncertain heat powers
        T_prior_dist: distribution over uncertain temperatures
    """

    grid_file = f'./grids/{grid_identifier}.json'
    pp_ind, dem_ind, active_edges, heat_ind = _parse_gridfile(grid_file)
    SE, cycles, gird = _setup_SE(grid_file, active_edges, dem_ind, heat_ind, **grid_specification)
    d_prior_dist = _calculate_d_prior_dist(dem_ind,
                                           dem_mean={k: v['Power'] for k, v in SE.demands.items()},
                                           dem_std=grid_specification['dem_std'],
                                           d_prior_type=grid_specification['d_prior_type'],
                                           custom_cor=grid_specification['custom_cor'],
                                           dem_cor_factor=grid_specification['dem_cor_factor'],
                                           heat_powers=grid_specification['heat_powers'], include_heatings=True)
    T_prior_dist = _calculate_T_prior_dist(dem_ind, **grid_specification)
    StateSimulator = SteadyStateSimulator(SE, tf.sign([d['Power'] for d in SE.demands.values()]), cycles)
    Measurements = Measurument(grid_specification['measurements'], SE)
    return SE, StateSimulator, Measurements, d_prior_dist, T_prior_dist


def setup_training(grid_identifier, grid_specification):
    """
        This function sets up the different objects needed for the DNN training
            the function is similar to setup_prior_settings but the distributions are different

        :param grid_identifier: string identifying the grid
        :param grid_specification: dictionary containing the grid specification
        :return:
            SE: State-equations Object
            d_prior_dist: distribution over uncertain heat powers
            T_prior_dist: distribution over uncertain temperatures
        """
    grid_file = f'./grids/{grid_identifier}.json'
    pp_ind, dem_ind, active_edges, heat_ind = _parse_gridfile(grid_file)
    SE, cycles, grid = _setup_SE(grid_file, active_edges, dem_ind, heat_ind, **grid_specification)
    d_prior_dist = _calculate_d_prior_dist(SE.dem_ind,
                                           dem_mean={k: v['Power'] for k, v in SE.demands.items()},
                                           dem_std=grid_specification['dem_std'],
                                           d_prior_type=grid_specification['d_prior_type'],
                                           custom_cor=grid_specification['custom_cor'],
                                           dem_cor_factor=grid_specification['dem_cor_factor'],
                                           heat_powers=grid_specification['heat_powers'], include_heatings=True)
    # extend the pior distribution to include the heating values as well:
    heat_ind = {key: value for key, value in SE.dem_ind.items() if not key in dem_ind.keys()}
    T_prior_dist = _calculate_T_prior_dist(SE.dem_ind, heat_ind=heat_ind, **grid_specification, include_heating=True,
                                           slack=SE.heatings)
    return SE, d_prior_dist, T_prior_dist, cycles, grid


def setup_scenario(Scenario, d_prior_dist, T_prior_dist, SE, Measurements, state_model, Results,
                   state_estimator_settings, OPT_settings, verbose):
    """
    This function initialises the scenario and sets up the state estimation and optimisation objects
    :param Scenario: Scenario Object
    :param d_prior_dist: distribution over uncertain heat powers
    :param T_prior_dist: distribution over uncertain temperatures
    :param SE: State-equations Object
    :param Measurements: Measurement Object
    :param state_model: string, state model used for the state estimation
    :param Results: Results Object
    :param state_estimator_settings: dictionary containing the settings for the state estimation
    :param OPT_settings: dictionary containing the settings for the optimisation
    :param verbose: boolean, whether to print additional information
    :return:
        schedule: dictionary containing the schedules
        dhs_prior_schedules: dictionary containing the prior schedules
        EST: State-Estimation Object
        OPT: Optimisation Object
    :return:
    """

    Scenario.save_to_log(Results, fmt='dict', end='\n')
    schedule = Scenario.get_schedules()
    dhs_prior_schedules = Scenario.get_dhs_inputs()
    shared_params = {'SE': SE, 'state_model': state_model, 'verbose': verbose,
                     'd_prior_dist': d_prior_dist, 'T_prior_dist': T_prior_dist}
    # setup functions:
    EST = est.state_estimation_selector(**Scenario.settings, **state_estimator_settings, Measurements=Measurements,
                                        **shared_params)
    OPT = opt.Optimiser(**Scenario.settings, **OPT_settings, **Scenario.settings['schedules'], **shared_params)
    return schedule, dhs_prior_schedules, EST, OPT