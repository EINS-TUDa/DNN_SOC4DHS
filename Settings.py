import tensorflow as tf
import numpy as np
from lib_dnn_soc.utility import Setting

grid_settings = {
    'dem_mean': {'DEM1': 200., 'DEM2': 20, 'DEM3': 200.},
    'dem_std': {'DEM1': 0.2, 'DEM2': 0.2, 'DEM3': 0.2, 'PP4': 0.1},
    'd_prior_type': 'custom',  # dist: distance based correlation, ind: independent, design: specified in setup function
    'custom_cor': np.array([[1., 0., -0.8, 0.], [0., 1., 0., 0.], [-0.8, 0., 1., 0.], [0., 0., 0., 1.]]),
    'dem_cor_factor': 5.,
    'dem_T_min': 55.,
    'dem_T_max': 55.,
    'heat_powers': {'PP4': -240},    # excluding the slack power plant
    'heat_T_min': 75.,
    'heat_T_max': 110.,
    'Ta': 10.,
    'fix_dp': 'auto',
    'measurements': {'mf': [['PP0', 5]],
                     'T': [['NR_PP0', 5], ['NR_PP4', 5]],
                     'p': [],
                     'T_end': []},
}

scenario_settings = {
    'schedules': {
        'sched_d': tf.constant([[200., 20, 200]], dtype=tf.float64),
        'sched_T_d': tf.constant([[55., 55., 55.]], dtype=tf.float64),
        'sched_q': tf.constant([[-240, -215.51]], dtype=tf.float64),
        'sched_T_q': tf.constant([[95., 85.]], dtype=tf.float64),
    },
    'state_estimator': 'SIR',  # i.e. use estimation from scenario_settings
    'T_dem_min': 80,
    'powerplant_params': {
        'PP4': {'q1h': 3.2, 'q2h': 0.1, 'q1l': 3.2, 'q2l': 0.1, 't1h': 10., 't1l': 10.,
                'T_min': 85., 'T_max': 130, 'q_min_rel': 0.8, 'q_max_rel': 1.2},
        'PP0': {'q1h': 3.2, 'q2h': 0.1, 'q1l': 3.2, 'q2l': 0.1, 't1h': 10., 't1l': 10.,
                'T_min': 85., 'T_max': 130, 'q_min_rel': 0.8, 'q_max_rel': 1.2}},
}

DNN_settings = {
    'layers_spec': [200, 400, 400],
    'n_training': 150_000,
    'n_val': 10_000,
    'n_test': 10_000,
    'batch_size': 32,
    }

state_estimator_settings = {
    'SIR_settings': {
        'n_initial_samples': 1_000_000,
        'n_results': 1_000,
    },
    'linear_settings': {
        'n_samples': 1_000,
    },
}

OPT_settings = {
    'clip_results': True,
    'n_steps_per_opt': 'auto',
    'boundary_lambda': 10_000,
    'optimiser': 'Adam',        # currently: only adam supported.
                                # To add more optimisers: change _get_optimiser in lib_dnn_soc/optimisation/optimiser.py
    'lr': 0.01,
    'batchsize': 10,
}

setting = Setting('default_parametrisation', scenario_settings)
