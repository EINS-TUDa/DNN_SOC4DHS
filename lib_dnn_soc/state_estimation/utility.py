"""
Author: Andreas Bott
year: 2024
mail: andreas.bott@eins.tu-darmstadt.de or andi-bott@web.de

This module features the base class for the state Estimators and a function to select the state estiator based on a
string identifier (given in the settings file).
"""

# %%
from lib_dnn_soc.state_estimation.det_linear import DeterministicLinearStateEstimator
from lib_dnn_soc.state_estimation.mcmc import MCMC
from lib_dnn_soc.state_estimation.sir import SIR
from lib_dnn_soc.state_estimation.settings_estimator import Settings_state_estimator


def state_estimation_selector(state_estimator, **kwargs):
    state_estimators = {
        'linear': DeterministicLinearStateEstimator,
        'MCMC': MCMC,
        'SIR': SIR,
        'Settings': Settings_state_estimator
    }
    if state_estimator in state_estimators.keys():
        return state_estimators[state_estimator](**kwargs.get(f'{state_estimator}_settings', {'_': None}), **kwargs)
    else:
        raise Exception('unknown state estimator identifier')


