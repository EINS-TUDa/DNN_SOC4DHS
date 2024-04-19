
"""
Author: Andreas Bott
year: 2024
mail: andreas.bott@eins.tu-darmstadt.de or andi-bott@web.de

This module contains the parent class for all state estimators
"""

from lib_dnn_soc.DNN_lib import WrappedModel
class StateEstimator(object):
    def __init__(self, Measurements, SE, d_prior_dist, state_model=None, **kwargs):
        self.Measurements = Measurements
        self.SE = SE
        self._state_model = state_model
        self.d_prior_dist = d_prior_dist

    def get_state_model(self, q, T_d, T_q):
        if self._state_model is None:
            raise Exception('No state model has been set')
        return WrappedModel(self._state_model, input_masks={'q': q, 't_d': T_d, 't_q': T_q}, output_selector=0)

    def estimate_state(self, q, T_d, T_q, meas_vals, **kwargs):
        raise NotImplementedError('Must be implemented by subclass')
