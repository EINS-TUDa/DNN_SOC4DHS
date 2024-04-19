"""
Author: Andreas Bott
year: 2024
mail: andreas.bott@eins.tu-darmstadt.de or andi-bott@web.de

State Estimator like class that returns a predefined result.
"""

from lib_dnn_soc.state_estimation.stateestimator import StateEstimator


class Settings_state_estimator(StateEstimator):
    """
    This class is not a "real" state estimator, but rather a placeholder returning the estimated demands in the settings
    """
    def __init__(self, schedules, estimated_demands, **kwargs):
        super().__init__(**kwargs)
        self.schedules = schedules
        self.estimated_demands = estimated_demands

    def _read_schedules(self, x):
        if type(x) is list:
            return list(self.schedules[x_i] for x_i in x)
        else:
            return self.schedules[x]

    def _get_estimated_demands(self):
        if self.estimated_demands is not None:
            return self.estimated_demands
        else:
            return self._read_schedules('sched_d')

    def estimate_state(self, q, T_d, T_q, meas_vals, **kwargs):
        return self._get_estimated_demands()