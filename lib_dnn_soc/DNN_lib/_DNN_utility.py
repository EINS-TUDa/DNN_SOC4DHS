"""
Author: Andreas Bott
year: 2024
mail: andreas.bott@eins.tu-darmstadt.de or andi-bott@web.de


This module contains objects to handle the input-output data for the DNN training process.
"""

import numpy as np
import tensorflow as tf
import warnings
from .se_NN_lib import import_training_data, InputGenerator


class InsufficientDataError(Exception):
    """
    Exception to be raised when the requested number of samples is insufficient
    The samples are distributed over multiple files. The exception thus can contain information regarding the number of
    found files, among the expected information, i.e. the number of requested and available samples.
    """
    def __init__(self, message=None, task=None, n_avail=None, n_req=None, file_s=None, file_e=None, skip=None, **kwargs):
        self.kwargs = dict()
        if message is not None:
            self.kwargs['message'] = message
        self.task = task
        if task is not None:
            self.kwargs['task'] = task
        self.n_avail = n_avail
        if n_avail is not None:
            self.kwargs['n_avail'] = n_avail
        self.n_req = n_req
        if n_req is not None:
            self.kwargs['n_req'] = n_req
        self.file_s = file_s
        if file_s is not None:
            self.kwargs['file_s'] = file_s
        self.file_e = file_e
        if file_e is not None:
            self.kwargs['file_e'] = file_e
        self.skip = skip
        if skip is not None:
            self.kwargs['skip'] = skip

        task_snippet = f'for task: {self.task} ' if self.task is not None else ''
        file_snippet = f'in files {self.file_s} ... {self.file_e}' if (self.file_s is not None and
                                                                       self.file_e is not None) else ''
        request_snippet = f'Requested samples: {self.n_req}' if self.n_req is not None else ''
        avail_snippet = f'available: {self.n_avail} ' if self.n_avail is not None else ''
        skip_snippet = f' (skipped first {self.skip} samples)' if self.skip is not None else ''
        sep1 = ', ' if self.n_req is not None or self.n_avail is not None else ''
        sep2 = '; ' if self.n_req is not None and self.n_avail is not None else ''

        if message is None:
            message = f'Insufficient number of training samples ' \
                      f'{task_snippet}{file_snippet}{sep1}{request_snippet}{sep2}{avail_snippet}{skip_snippet}'
        super().__init__(message)


class DnnData(object):
    """
    Wrapper around the data to provide additional functionality.

    public methods:
    get_data(n_samples, section) -> [inputs], [outputs]: returns the requested number of samples from the specified
                                                             section, i.e., training, validation or test data
    get_data_generator(n_samples, batch_size) -> generator: returns a generator object to provide the data in batches
                                                                 the generator generates independent inputs in each
                                                                 batch/epoch but does not provide the true output

    Internal functionality for data loading:
    Data is stored in the form
    [dq_vals], [dq_temp_vals], [[grid state]]
    where dq_vals and dq_temp_vals contain information for supplies and demands.

    The desired input-output format is:
    [d_vals, q_vals, T_d_vals, T_q_vals], [state, dq]
    where d_ and q_ values are only supplies and demands respectively. dq is the initial power vector which may be used
    to compute an additional loss during the training process.

    Additional data preprocessing includes adding or removing the power value for the slack power plant if present in
    the data and desired for the inputs or outputs respectively.
    """
    def __init__(self, data_file, SE, n_train=0, n_val=0, d_dist=None, T_dist=None,
                 include_slack_input=False, include_slack_output=False):
        self.data_file = data_file
        self.start_ind_validation = n_train
        self.start_ind_test = n_train + n_val
        self.include_slack_input = include_slack_input
        self.include_slack_output = include_slack_output

        # Data shape specifications
        self.n_demands = len(SE.dem_order)
        self.n_supplies = len(SE.pp_order)
        n_actives = self.n_supplies + self.n_demands
        self.input_shape = [(None, self.n_demands), (None, self.n_supplies if include_slack_input else self.n_supplies-1),
                            (None, self.n_demands), (None, self.n_supplies)]
        self.output_shape = [(None, (SE.n_edges + SE.n_nodes)*2),
                             (None, n_actives if include_slack_output else n_actives - 1)]

        # store input-output samples as tensors
        self.train_inputs = None
        self.train_outputs = None
        self.validation_inputs = None
        self.validation_outputs = None
        self.test_inputs = None
        self.test_outputs = None

        # settings for data_generator:
        self.d_dist = d_dist
        self.T_dist = T_dist
        self.state_shape = 2*SE.n_nodes + 2*SE.n_edges

        # setup divider between supply and demand data
        self.dem_sign = SE.dem_sign
        self.temp_sign = SE.temp_sign
        self.dem_index_list = [ind for ind in SE.demands.keys() if self.dem_sign[SE.dem_ind[ind]] > 0]
        self.heat_index_list = [ind for ind in SE.demands.keys() if self.dem_sign[SE.dem_ind[ind]] < 0]
        self.heat_index_list.extend(SE.heatings.keys())

    def _load_data(self, n_samples, offset):
        [dq, T, state] = import_training_data(n_samples, self.data_file, skip=offset, f_id0=0)
        # reorder data:
        d_vals = tf.gather(dq, np.where(self.dem_sign > 0)[0], axis=1)
        q_vals = tf.gather(dq, np.where(self.dem_sign < 0)[0], axis=1)
        T_d_vals = tf.gather(T, np.where(self.temp_sign > 0)[0], axis=1)
        T_q_vals = tf.gather(T, np.where(self.temp_sign < 0)[0], axis=1)

        # potentially remove slack input and output
        if not self.include_slack_input:
            if tf.shape(q_vals)[-1] == self.n_supplies:
                q_vals = q_vals[:, :-1]
        if not self.include_slack_output:
            if tf.shape(dq)[-1] == self.n_supplies+self.n_demands:
                dq = dq[:, :-1]

        # return input powers and feed in temperatures - true state and powers in original order
        return [d_vals, q_vals, T_d_vals, T_q_vals], [state, dq]

    def get_data(self, n_samples, section='train', adjust_validation_split=True):
        if n_samples == 0:
            warnings.warn('Requested 0 samples from data set. Returning None.')
            return [None], [None]

        if section == 'train':
            inputs = self.train_inputs
            outputs = self.train_outputs
            offset = 0
            if adjust_validation_split:
                self.start_ind_validation = max(self.start_ind_validation, n_samples)
        elif section == 'validation':
            inputs = self.validation_inputs
            outputs = self.validation_outputs
            offset = self.start_ind_validation
            self.start_ind_test = max(self.start_ind_test, offset + n_samples)
        elif section == 'test':
            inputs = self.test_inputs
            outputs = self.test_outputs
            offset = self.start_ind_test
        else:
            raise Exception(f'Unknown section specification {section}; supported sections are "train", "validation" '
                            f'or "test"')

        n_available = 0 if inputs is None else inputs[0].shape()[0]
        if n_samples > n_available:
            try:
                new_inputs, new_outputs = self._load_data(n_samples-n_available, offset=offset)
            except InsufficientDataError as e:
                raise InsufficientDataError(task=f'load {section} data', skip=offset, **e.kwargs)
            if inputs is not None:
                for i, new in enumerate(new_inputs):
                    inputs[i] = tf.concat([inputs[i], new], axis=1)
            else:
                inputs = new_inputs
            if outputs is not None:
                for i, new in enumerate(new_outputs):
                    outputs[i] = tf.concat([outputs[i], new], axis=1)
            else:
                outputs = new_outputs
        return [inp[:n_samples, :] for inp in inputs], [out[:n_samples, :] for out in outputs]

    def get_data_generator(self, n_samples, batch_size):
        if self.d_dist is None or self.T_dist is None:
            raise Exception(f'pls specify "d_dist" and "T_dist" in order to use a data generator')
        return InputGenerator(self.d_dist, self.T_dist, n_states=self.state_shape,
                              n_samples_per_epoch=n_samples, batch_size=batch_size,
                              include_slack_output=self.include_slack_output)