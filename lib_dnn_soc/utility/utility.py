"""
Author: Andreas Bott
year: 2024
mail: andreas.bott@eins.tu-darmstadt.de or andi-bott@web.de

This module contains utility functions.

ResFile:
    A logging interface that writes to a file and optionally prints to the console. Features some formatting functions.
Uniform:
    A modified version of the Uniform distribution from tensorflow_probability. Featuring gradients in the
    zero-probability areas
ZeroTruncatedMultivariateNormal:
    Zero truncated version of the MultivariateNormal distribution from tensorflow_probability.
BotchedNormalDist:
    A normal distribution where some elements are not defined, i.e. have no variance.
GridModel:
    A class that stores all data about the pipes in a network. Uses networkx.Graph as base.
Setting:
    A class that stores the settings for a testcase and features some convenience functions.
"""

#%% imports
import os
import copy
import time
import networkx as nx
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions


class ResFile(object):
    """
    A logging interface that writes to a file and optionally prints to the console.

    Behaves like a file object and can be used with the 'with' statement.
    Allows for shorthand notation using the .log_... functions.
    """
    def __init__(self, name, path='./', print=False):
        self.name = name
        self.path = path
        self._create_path(path)
        self._clear()
        self.print = print

    def __enter__(self):
        self.file = open(f'{self.path}{self.name}', 'a')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()

    @staticmethod
    def _create_path(path):
        if not os.path.exists(path):
            os.makedirs(path)

    def _clear(self):
        self.file = open(f'{self.path}{self.name}', 'w')
        self.file.close()

    def _get_formater(self, fmt):
        def bracket_formater(message, *args, **kwargs):
            # add linebreaks before and after each bracket
            message = message.replace('{', '{\n')
            message = message.replace('}', '\n}\n')
            # correct linebreaks around commas::
            message = message.replace('\n, ', ', \n')
            # indent lines:
            lines = message.split('\n')
            indent = 0
            for i, line in enumerate(lines):
                if '}' in line:
                    indent -= 1
                lines[i] = '    ' * indent + line
                if '{' in line:
                    indent += 1
            message = '\n'.join(lines)
            return message

        def dict_formater(message, *args, **kwargs):
            # add linebreaks for each key-value pair
            sections = message.split(', ')
            for i, section in enumerate(sections):
                if ':' in section:
                    sections[i] = f'\n{section}'
                message = ''.join(sections)
            # add bracket_formater style linebreaks
            message = bracket_formater(message[1:])
            return message

        def time_formater(message, time, *args, **kwargs):
            minutes, seconds = divmod(time, 60)
            return f'{message}: {minutes:2.0f}:{seconds:2.2f}   ({time:2.2f}s) \n'

        formats = {'brackets_indented': bracket_formater,
                   'dict': dict_formater,
                   'time': time_formater}
        return formats.get(fmt, lambda x: x)

    def write(self, message, fmt=None, end='', *args, **kwargs):
        message = self._get_formater(fmt)(message, *args, **kwargs)
        self.file.write(message+end)
        if self.print:
            print(message, end=end)

    def log(self, message, fmt=None, end='\n'):
        ''' convenience function for writing a single line to the file '''
        with self:
            self.write(message, fmt, end)

    def log_times(self, times):
        ''' logs the times given in the dictionary times to the file'''
        abbreviations = {'est_time': 'state estimation', 'opt_time': 'optimisation', 'total_time': 'total'}
        with self:
            self.write(f'{"-" * 20} \n')
            for key, value in times.items():
                self.write(f'time {abbreviations.get(key, key)}', time=value, fmt='time')
            self.write(f'{"-" * 20} \n')

    def log_testcase(self, grid_identifier=None, grid_settings=None, scenario_settings=None,
                     state_estimator_settings=None, OPT_settings=None):
        ''' log all test case settings to the file '''
        with self:
            self.write(f'numerical results for testcase {grid_identifier} calculated at {time.ctime()}\n')
            self.write(f'{"-"*20} \n Settings \n')
            if grid_settings is not None:
                self.write('grid: \n')
                self.write(grid_settings.__str__(), fmt='brackets_indented')
            if scenario_settings is not None:
                self.write('scenario: \n')
                self.write(scenario_settings.__str__(), fmt='brackets_indented')
            if state_estimator_settings is not None:
                self.write('state estimator: \n')
                self.write(state_estimator_settings.__str__(), fmt='brackets_indented')
            if OPT_settings is not None:
                self.write('OPT: \n')
                self.write(OPT_settings.__str__(), fmt='brackets_indented')
            self.write(f'{"-"*20} \n \n')

    def log_df(self, desc_str='', df=None):
        assert df is not None,  'no Value for df passed'
        with self:
            self.write('\n' + desc_str + '\n')
            self.write(df.to_string())
            self.write('\n')

class Uniform(tfd.Uniform):
    '''
    modified version of Uniform distribution from tensorflow_probability.
    Feature gradients in the zero-probability areas which point in the direction of the nearest valid value
    '''

    def __repr__(self):
        return f"<state_estimation.se_lib.utility.Uniform '{self.name}' " \
               f"batch_shape={self.batch_shape.as_list()} event_shape={self.event_shape.as_list()} " \
               f"d_type={self.dtype.__repr__()}>"

    def __str__(self):
        return f'state_estimation.se_lib.utility.Uniform("{self.name}", ' \
               f'batch_shape={self.batch_shape.as_list()}, event_shape={self.event_shape.as_list()}, ' \
               f'd_type={self.dtype.__repr__()})'

    @tf.custom_gradient
    def prob(self, samples):
        val = super(Uniform, self).prob(samples)
        def grad(upstream):
            M = tf.map_fn(lambda d: tf.cond(tf.logical_and(self.low <= d, d <= self.high),
                                                lambda: 0,
                                                lambda: tf.cond(d < self.low,
                                                                lambda: 1,
                                                                lambda: -1)), samples)
            return tf.expand_dims(M, axis=-1) * upstream
        return val, grad

    @tf.custom_gradient
    def log_prob(self, samples):
        val = super(Uniform, self).log_prob(samples)
        def grad(upstream):
            M = tf.map_fn(lambda d: tf.cond(tf.logical_and(self.low <= d, d <= self.high),
                                                lambda: 0,
                                                lambda: tf.cond(d < self.low,
                                                                lambda: 1,
                                                                lambda: -1)), samples)
            return tf.expand_dims(M, axis=-1) * upstream
        return val, grad


class ZeroTruncatedMultivariateNormal(tfd.MultivariateNormalTriL):
    """
    adaptation of the MultivariateNormal distribution provided by tfd
    Cut off all values below zero; overwrite functions: sample, prob, log_prob accordingly
    """
    def __init__(self, loc, scale_tril, validate_args=True, name='ZeroTruncatedMultivariateNormal'):
        parameters = dict(locals())
        self.signs = tf.math.sign(loc)
        self.sample_dim = loc.get_shape()[0]
        super().__init__(loc=tf.math.abs(loc), scale_tril=scale_tril, validate_args=validate_args, name=name)
        self._parameters = parameters

    def __repr__(self):
        return f"<state_estimation.se_lib.utility.ZeroTruncatedMultivariateNormal '{self.name}' " \
               f"batch_shape={self.batch_shape.as_list()} event_shape={self.event_shape.as_list()} " \
               f"d_type={self.dtype.__repr__()}>"

    def __str__(self):
        return f'state_estimation.se_lib.utility.ZeroTruncatedMultivariateNormal("{self.name}", ' \
               f'batch_shape={self.batch_shape.as_list()}, event_shape={self.event_shape.as_list()}, ' \
               f'd_type={self.dtype.__repr__()})'

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        # parameters have the same properties (e.g. shapes) as in non-truncated normal distributions
        return tfd.MultivariateNormalTriL._parameter_properties(dtype, num_classes)

    def _mean(self):
        return self._parameters['loc']

    def sample(self, n_samples):
        # We override sample rather than _sample_n as advised in the tfp documentation.
        # Reasoning:
        #   the parent (tfd.MultivariateNormalTriL) overwrites _call_sample_n instead of the usual _sample_n
        #   therefore, self.sample() does is no longer redirected to _sample_n

        # initialise output tensor with all zeros
        init_samples = tf.zeros(shape=(n_samples, self.sample_dim), dtype=tf.float64)
        # check if any sample state is <0 (invalid sample) or =0 (not jet defined or removed due to production surplus)
        cond = lambda samples: tf.reduce_any(samples <= 0)

        # sample and fill valid rows into the output tensor
        def body(samples):
            new_samples = super(ZeroTruncatedMultivariateNormal, self).sample(n_samples)
            # replace samples that are not jet placed or which contain values below 0 with new samples
            #                  < overhat >  vvv actual cond. vvv      <    overhead to slice the right dimension    >
            samples = tf.where(tf.repeat(tf.reduce_all(samples > 0, axis=1, keepdims=True), self.sample_dim, axis=1),
                               samples, new_samples)

            # remove rows, which do not add up to values above zero (i.e. total demand is lower than production)
            samples = tf.where(tf.repeat(tf.reduce_sum(self.signs*samples, axis=1, keepdims=True) > 0,
                                         self.sample_dim, axis=1), samples, tf.zeros_like(samples))
            return [samples]

        [samples] = tf.while_loop(cond=cond, body=body, loop_vars=[init_samples])
        return samples * self.signs


class BotchedNormalDist:
    """
    "Normal Distribution", where some Elements don't have a variance

    botched_attributes only consider nonzero entries,
    true_attributes have the "true" dimensions
    """
    def __init__(self, mean, botched_cov, mask_matrix):
        self.true_mean = mean
        self.botched_mean = tf.squeeze(tf.sparse.sparse_dense_matmul(tf.sparse.transpose(mask_matrix), mean))
        self.inv_botched_mean = self.true_mean - tf.sparse.sparse_dense_matmul(mask_matrix, tf.expand_dims(self.botched_mean, axis=-1))
        self.true_cov = tf.matmul(tf.sparse.sparse_dense_matmul(mask_matrix, botched_cov), tf.transpose(tf.sparse.to_dense(mask_matrix)))
        self.botched_cov = botched_cov
        self.mask_matrix = mask_matrix

    def sample(self, n_samples):
        botched_samples = tf.constant(np.random.multivariate_normal(self.botched_mean, self.botched_cov, n_samples))
        return tf.transpose(
            tf.sparse.sparse_dense_matmul(self.mask_matrix, tf.transpose(botched_samples)) + self.inv_botched_mean)

    def marginal(self, d):
        return tfd.Normal(loc=self.true_mean[d], scale=tf.sqrt(self.true_cov[d, d]))


class GridModel(nx.DiGraph):
    """
    Stores all data about the pipes in a network

    Uses modul "networkx.Graph" as base. Read networkx docs for further information.

    Author: Friedrich P. Bott A.
    """
    def __init__(self, networkData=None, filepath=None, name="grid"):
        super().__init__()
        if filepath:
            raise NotImplementedError("Cannot yet process filepaths.")
        if networkData:
            networkData = [el for el in networkData if
                           'type' in el.keys()]  # Filter for elements that have the field 'type'
            nodes = [(networkData[i]['index'], networkData[i]) for i in range(len(networkData)) if
                     networkData[i]['type'] == 'Node']
            edges = [(networkData[i]['from'], networkData[i]['to'], networkData[i]) for i in range(len(networkData)) if
                     networkData[i]['type'] == 'Edge']
            if not len(nodes) + len(edges) == len(networkData):
                raise Exception('Some network elements are neither type "node" nor "edge"')

            self.add_nodes_from(nodes)
            self.add_edges_from(edges)
            self.json = networkData
            self.name = name

    def get_passive_grid(self):
        """
        returns a grid object that contains only the passive grid elements
        """
        if not hasattr(self, '_passive_grid'):
            sup_nodes = (node for node, data in self.nodes(data=True) if data['nw_section'] == 'Sup')
            ret_nodes = (node for node, data in self.nodes(data=True) if data['nw_section'] == 'Ret')
            self._sup_graph = self.subgraph(sup_nodes)
            self._ret_graph = self.subgraph(ret_nodes)
            self._passive_grid = nx.compose(self._sup_graph, self._ret_graph)
        return self._passive_grid

    def get_section(self, section):
        """Returns supply or return section of network.

        :param section: Either 'Sup' or 'Ret'
        :type section: str
        :return: Grid model of sub section
        :rtype: GridModel
        """
        nodes = {n for n in self.nodes if
                 self.nodes[n]['nw_section'] == section}  # select supply section only for grid constraints
        return self.subgraph(nodes)

    def get_active_nodes(self):
        "returns nodes that active edges are connected to"
        active_nodes = []
        for u, v, ctrl_type in self.edges(data='edge control type'):
            if ctrl_type == 'active':
                active_nodes.extend([u, v])

        return active_nodes

    def find_all_cycle(self):
        """
        returns a list containing all cycles within the grid, ignoring active edges
        """
        search_grid = self.get_passive_grid()
        cycles = []
        while True:
            try:
                cycle = nx.algorithms.cycles.find_cycle(search_grid, orientation='ignore')
                cycles.append(cycle)
            except nx.exception.NetworkXNoCycle:  # exception appears if no graph is found
                break
            for u, v, direction in cycle:
                search_grid.remove_edge(u, v)
        return cycles

    def all_edges(self, node):
        """Returns all edges connected to given nodes,
        regardless of starting or ending at that nodes
        (in contrast to self.edges(node)).
        Returns edges in correct direction!"""
        edges = list(self.in_edges(node))
        edges.extend(list(self.out_edges(node)))
        return edges

    def has_edge_both_dir(self, u, v):
        if self.has_edge(u, v):
            return True
        elif self.has_edge(v, u):
            return True
        else:
            return False

    def has_edge_by_index(self, idx):
        for u, v in self.edges:
            if self.edges[u, v]['index'] == idx:
                return True
        return False

    def edge_by_index(self, idx):
        for u, v in self.edges:
            if self.edges[u, v]['index'] == idx:
                return self.edges[u, v]
        # if not found:
        raise KeyError("edge with index %s not found" % idx)

    def node_edge_idx(self, node, mode):
        """Returns all indeces of edges that are connected to a node. Lets you chose which edges you need by parameter "mode":
        Going to, starting at or all edges of node.

        :param node: id of node
        :type node: str
        :param mode: Either "to", "from" or "all"
        :return: edge indeces
        :rtype: list
        """
        if mode == 'to':
            edges = self.in_edges(node)
        elif mode == 'from':
            edges = self.out_edges(node)
        elif mode == 'all':
            edges = set(self.in_edges(node))
            edges.update(set(self.out_edges(node)))
        else:
            raise ValueError("Unknown mode \"%s\"" % mode)
        idx = [self.edges[e]['index'] for e in edges]

        return idx

    def edge_idx_by_ctrl_type(self, ctrl_type):
        """Returns list of indeces of all edges with given control type.

        :param ctrl_type: 'active' or 'passive' (not pT, QT etc.)
        :type ctrl_type: str
        :return: list of indeces
        :rtype: List[str]
        """
        return [e['index'] for e in self.edges.values() if e['edge control type'] == ctrl_type]

    def node_pos(self, *args, **kwargs):
        """
        Returns dict of node x-y positions for plots: {n:(x,y)}
        """
        return {n: (self.nodes[n]['coordinate'][0], self.nodes[n]['coordinate'][1]) for n in self.nodes}


class Setting(object):
    """
    A class that stores the settings for a testcase and features some convenience functions.
    Allows for easy adaptation of individual parameters without carrying about the nested dictionary structure.

    If key-value pairs are passed as **parameter, the corresponding key is searched in the default parameter dict and
    updated with the new value. If the key is not found, an exception is raised.

    The remaining functions are convenience functions that return the schedules, demands, power plant parameters, etc.
    """
    def __init__(self, name, default_params, **parameter):
        self.name = name
        self.settings = self.merge_parameter(copy.deepcopy(default_params), **parameter)
        self._absolute_pp_boundaries(self.settings['powerplant_params'])
        self._parameter = parameter    # stores the non-default parameter of this experiment - for logging purposes only

    @staticmethod
    def merge_parameter(params, **updates):
        def _merge_parameter(params, update_key, update_value):
            updated = False
            if update_key in params.keys():
                params[update_key] = update_value
                updated = True
            else:
                for subkey, subvalue in params.items():
                    if type(subvalue) == dict:
                        updated, subvalue = _merge_parameter(subvalue, update_key, update_value)
                    if updated:
                        break
            return updated, params

        for key, value in updates.items():
            key_found, params = _merge_parameter(params, key, value)
            if not key_found:
                raise Exception(f'Key {key} not found in default parameter dict')
        return params

    def _absolute_pp_boundaries(self, pp_settings):
        # add absolute min-max constraints to the power plant parameters, if they are missing
        # ! caution: changes pp_settings in place !
        for i, (pp, settings) in enumerate(pp_settings.items()):
            if 'q_min' not in settings.keys():
                settings['q_min'] = settings['q_min_rel'] * self._read_schedules('sched_q')[0, i]
            if 'q_max' not in settings.keys():
                settings['q_max'] = settings['q_max_rel'] * self._read_schedules('sched_q')[0, i]

    def _read_schedules(self, x):
        if type(x) is list:
            return list(self.settings['schedules'][x_i] for x_i in x)
        else:
            return self.settings['schedules'][x]

    def get_schedules(self):
        return self._read_schedules(['sched_q', 'sched_T_q'])

    def get_demands(self):
        return self._read_schedules(['sched_d', 'sched_T_d'])

    def get_dhs_inputs(self):
        vals = self._read_schedules(['sched_q', 'sched_T_q', 'sched_d', 'sched_T_d'])
        return {p: vals[i] for i, p in enumerate(['q', 'T_q', 'd', 'T_d'])}

    def get_estimated_demands(self):
        if self.settings['estimated_demands'] is not None:
            return self.settings['estimated_demands']
        else:
            return self._read_schedules('sched_d')

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

