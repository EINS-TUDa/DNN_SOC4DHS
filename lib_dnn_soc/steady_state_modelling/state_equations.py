"""
Author: Andreas Bott
year: 2024
mail: andreas.bott@eins.tu-darmstadt.de or andi-bott@web.de

This module contains the class StateEquations which encapsulate the steady-state equations of the district heating grid.

The class features a variety of methods to setup, evaluate and manipulate the state equations. The function
"calculate_physics_violation" mimics the behavior of the "evaluate_state_equations" called with mode='forwardpass'.
It has a reduced flexibility which allows for a better optimisation if used in the context of a loss function.
"""

import tensorflow as tf
import numpy as np
import networkx as nx
import warnings

from typing import Tuple, Set, List

tf.keras.backend.set_floatx('float64')


class TemperatureProbagationException(Exception):
    def __init__(self, message='Temperature propagation does not reach all nodes'):
        super().__init__(message)


class StateEquations():
    def __init__(self, edges, nodes, demands, heatings, fix_dp: dict, use_demands=True,
                 Ta=tf.constant(5., dtype=tf.float64)):
        """
        :param edges: list of dictionaries with the following keys
            - index: unique identifier of the edge
            - from: index of the node the edge is coming from
            - to: index of the node the edge is going to
            - edge control type: 'active' or 'passive'
            - if the edge type is passive:
                - diameter: diameter of the pipe in m
                - length [m]: length of the pipe in m
                - temp_loss_coeff: inverse of the thermal resistance per unit length of the pipe in kW/(m K)
                - fd_nom: Darcy-Weisbach friction factor for nominal flow
                - bend_factor: factor to adjust the flow coefficient for bends
                - nw_section: 'Sup' or 'Ret' for supply or return edge
            - if the edge type is active:
                - nw_section: 'internal'
        :param nodes: list of dictionaries with the following keys
            - index: unique identifier of the node
            - nw_section: 'Sup' or 'Ret' for supply or return node
            - coordinate: tuple of the x and y coordinate of the node
        :param demands: specifies demands and heatings
            dictionary with the following structure
            - key: unique identifier of the demand
            - value: dictionary with the following keys
                - Power: power demand in kW
                - Temperature: temperature at the end of the edge in °C
        :param heatings: referes only to the slack supply edge, structure equal to demands
        :param fix_dp: dict with two entries; key: node index, value: fixed pressure difference in bar
        :param use_demands: boolean, if True, demands are used in the state equations, if False, they are ignored
        :param Ta: ambient temperature in °C
        """
        self.use_demands = use_demands
        # nodes and edges:
        self.edges = edges
        self.nodes = nodes
        self.n_edges = len(edges)
        self.n_nodes = len(nodes)
        self.n_demands = len(demands)
        find_node = dict()  # dict to find the numerical index of a node by name
        find_edge = dict()  # dict to find the numerical index of an edge by name
        for i in range(self.n_nodes):
            find_node[nodes[i]['index']] = i
        for i in range(self.n_edges):
            find_edge[edges[i]['index']] = i
        self.find_node = find_node
        self.find_edge = find_edge
        self.n_active_edges = len([edg for edg in edges if edg['edge control type'] == 'active'])

        # demands and heatings:
        self.demands = demands
        self.heatings = heatings
        self.fix_dp = fix_dp

        # state variables
        self.T = tf.Variable(tf.ones((self.n_nodes, 1), dtype=tf.float64), dtype=tf.float64, name='temperaturs')
        self.p = tf.Variable(tf.ones((self.n_nodes, 1), dtype=tf.float64), dtype=tf.float64, name='pressures')
        self.mf = tf.Variable(tf.ones((self.n_edges, 1), dtype=tf.float64), dtype=tf.float64, name='mass_flows')
        # T_from = tf.Variable([120., 120., 40., 40.], dtype=tf.float64)
        self.T_end = tf.Variable(tf.ones((self.n_edges, 1), dtype=tf.float64), dtype=tf.float64,
                                 name='end-of-line_temperature')
        self.Q_heat = tf.Variable(tf.ones(self.n_active_edges - 1, dtype=tf.float64), dtype=tf.float64,
                                  name='demand_power_values')
        self.Q_T_end = tf.Variable(tf.ones(self.n_active_edges - 1, dtype=tf.float64), dtype=tf.float64,
                                   name='feed_in_temperatures')

        # constants
        self.Ta = tf.constant(Ta, name='T_amb', dtype=tf.float64)
        self.cp = tf.constant(4.180, name='cp_water_in_kJ/kgK', dtype=tf.float64)
        self.pi = tf.constant(np.pi, name='pi', dtype=tf.float64)
        self.rho = tf.constant(997., name='density_water', dtype=tf.float64)
        self.gravitation = tf.constant(9.81, name='gravitation_constant', dtype=tf.float64)

        # setting up connectivity matrix A, masks and setting initial values for state variables
        A_row = []  # setup for sparse connectivity matrix A
        A_col = []
        A_val = []
        A_active_row = []  # identity mapping matrix active mass flows
        A_active_col = []
        A_active_val = []
        dem_ind = dict()

        dem_pos = []
        pp_pos = []
        dem_order = []
        pp_order = []
        d_count = 0

        # parse edges connectivity matrix A
        for i, edg in enumerate(self.edges):
            n_from = self.find_node[edg['from']]
            n_to = self.find_node[edg['to']]
            # add entry in A
            A_row.extend([n_from, n_to])
            A_col.extend([i, i])
            A_val.extend([-1., 1.])

            if edg['edge control type'] == 'active':
                # for now: heating and demand
                if edg['index'] in self.demands.keys():
                    dem_ind[edg['index']] = d_count
                    self.Q_heat[d_count].assign(self.demands[edg['index']]['Power'])
                    self.Q_T_end[d_count].assign(self.demands[edg['index']]['Temperature'])
                    A_active_row.append(d_count)
                    A_active_col.append(i)
                    A_active_val.append(1)
                    # if the edge is oriented from return to supply side, i.e. it is a supply edge
                    if self.nodes[self.find_node[edg['to']]]['nw_section'] == 'Sup':
                        pp_pos.extend([d_count])
                        pp_order.extend([edg['index']])
                    else:
                        dem_pos.extend([d_count])
                        dem_order.extend([edg['index']])
                    d_count += 1
        pp_order.extend(list(self.heatings.keys()))
        pp_pos.extend([d_count])
        self.pp_order = pp_order                             # order in which heatings appear in self.Q_heat
        self.pp_pos = tf.constant(pp_pos, dtype=tf.int64)    # indices of heatings in the Q_heat vector
        self.dem_order = dem_order                           # order in which demands appear in self.Q_heat
        self.dem_pos = tf.constant(dem_pos, dtype=tf.int64)  # indices of demands in the Q_heat vector
        self.eq_id = dict()                                  # dict to identify the meaning of an specific equation -> filled on demand

        # construct connectivity matrix A
        A = tf.SparseTensor(indices=list(zip(A_row, A_col)), values=tf.constant(A_val, dtype=tf.float64),
                            dense_shape=[self.n_nodes, self.n_edges])
        A = tf.sparse.reorder(A)
        A_red_size = tf.cast((tf.shape(A) - [1, 0]), tf.int64)

        A_active = tf.SparseTensor(indices=list(zip(A_active_row, A_active_col)),
                                   values=tf.constant(A_active_val, dtype=tf.float64),
                                   dense_shape=[d_count, self.n_edges])

        self.A = A
        self.A_dense = tf.sparse.to_dense(A)
        self.A_active = A_active
        self.A_red_size = A_red_size
        self.dem_ind = dem_ind
        self.A_red_dense = tf.sparse.to_dense(tf.sparse.slice(A, start=[0, 0], size=self.A_red_size))
        # setup masks
        # masks are used to avoid calculating equations for fixed values, i.e. pressures at fixed pressure nodes or the
        # temperatures at the end of active edges. This greatly enhances the numerical stability of the model / solvers.
        self._set_masks()
        self.state_dimension_segments = [
            (int(sum(np.sum(self.masks[i]) for i in range(j))), int(sum(np.sum(self.masks[i]) for i in range(j + 1))))
            for j in range(len(self.masks))]
        self.set_init_state()

    def _setup_T_masks(self):
        T_mask = np.ones((self.n_nodes, 1))
        Tend_mask = np.ones((self.n_edges, 1))
        for i, edg in enumerate(self.edges):
            if edg['edge control type'] == 'active':
                n_to = self.find_node[edg['to']]
                T_mask[n_to] = 0  # the temperature at the end of active edge is fixed.
                Tend_mask[i] = 0
        return T_mask, Tend_mask

    def _setup_mf_masks(self):
        mf_mask = np.ones((self.n_edges, 1))
        return mf_mask

    def _setup_p_masks(self):
        p_mask = np.ones((self.n_nodes, 1))
        for i in range(self.n_nodes):
            nd = self.nodes[i]
            if nd['index'] in self.fix_dp.keys():
                p_mask[i] = 0
                self.p[i].assign([self.fix_dp[nd['index']]])
        return p_mask

    def _set_masks(self):
        # generates masks and mask matrices for the state variables
        T_mask, Tend_mask = self._setup_T_masks()
        mf_mask = self._setup_mf_masks()
        p_mask = self._setup_p_masks()
        self.masks = [T_mask, mf_mask, p_mask, Tend_mask]
        self.B_mask = tf.squeeze(1 - tf.sparse.sparse_dense_matmul(self.A, tf.cast((1 - Tend_mask), tf.float64)))
        self.B_mask_matrix_dense = tf.transpose(tf.sparse.to_dense(self._get_mask_matrix(tf.cast(self.B_mask, tf.bool))))
        self.mask_matrix = [self._get_mask_matrix(mask) for mask in self.masks]
        self.mask_matrix_dense = [tf.sparse.to_dense(m) for m in self.mask_matrix]
        self.mask_matrix_full = self.combine_mask_matrix(self.mask_matrix)

    @property
    def dem_sign(self):
        if not hasattr(self, '_dem_sign'):
            dem_sign = np.zeros_like(self.Q_heat)
            for d, q in self.demands.items():
                dem_sign[self.dem_ind[d]] = np.sign(q['Power'])
            self._dem_sign = tf.constant(dem_sign, dtype=tf.float64)
        return self._dem_sign

    @property
    def temp_sign(self):
        if not hasattr(self, '_temp_sign'):
            temp_sign = np.pad(self.dem_sign, (0, 1), constant_values=-1)
            self._temp_sign = tf.constant(temp_sign, dtype=tf.float64)
        return self._temp_sign

    def get_config(self):
        # return the values needed for the init to create an identical copy of this object
        return {'edges': self.edges,
                'nodes': self.nodes,
                'demands': {
                    d: {k: self.demands[d][k].numpy() if tf.is_tensor(self.demands[d][k]) else self.demands[d][k]
                        for k in self.demands[d].keys()} for d in self.demands.keys()},
                'heatings': {
                    h: {k: self.heatings[h][k].numpy() if tf.is_tensor(self.heatings[h][k]) else self.heatings[h][k]
                        for k in self.heatings[h].keys()} for h in self.heatings.keys()},
                'fix_dp': self.fix_dp,
                'use_demands': self.use_demands,
                'Ta': self.Ta.numpy() if tf.is_tensor(self.Ta) else self.Ta}

    def get_phyical_loss_config(self):
        # return the constant values passed to "calculate_physics_violation" defined further below. The later function
        # is not part of this class for performance reasons, but is associated with a StateEquation object.
        return {'n_nodes': self.n_nodes,
                'n_edges': self.n_edges,
                'n_active_edges': self.n_active_edges,
                'A_red_dense': self.A_red_dense,
                'A_dense': self.A_dense,
                'B_mask_matrix_dense': self.B_mask_matrix_dense,
                'find_node': self.find_node,
                'edges': self.edges,
                'Ta': self.Ta, 'cp': self.cp, 'rho': self.rho}

    def _get_mask_matrix(self, mask):
        rows = np.nonzero(mask)[0].tolist()
        n = int(np.sum(mask))  # number on not-masked entries
        cols = list(range(n))
        ones = tf.ones(n, dtype=tf.float64)
        mask_matrix = tf.SparseTensor(indices=list(zip(rows, cols)), values=ones, dense_shape=[len(mask), n])
        mask_matrix = tf.sparse.reorder(mask_matrix)
        return mask_matrix

    def combine_mask_matrix(self, masks):
        M_ind = masks[0].indices
        M_val = masks[0].values
        shape = masks[0].shape
        for i in range(1, len(masks)):
            M_ind = tf.concat([M_ind, masks[i].indices + tf.cast(shape, tf.int64)], axis=0)
            M_val = tf.concat([M_val, masks[i].values], axis=0)
            shape = tf.add(shape, masks[i].shape)
        M = tf.SparseTensor(indices=M_ind, values=M_val, dense_shape=tf.cast(shape, tf.int64))
        return tf.sparse.reorder(M)

    def identify_sup(self, i, verbose=True):
        """
            identifies the i-th supply in the demand vector - does not count demands, only supplies
        """
        def _identify_sup(i):
            for dem, vals in self.demands.items():
                if vals['Power'] <= 0:
                    if i == 0:
                        return dem
                    else:
                        i -= 1
            return list(self.heatings.keys())[0]

        id = _identify_sup(i)
        if verbose:
            print(f'{id}')
        else:
            return id

    def identify_pos(self, pos, verbose=True):
        """
            identifies the state dimension belonging to a position in the state vector
            if verbose: print the result, else: return it
        """
        def _identify_pos(pos):
            if pos < self.n_nodes:
                return ('Temp', self.nodes[pos]['index'])
            else:
                pos -= self.n_nodes
            if pos < self.n_edges:
                return ('mf', self.edges[pos]['index'])
            else:
                pos -= self.n_edges
            if pos < self.n_nodes:
                return ('pr', self.nodes[pos]['index'])
            else:
                pos -= self.n_nodes
            if pos < self.n_edges:
                return ('T_end', self.edges[pos]['index'])
            else:
                raise IndexError(f'pos {pos+self.n_edges+2*self.n_nodes} is out of range for the state vector with length '
                                 f'{2*(self.n_nodes+self.n_edges)}')
        tpe, id = _identify_pos(pos)
        if verbose:
            print(f'{tpe} at {id}')
        else:
            return tpe, id

    def identify_eq(self, eq, use_demands=None, verbose=True):
        """
            identifies the equation belonging to a specified position in the right hand side vector
            if verbose: print the result, else: return it
        """
        def _identify_eq(eq, use_demands):
            # dict lookup if value is already known
            value = self.eq_id.get(eq, None)
            if value is not None:
                return value

            # identify the equation and add new entry to eq_id for future use
            # mf conservation equations
            if eq < self.A_red_size[0]:
                tpe = 'mf conservation'
                id = f'node {self.nodes[eq]["index"]}'
                self.eq_id[eq] = (id, tpe)
                return id, tpe

            # temperature mixing equations
            if eq < self.A_red_size[0] + self.B_mask_matrix_dense.shape[0]:
                tpe = 'Temp mixing'
                node_id = np.where(self.B_mask_matrix_dense[eq-self.A_red_size[0], :])[0][0]
                id = f'node {self.nodes[node_id]["index"]}'
                self.eq_id[eq] = (id, tpe)
                return id, tpe

            # edge related equations - mimic the structure of the for-loop in the "evaluate_state_equations" function
            skip_edge = 0   # counter for skipped edges if demands were not used
            for i in range(self.n_edges):
                edg = self.edges[i]
                # edge equations:
                if edg['edge control type'] == 'passive':
                    # first equation: mass temperature loss:
                    if eq == self.A_red_size[0] + self.B_mask_matrix_dense.shape[0] + 2*(i-skip_edge):
                        tpe = 'Temp loss'
                        id = f'edge {edg["index"]}'
                        self.eq_id[eq] = (id, tpe)
                        return id, tpe
                    # second equation: pressure loss:
                    if eq == self.A_red_size[0] + self.B_mask_matrix_dense.shape[0] + 2*(i-skip_edge) + 1:
                        tpe = 'Pressure loss'
                        id = f'edge {edg["index"]}'
                        self.eq_id[eq] = (id, tpe)
                        return id, tpe
                elif edg['index'] in self.demands.keys():
                    if use_demands:
                        tpe = 'power equation'
                        id = f'demand {edg["index"]}'
                        self.eq_id[eq] = (id, tpe)
                        return id, tpe
                    else:
                        skip_edge += 1
                else: # edge is the slack edge, no equation is associated with it
                    skip_edge += 1

        if use_demands is None:
            use_demands = self.use_demands
        id, tpe = _identify_eq(eq, use_demands)
        if verbose:
            print(f'{tpe} {id}')
        else:
            return id, tpe

    def set_active_edge_temperature(self, edg_index, temp):
        """ update temperature for active edge - updates T_end, T_node and entry in demands / heatings """
        temp = tf.squeeze(temp)
        if edg_index in self.heatings.keys():
            self.heatings[edg_index]['Temperature'] = temp
        else:
            self.demands[edg_index]['Temperature'] = temp

        n_edg = self.find_edge[edg_index]
        n_to = self.find_node[self.edges[n_edg]['to']]
        self.T_end[n_edg].assign([temp])
        self.T[n_to].assign([temp])

    def set_init_state(self, T_ret=40, T_sup=None):
        # sets all state variables to initila values
        T_sup = T_sup if T_sup is not None else self.heatings[list(self.heatings.keys())[0]]['Temperature']
        for i, edg in enumerate(self.edges):
            n_from = self.find_node[edg['from']]
            n_to = self.find_node[edg['to']]

            if edg['edge control type'] == 'active':
                # for now: heating and demand
                if edg['index'] in self.heatings.keys():
                    self.T[n_to].assign([self.heatings[edg['index']]['Temperature']])
                    self.T_end[i].assign([self.heatings[edg['index']]['Temperature']])
                    self.H_T_end = tf.Variable([self.heatings[edg['index']]['Temperature']], dtype=tf.float64)
                    # Q_heat.assign([heatings[edg['index']]['Power']])
                else:
                    self.T[n_to].assign([self.demands[edg['index']]['Temperature']])
                    self.T_end[i].assign([self.demands[edg['index']]['Temperature']])
            else:
                # use masks to avoid replacing fixed temperatures determined by active edges
                if edg['nw_section'] == 'Sup':
                    if self.masks[0][n_to]:
                        self.T[n_to].assign([T_sup])
                    if self.masks[0][n_from]:
                        self.T[n_from].assign([T_sup])
                    if self.masks[3][i]:
                        self.T_end[i].assign([T_sup])
                else:
                    if self.masks[0][n_to]:
                        self.T[n_to].assign([tf.cast(T_ret, tf.float64)])
                    if self.masks[0][n_from]:
                        self.T[n_from].assign([tf.cast(T_ret, tf.float64)])
                    if self.masks[3][i]:
                        self.T_end[i].assign([tf.cast(T_ret, tf.float64)])
        self.init_state = [tf.identity(self.T), tf.identity(self.mf), tf.identity(self.p), tf.identity(self.T_end)]

    def set_save_state(self):
        # save a 'hardcopy' of the initial state
        self.save_state = [tf.identity(self.T), tf.identity(self.mf), tf.identity(self.p), tf.identity(self.T_end),
                           tf.identity(self.Q_heat)]

    def load_save_state(self):
        # resets the state of all state variables to its saved values - if no state is saved, set to init state
        if hasattr(self, 'save_state'):
            self.T.assign(self.save_state[0])
            self.mf.assign(self.save_state[1])
            self.p.assign(self.save_state[2])
            self.T_end.assign(self.save_state[3])
            self.Q_heat.assign(self.save_state[4])
        else:
            self.T.assign(self.init_state[0])
            self.mf.assign(self.init_state[1])
            self.p.assign(self.init_state[2])
            self.T_end.assign(self.init_state[3])

    # edge functions
    @tf.function
    def eq_pipe_T(self, mf, T0, l, Lambda):
        return (T0 - self.Ta) * tf.math.exp(
            tf.math.divide_no_nan(tf.cast(-l * Lambda, tf.float64), (1000 * self.cp * mf))) + self.Ta

    # if mf = 0: divide_no_nan returns 0 -> in SE set T0 = Ta for these edges, eq_pipe_T returns 0 and zero gradients

    @tf.function
    def eq_pipe_p(self, mf, fd, l, d):
        return fd * 8 * l * mf * tf.math.abs(mf) / (self.pi ** 2 * self.rho * d ** 5) * 1e-5

    @tf.function
    def eq_pipe_p_static(self, zf, zt):
        return (zt - zf) * self.gravitation * self.rho * 1e-5

    @tf.function
    def eq_pipe_p_lin(self, mf0, fd, l, d):  # 1st derivertive of eq_pipe_p at mf0
        return 2 * fd * 8 * l * tf.math.abs(mf0) / (self.pi ** 2 * self.rho * d ** 5)

    @tf.function
    def eq_demand_Q(self, mf, Q, Ts, Tr):
        return mf * self.cp * (Ts - Tr) - Q

    @tf.function()
    def eq_fix_dp(self, pf, pt, p_set, lambda1=1.e-2, lambda2=1.e0):
        return lambda1 * tf.clip_by_value(pt - pf - p_set, 0, 1.e7) + lambda2 * tf.clip_by_value(pt - pf - p_set, -1.e7,0)

    @tf.function(jit_compile=False)  # jit_compile=True does not work for whatever reason, I do not know
    def evaluate_state_equations(self, mode, T=None, mf=None, p=None, T_end=None, Q_heat=None, adjust_for_hight=False,
                                 use_heating=False):
        verbose = False
        if tf.executing_eagerly():
            verbose = True
            warnings.warn('state equations evaluated eagerly!')
        '''
        evaluates the state equations,
        modes:
            forwardpass -> calculates the loss for given variables
            gradient -> returns loss and gradient of squared sum of the loss
            jacobian -> retruns loss and jacobian of the loss
        '''
        # alias state variables for readability
        A = self.A
        demands = self.demands
        if Q_heat is None:
            Q_heat = self.Q_heat
        if T is None:
            T = self.T
        if mf is None:
            mf = self.mf
        if p is None:
            p = self.p
        if T_end is None:
            T_end = self.T_end
        # tf.print(tf.executing_eagerly())
        if mode == 'demand jacobian' or mode == 'demand and temperature jacobian' or tf.executing_eagerly():
            persistence = True
            # for this case two jac have to be calculated, forcing for persistent=True for the GradientTape
        else:
            persistence = False
        with tf.GradientTape(persistent=persistence) as tape:
            # loss 1: massflow conservation in nodes - one equation in A is lin. dep. on the others -> drop one row
            A_red = tf.sparse.slice(A, start=[0, 0], size=self.A_red_size)
            if mode == 'gradient':
                loss = tf.reduce_sum(tf.sparse.sparse_dense_matmul(A_red, mf) ** 2)
            else:
                loss = (tf.sparse.sparse_dense_matmul(A_red, mf))
            # temperature in each nodes:
            # B = A.with_values(tf.cast(A.__mul__(tf.transpose(mf)).values > 0, dtype=A.dtype))
            B = tf.clip_by_value(
                tf.math.multiply(tf.expand_dims(self.A_dense, axis=0),
                                 tf.expand_dims(tf.transpose(mf), axis=1)), 0, 1.e7)
            # T = Sum(m*T_end)/Sum(m)  -> B: inflow matrix, matmul leads to the sum in the numerator
            # if Sum(B, axis=1) == 0 (no incoming mass flow, e.g. at zero demand) this leads to a 0/0 division
            # add self.Ta*1.e-15/1.e-15 to get analytical solution Ta in this case, 1.e-15 to "ignore" it if mf != 0
            l = T - tf.squeeze(
                tf.transpose((tf.matmul(B, tf.expand_dims(tf.transpose(T_end), axis=-1)) + self.Ta * 1.e-15) /
                             (tf.reduce_sum(B, axis=-1, keepdims=True) + 1.e-15)), axis=0)
            # l = tf.boolean_mask(l, self.B_mask, axis=0)
            l = self.B_mask_matrix_dense @ l
            # tf.sparse.sparse_dense_matmul(B, tf.cast(self.masks[3], tf.float64))
            if mode == 'gradient':
                loss += tf.reduce_sum(l ** 2, axis=0)
            else:
                loss = tf.concat([loss, l], axis=0)

            count_edge = 0
            count_dem = 0
            count_sp = 0
            # equations for each edges:
            for i in range(self.n_edges):
                edg = self.edges[i]
                n_from = self.find_node[edg['from']]
                n_to = self.find_node[edg['to']]

                # edge equations:
                if edg['edge control type'] == 'passive':
                    # for mf = 0; set T_start = T_a -> leads to T_end = T_a in eq_pipe_T and all zero gradients
                    temp = tf.cond(
                        pred=mf[i, :] != 0,
                        true_fn=lambda: (mf[i, :] + tf.abs(mf[i, :])) / (2 * mf[i, :]) * T[n_from, :] + \
                                        (mf[i, :] - tf.abs(mf[i, :])) / (2 * mf[i, :]) * T[n_to, :],
                        false_fn=lambda: self.Ta)
                    l = T_end[i, :] - self.eq_pipe_T(tf.abs(mf[i, :]), temp, edg['temp_loss_coeff'], edg['length [m]'])
                    if verbose:
                        print(f'T: {i}, {edg["index"]}, {l}')
                        if edg['index'] == 'sup_access_HeatEx_Woog_DaNo':
                            print('break')
                    if mode == 'gradient':
                        loss += tf.reduce_sum(l ** 2, axis=0)
                    else:
                        loss = tf.concat([loss, tf.expand_dims(l, axis=0)], axis=0)
                    count_edge += 1
                    l = (p[n_from] - p[n_to] - self.eq_pipe_p(mf[i], edg['fd_nom'],
                                                              edg['length [m]'] * edg['bend_factor'],
                                                              edg['diameter']))
                    if adjust_for_hight:
                        if mf[i] > 0:
                            l += self.eq_pipe_p_static(self.nodes[n_from]['coordinate'][-1], self.nodes[n_to]['coordinate'][-1])
                        else:
                            l -= self.eq_pipe_p_static(self.nodes[n_from]['coordinate'][-1],
                                                       self.nodes[n_to]['coordinate'][-1])
                    if verbose:
                        print(f'p: {i}, {edg["index"]}, {l}')
                    if mode == 'gradient':
                        loss += tf.reduce_sum(l ** 2, axis=0)
                    else:
                        loss = tf.concat([loss, tf.expand_dims(l, axis=0)], axis=0)
                    count_edge += 1
                elif edg['index'] in demands.keys():
                    if self.use_demands:
                        l = (self.eq_demand_Q(mf[i], Q_heat[self.dem_ind[edg['index']]], T[n_from], T_end[i]))
                        # else:
                        #     l = T_end[i] - Q_T_end[self.act_edg_ind[edg['index']]]
                        if mode == 'gradient':
                            loss += tf.reduce_sum(l ** 2, axis=0)
                        else:
                            loss = tf.concat([loss, tf.expand_dims(l, axis=0)], axis=0)
                        count_dem += 1
                elif edg['index'] in self.heatings.keys() and use_heating:
                    l = (self.eq_demand_Q(mf[i], Q_heat[-1], T[n_from], T_end[i]))
                    if mode == 'gradient':
                        loss += tf.reduce_sum(l ** 2, axis=0)
                    else:
                        loss = tf.concat([loss, tf.expand_dims(l, axis=0)], axis=0)
                    # print(f'counter: {len(loss)}, dem {edg["index"]}, loss value: {l}')

        if mode == 'forwardpass':
            return loss
        elif mode == 'gradient':
            gradients = tape.gradient(loss, [T, mf, p, T_end])
            masked_gradients = [grad * self.masks[i] for i, grad in enumerate(gradients)]
            return masked_gradients, loss
        elif mode == 'jacobian':
            jacobian = tape.jacobian(loss, [T, mf, p, T_end], experimental_use_pfor=False)
            jacobian = [tf.squeeze(jacobian[i]) for i in range(len(jacobian))]
            # masked_jacobian = [tf.boolean_mask(jacobian[i], self.masks[i][:, 0], axis=1) for i in range(len(jacobian))]
            masked_jacobian = [jac @ self.mask_matrix_dense[i] for i, jac in enumerate(jacobian)]
            return masked_jacobian, loss
        elif mode == 'demand jacobian':
            # implicit function theorem to calculate
            #        d(T, mf p, T_end) / d(Q_heat) = - (d(loss) / d(T, mf, p, T_end)) ^-1 * (d(loss) / d(Q_heat)
            # <=> -  d(loss) / d(T, mf, p, T_end) * d(T, mf p, T_end) / d(Q_heat) = (d(loss) / d(Q_heat)
            #                                       |--      return value    ---|
            jacobian_sv = tape.jacobian(loss, [T, mf, p, T_end], experimental_use_pfor=False)
            jacobian_sv = [tf.squeeze(jacobian_sv[i]) for i in range(len(jacobian_sv))]
            jacobian_sv = [tf.boolean_mask(jacobian_sv[i], self.masks[i][:, 0], axis=1) for i in
                           range(len(jacobian_sv))]
            jacobian_sv = tf.concat(jacobian_sv, axis=1)
            jacobian_dem = tape.jacobian(loss, Q_heat, experimental_use_pfor=False)
            jacobian_dem = tf.squeeze(jacobian_dem)

            # use pseudoinverse if jacobian is not invertible
            # if tf.linalg.svd(jacobian_sv, compute_uv=False)[-1] == 0:
            # jac = - tf.linalg.pinv(jacobian_sv, rcond=1.e-15) @ jacobian_dem
            if tf.rank(jacobian_dem) == 1:
                jacobian_dem = tf.expand_dims(jacobian_dem, axis=-1)
            jac = -tf.linalg.lstsq(matrix=jacobian_sv, rhs=jacobian_dem, fast=False)
            return jac
        elif mode == 'temperature jacobian':
            jacobians = tape.jacobian(loss, [T, mf, p, T_end], experimental_use_pfor=False)
            jacobians = [tf.squeeze(jacobians[i]) for i in range(len(jacobians))]
            # jacobian_sv = [tf.boolean_mask(jacobian_sv[i], self.masks[i][:, 0], axis=1) for i in range(len(jacobian_sv))]
            jacobian_sv = tf.concat(jacobians, axis=1)
            # identify relevant temperatures at the node behind a supply or demand
            active_edges_ret_node = [self.find_node[self.edges[self.find_edge[ae]]['to']] for ae in self.demands.keys()]
            active_edges_ret_node.extend([self.find_node[self.edges[self.find_edge[ae]]['to']] for ae in self.heatings.keys()])
            jacobian_temp = tf.gather(jacobians[0], active_edges_ret_node, axis=1)

            jac = -tf.linalg.lstsq(matrix=jacobian_sv, rhs=jacobian_temp, fast=False)
            return jac
        elif mode == 'demand and temperature jacobian':
            jacobians = tape.jacobian(loss, [T, mf, p, T_end], experimental_use_pfor=False)
            jacobians = [tf.squeeze(jacobians[i]) for i in range(len(jacobians))]
            jacobian_sv = tf.concat(jacobians, axis=1)

            # demands:
            jacobian_dem = tape.jacobian(loss, Q_heat, experimental_use_pfor=False)
            jacobian_dem = tf.squeeze(jacobian_dem)
            # temperatures:
            active_edges_ret_node = [self.find_node[self.edges[self.find_edge[ae]]['to']] for ae in self.demands.keys()]
            active_edges_ret_node.extend(
                [self.find_node[self.edges[self.find_edge[ae]]['to']] for ae in self.heatings.keys()])
            jacobian_temp = tf.gather(jacobians[0], active_edges_ret_node, axis=1)

            # solve linear system of equations for the implicit function theorem
            jac = -tf.linalg.lstsq(matrix=jacobian_sv, rhs=tf.concat([jacobian_dem, jacobian_temp], axis=1), fast=False)
            return jac

    def get_demand_from_grid(self, heating=False):
        """
        calculates the power consumption that corresponds to the actual temperatures and mass flows at the demands

        if heating=True: return the power of the slack demand as well
        """
        for i, dem in enumerate(self.demands.keys()):
            dem_pos = self.find_edge[dem]
            Ts = self.T[self.find_node[self.edges[dem_pos]['from']]]
            dem_val = self.eq_demand_Q(self.mf[dem_pos], 0, Ts, self.T_end[dem_pos])
            self.Q_heat[self.dem_ind[dem]].assign(tf.squeeze(dem_val))
        if heating:
            e_ind = list(self.heatings.keys())[0]
            heat_pos = self.find_edge[e_ind]
            Ts = self.T[self.find_node[self.edges[heat_pos]['from']]]
            dem_val = self.eq_demand_Q(self.mf[heat_pos], 0, Ts, self.T_end[heat_pos])
            return tf.Variable(tf.concat([self.Q_heat, dem_val], axis=0))
        else:
            return self.Q_heat

    def linear_cycle_equations(self, cycles):
        """
        returns the matrix and right hand side for the cycle equations A * mf = b
        """
        A_c = np.zeros(shape=(len(cycles), self.n_edges), dtype='float64')
        b_c = np.zeros(shape=(len(cycles)), dtype='float64')
        for j, c in enumerate(cycles):
            for e, direction in c:
                i = self.find_edge[e]
                edg = self.edges[i]
                # pressure_loss is given by mf * dp_mf
                # A_c @ mf -> sum up all pressure losses among the cycle
                dp_mf = self.eq_pipe_p_lin(self.mf[i], tf.constant(edg['fd_nom'], dtype=tf.float64),
                                           tf.constant(edg['length [m]'] * edg['bend_factor'], dtype=tf.float64),
                                           tf.constant(edg['diameter'], dtype=tf.float64))

                A_c[j, i] = {'forward': 1, 'reverse': -1}[direction] * dp_mf
                b = self.eq_pipe_p(self.mf[i], tf.constant(edg['fd_nom'], dtype=tf.float64),
                                   tf.constant(edg['length [m]'] * edg['bend_factor'], dtype=tf.float64),
                                   tf.constant(edg['diameter'], dtype=tf.float64))

                b_c[j] += b
        b = tf.constant(b_c, dtype=tf.float64)
        A = tf.constant(A_c, dtype=tf.float64)
        return A, b

    def solve_massflow_fixed_temp(self, mf_vals=None, cycles=[], alpha=1.):
        '''
        solves the massflow conservation equations as well as the heat demand equations for fixed tempeatures
        if cycles are apparent: add condition for the pressure loss among the cycle to be zero (locally linearised)

        :param mf_vals: mass flow values for the active edges (optional), (ordered by self.act_edg_ind)
            if None: calculate the mass flows from the heat demand
        :param cycles: list of cycles, each cycle is a list of tuples (edge, direction)
        :param alpha: if alpha != 1: use exponential smoothing for the calculated mass flows at demands
        '''
        # alias state variables for readability
        A = self.A
        A_active = self.A_active
        demands = self.demands
        Q_heat = self.Q_heat
        T = self.T
        A_red = tf.sparse.slice(A, start=[0, 0], size=self.A_red_size)
        A_ext = tf.concat([tf.sparse.to_dense(A_red), tf.sparse.to_dense(A_active)], axis=0)
        # construct left hand side:
        bs = tf.Variable(tf.zeros(tf.shape(A_ext)[0], dtype=tf.float64))
        offset = self.A_red_size[0]
        d_count = 0
        for i, edg in enumerate(self.edges):
            # edg = self.edges[i]
            if edg['edge control type'] == 'active':
                if edg['index'] in demands.keys():
                    if mf_vals is None:
                        mf_dem = Q_heat[self.dem_ind[edg['index']]] / \
                                 (self.cp * (T[self.find_node[edg['from']]] - T[self.find_node[edg['to']]]))
                        # mf is set to none, if the demand is zero and the temperature at both sides reaches Ta (division 0/0)
                        mf_dem = tf.where(tf.math.is_nan(mf_dem), tf.zeros_like(mf_dem), mf_dem)
                        # demand mass flows should never be smaller than zero during internal calculations. For prosumer, the edge orientation gets switched, denoted by the key-entries in the steady-state-simulator
                        # set mf_dem = 1 to prevent everything from blowing up, 1 should be in a reasonable range
                        mf_dem = tf.where(tf.math.less(mf_dem, 0), tf.ones_like(mf_dem), mf_dem)
                    else:  # if mfs are passed:
                        mf_dem = mf_vals[self.dem_ind[edg['index']]]
                    # exponential smoothing:
                    mf_exp_s = alpha * mf_dem + (1 - alpha) * self.mf[offset + d_count]
                    bs[offset + d_count].assign(tf.squeeze(mf_exp_s))
                    d_count += 1
        # if needed: expand the calculation to incorporate cycles:
        if not cycles == []:
            A_cycles, bs_cycles = self.linear_cycle_equations(cycles)
            A_ext = tf.concat([A_ext, A_cycles], axis=0)
            bs = tf.concat([bs, bs_cycles], axis=0)

        # solve linear system of equations: - note: lstsq is faster in general,
        # however if a demand is 0 we want a true zero mass flow for the corresponding edges. The numerical precision
        # of tf.lstsq is not sufficient in these cases
        # mf_res = tf.linalg.lstsq(matrix=A_ext, rhs=tf.expand_dims(bs, axis=-1))
        mf_res = tf.linalg.solve(matrix=A_ext, rhs=tf.expand_dims(bs, axis=-1))
        return mf_res

    @tf.function
    def evaluate_mf_equations(self, mode, dem_mf_vals, cycles=[]):
        """
        calculates the loss due to mass-flow missmatches and for pressure losses in cycles if there are given any
        does not assign values - acts more like evaluate_state_equations
        """

        '''
        evaluates the mass flow equations,
        modes:
            forwardpass -> calculates the loss for given variables
            gradient -> returns loss and gradient of squared sum of the loss
            jacobian -> retruns loss and jacobian of the loss
        dem_mf_vals: 
            tensorflow variable, mass flow values ordered the same way as self.Q_heat, indexed by self.act_edg_ind!
        '''
        # alisases:
        A = self.A
        A_active = self.A_active
        demands = self.demands
        mf = self.mf

        A_red = tf.sparse.slice(A, start=[0, 0], size=self.A_red_size)
        A_ext = tf.concat([tf.sparse.to_dense(A_red), tf.sparse.to_dense(A_active)], axis=0)
        # bs = np.zeros((tf.shape(A_ext)[0].numpy(),1))
        if not hasattr(self, 'bs'):
            # bs = tf.Variable(tf.zeros((A_ext.get_shape()[0], 1)), dtype=tf.float64)
            bs = tf.Variable(tf.zeros((self.A_red_size[0] + A_active.get_shape()[0], 1), dtype=tf.float64))
            self.bs = bs
        else:
            bs = self.bs
        d_count = bs.get_shape()[0] - self.n_active_edges + 1
        # d_count = A_red.get_shape()[0]
        for i in range(self.n_edges):
            edg = self.edges[i]
            if edg['edge control type'] == 'active':
                if edg['index'] in demands.keys():
                    mf_dem = dem_mf_vals[self.dem_ind[edg['index']]]
                    bs[d_count].assign([mf_dem])
                    d_count += 1

        if tf.executing_eagerly():
            warnings.warn('mass flow equations evaluated eagerly!')
            persistence = True
        else:
            persistence = False
        with tf.GradientTape(persistent=persistence) as tape:
            loss = (tf.matmul(A_ext, mf, a_is_sparse=True)) - bs
            for c in cycles:
                l = 0
                for e, direction in c:
                    i = self.find_edge[e]
                    edg = self.edges[i]
                    if direction == 'forward':
                        l += self.eq_pipe_p(mf[i], edg['fd_nom'], edg['length [m]'] * edg['bend_factor'],
                                            edg['diameter'])
                    else:
                        l -= self.eq_pipe_p(mf[i], edg['fd_nom'], edg['length [m]'] * edg['bend_factor'],
                                            edg['diameter'])
                loss = tf.concat([loss, tf.expand_dims(l, axis=0)], axis=0)

        if mode == 'forwardpass':
            return loss
        elif mode == 'jacobian':
            return loss, tf.squeeze(tape.jacobian(loss, mf))

    def neighbours(self, node, relationship=None, passive_only=True) -> Tuple[Set[int], Set[int], List]:
        """
        returns the neighbours of a node
        input:
            node -> node to search the neighbours of, given by its indices
            relationship -> either 'parent' or 'child' or None -> determined in terms of mass flow direction
            passive_only -> boolean, if true, consider only nodes connected by passive edges

        returns:
            tuple consisting of node-indices, edge-indices and edges-dicts corresponding to the neighbours of the node
        """
        aim_dir = {'parent': -1, 'child': 1, None: 0}[relationship]

        neighbour_nodes_ind = set()
        neighbour_edges_ind = set()
        neighbour_edges = list()
        for edg_ind in range(len(self.edges)):
            e = self.edges[edg_ind]
            if passive_only & (e['edge control type'] == 'active'):
                continue
            if e['from'] == self.nodes[node]['index']:
                edg_dir = 1
                node_ind = self.find_node[e['to']]
            elif e['to'] == self.nodes[node]['index']:
                edg_dir = -1
                node_ind = self.find_node[e['from']]
            else:
                continue  # skip remaining loop if the node is no neighbour
            mf_dir = tf.sign(self.mf[edg_ind])
            if aim_dir == 0:
                neighbour_nodes_ind.add(node_ind)
                neighbour_edges_ind.add(edg_ind)
                neighbour_edges.append(e)
            # it holds true (?), that for related edges edg_dir * aim_dir * mf_dir = 1 right and -1 for wrong direction
            # if the mass flow is zero, the product is zero and the edge is not appended
            else:
                def true_fn():
                    neighbour_nodes_ind.add(node_ind)
                    neighbour_edges_ind.add(edg_ind)
                    neighbour_edges.append(e)

                def false_fn():
                    pass

                tf.cond(edg_dir * aim_dir * mf_dir == 1, true_fn, false_fn)
        return neighbour_nodes_ind, neighbour_edges_ind, neighbour_edges

    def solve_temperature_fixed_mf(self, slack_inlet=None):
        """
        solves the temperature loss and mixing equations for fixed mass flows

        caution: does only work for strictly positive mass flows at active edges
        """

        T = self.T
        mf = self.mf
        T_end = self.T_end

        # first: set T_end = T_a for all edges with zero mass flow
        # zero_mfs = tf.where(mf == 0)[:, 0]
        # for e_ind in zero_mfs:
        #     T_end[e_ind].assign([self.Ta])
        T_end.assign(tf.where(mf == 0, self.Ta, T_end))

        s_nodes = set(np.where(self.masks[0] == 0)[0])  # nodes, where the temperature is solved
        if slack_inlet is not None:
            s_nodes.add(slack_inlet)
        new_nodes = s_nodes.copy()  # nodes added in the last iteration of the loop
        # first: set T_end = T_a for all edges with zero mass flow
        while len(s_nodes) != self.n_nodes:
            # solve line temp loss for all edges connected to new_nodes
            c_nodes = set()  # candidate nodes to be solved next
            for ind in new_nodes:
                # edges_fr = [e for e in self.edges if (e['from'] == self.nodes[ind]['index']) & (e['edge control type'] == 'passive')]
                new_c_nodes, _, edges_fr = self.neighbours(ind, 'child')
                c_nodes.update(new_c_nodes)
                for edg in edges_fr:
                    i = self.find_edge[edg['index']]
                    T_end[i, :].assign(
                        self.eq_pipe_T(tf.abs(mf[i, :]), T[ind, :],
                                       tf.constant(edg['temp_loss_coeff'], dtype=tf.float64),
                                       tf.constant(edg['length [m]'], dtype=tf.float64)))
                    # c_nodes.add(self.find_node[edg['to']])

            new_nodes = set()
            # solve temperature mixing equation for all nodes where it is possible:
            for ind in c_nodes:
                p_nodes, p_edges, _ = self.neighbours(ind, 'parent')
                if p_nodes.issubset(s_nodes):  # if the temperature for all parent nodes is known:
                    temp = tf.reduce_sum(
                        tf.math.abs(tf.gather(mf, list(p_edges), axis=0)) * tf.gather(T_end, list(p_edges), axis=0)) / \
                           tf.reduce_sum(tf.math.abs(tf.gather(mf, list(p_edges), axis=0)))
                    if ind not in s_nodes:
                        T[ind].assign(tf.expand_dims(temp, axis=-1))
                    # add the new node to s_nodes and new_nodes
                    new_nodes.add(ind)
            s_nodes.update(new_nodes)

            if len(new_nodes) == 0:
                if len(s_nodes) != self.n_nodes:
                    '''
                    some nodes were not reached. This is either because:
                      a) some nodes are not connected with the supplied grid
                      b) some mass flows are zero and therefore the edges are not considered in the algorithm
                      c) something went horrible wrong
                    for a and b, set temperature to self.Ta (ambient temp), for c raise Exception
                    '''
                    missing_nodes = set(range(self.n_nodes)).difference(s_nodes)
                    for ind in missing_nodes:
                        _, edg_ind, _ = self.neighbours(ind)

                        def true_fn():
                            self.T[ind].assign([self.Ta])
                            for e_ind in edg_ind:
                                self.T_end[e_ind].assign([self.Ta])
                            s_nodes.add(ind)
                            new_nodes.add(ind)

                        def false_fn():
                            pass

                        # if the mass flow in all connected edges is zero:
                        tf.cond(tf.reduce_max(tf.math.abs(tf.gather(self.mf, list(edg_ind), axis=0))) < 1.e-8,
                                true_fn, false_fn)
                    if len(new_nodes) == 0:
                        raise TemperatureProbagationException

    def solve_pressures(self, gridmodel):
        '''
            dp = K * mf * |mf| -> this can be transformed into a Matrix multiplication:
            dp = K @ (mf . |mf|)

            pros: K has to be calculated only once and is constant even for cycles!
            grid is passed as nx.digraph object to create K-Matrix (way easier than with self.edges representation
        '''
        if not hasattr(self, 'dp_matrix'):
            K = np.zeros((self.n_nodes, self.n_edges))
            K_sparse_ind = []
            vals = []
            p0 = np.zeros((self.n_nodes, 1))
            passive_grid = gridmodel.get_passive_grid().to_undirected()
            # all pathes from all nodes to all nodes with fixed pressures
            paths = {}
            for f_dp in self.fix_dp.keys():
                if passive_grid.has_node(f_dp):
                    paths[f_dp] = nx.algorithms.single_target_shortest_path(passive_grid, f_dp)
            for i, params in enumerate(self.nodes):
                # find the path to fixed dp node:
                path, fixpoint = \
                [(paths[k].get(params['index'], ), k) for k in paths.keys() if paths[k].get(params['index'], )][0]
                # set p0:
                p0[i] = self.fix_dp[fixpoint]
                # set entries in K-matrix:
                for j in range(len(path) - 1):
                    # set sign according to edge orientation
                    try:
                        edg_params = gridmodel.edges[(path[j], path[j + 1])]
                        edg_ind = edg_params['index']
                        sign = 1
                    except KeyError:
                        edg_params = gridmodel.edges[(path[j + 1], path[j])]
                        edg_ind = edg_params['index']
                        sign = -1
                    col = self.find_edge[edg_ind]  # col: edge position in K-matrix
                    # val: value of entry for K-matrix  - pressure loss for mf = 1  (dp ~ mf^2)
                    val = sign * self.eq_pipe_p(tf.cast(1., tf.float64), tf.cast(edg_params['fd_nom'], tf.float64),
                                                tf.cast(edg_params['length [m]'] * edg_params['bend_factor'],
                                                        tf.float64),
                                                tf.cast(edg_params['diameter'], tf.float64))
                    K[i, col] = val
                    vals.append(val)
                    K_sparse_ind.append([i, col])
            Ks = tf.sparse.SparseTensor(indices=K_sparse_ind, values=vals, dense_shape=[self.n_nodes, self.n_edges])
            Ks = tf.sparse.reorder(Ks)
            self.p_0 = p0
            self.dp_matrix = Ks

        # solve pressure-equations in matrix-multiplication form
        self._solve_pressures()

    @tf.function
    def _solve_pressures(self):
        self.p.assign(self.p_0 + tf.sparse.sparse_dense_matmul(self.dp_matrix, self.mf * tf.math.abs(self.mf)))

    def evaluate_pressure_equations(self, grid):
        pass

    def add_to_variables(self, values):
        # adds the values to the not masked variables
        end = 0
        variables = [self.T, self.mf, self.p, self.T_end]
        for i in range(len(variables)):
            start = end
            end = start + int(np.sum(self.masks[i]))
            M = self.mask_matrix[i]
            variables[i].assign_add(tf.sparse.sparse_dense_matmul(M, values[start:end]))

    def report_state(self):
        # prints the current values for the state variables
        variables = [self.T, self.mf, self.p, self.T_end]
        for i in range(len(variables)):
            print(variables[i].name)
            if tf.shape(variables[i]).numpy()[0] == self.n_nodes:
                for j in range(self.n_nodes):
                    print(self.nodes[j]['index'] + '  ' + str(variables[i].numpy()[j]))
            else:
                for j in range(self.n_edges):
                    print(self.edges[j]['index'] + '  ' + str(variables[i].numpy()[j]))
            print("\n")


@tf.function
def calculate_physics_violation(n_nodes, n_edges, n_active_edges, A_red_dense, A_dense, B_mask_matrix_dense,
                                find_node, edges, Ta, cp, rho, y_pred):
    """
        this function evaluates the state equations for an input vector consisting of [T, mf, p, T_end],
        the functionality is similar to self.evaluate_state_equations('forwardpass' ...)

        some functions are decrypted to allow for a more efficient;
            - write as a stand alone function and not a stateful object (pass constant values instead of self.___)
            - don't treat state dimensions individually, but consider state vector y_pred only
            - don't include gradient tapes of any kind
            - don't consider active edges in any way -> assign zero loss for p and q equations
            - change gathering of loss values to tensor-array to allow for non-sequential calculation of dimensions
    """

    def eq_pipe_T(mf, T0, l, L):
        return (T0 - Ta) * tf.math.exp(
            tf.math.divide_no_nan(tf.cast(-l * L, tf.float64), (1000 * cp * mf))) + Ta

    def eq_pipe_p(mf, fd, l, d):
        return fd * 8 * l * mf * tf.math.abs(mf) / (np.pi ** 2 * rho * d ** 5) * 1e-5

    T_offset = 0
    T_range = range(T_offset, T_offset + n_nodes)
    mf_offset = n_nodes
    mf_range = range(mf_offset, mf_offset + n_edges)
    p_offset = n_nodes + n_edges
    p_range = range(p_offset, p_offset + n_nodes)
    T_end_offset = n_nodes + n_edges + n_nodes
    T_end_range = range(T_end_offset, T_end_offset + n_edges)

    n_entries = n_nodes * 2 + n_edges * 2 - n_active_edges - 1  # - self.n_active_edges * 3

    l = tf.matmul(A_red_dense, tf.gather(y_pred, mf_range, axis=0))
    loss = tf.reduce_sum(l ** 2, axis=0)
    B = tf.clip_by_value(
        tf.math.multiply(tf.expand_dims(A_dense, axis=0),
                         tf.expand_dims(tf.transpose(tf.gather(y_pred, mf_range, axis=0)), axis=1)), 0, 1.e7)
    l = tf.expand_dims(tf.gather(y_pred, T_range, axis=0), axis=0) - \
        (tf.matmul(B, tf.expand_dims(tf.gather(y_pred, T_end_range, axis=0), axis=0)) + Ta * 1.e-15) / \
        (tf.reduce_sum(B, axis=-1, keepdims=True) + 1.e-15)
    l = B_mask_matrix_dense @ l
    loss += tf.reduce_sum(l ** 2)

    for i, edg in enumerate(edges):
        mi = mf_offset + i
        n_from = find_node[edg['from']]
        n_to = find_node[edg['to']]

        if tf.math.equal(edg['edge control type'], 'passive'):
            # temperature equation
            temp = tf.cast(tf.cond(
                pred=y_pred[mi, :] != 0,
                true_fn=lambda: (y_pred[mi, :] + tf.abs(y_pred[mi, :])) / (2 * y_pred[mi, :]) * y_pred[
                                                                                                T_offset + n_from, :] + \
                                (y_pred[mi, :] - tf.abs(y_pred[mi, :])) / (2 * y_pred[mi, :]) * y_pred[T_offset + n_to,
                                                                                                :],
                false_fn=lambda: tf.expand_dims(Ta, axis=-1)), tf.float64)
            l = y_pred[T_end_offset + i, :] - eq_pipe_T(tf.abs(y_pred[mi, :]), temp,
                                                        edg.get('temp_loss_coeff', 0),
                                                        edg.get('length [m]', 0))
            loss += l ** 2
            # pressure equation
            l = (y_pred[p_offset + n_from, :] - y_pred[p_offset + n_to, :] -
                  eq_pipe_p(y_pred[mi, :], edg.get('fd_nom', 0),
                            edg.get('length [m]', 0) * edg.get('bend_factor', 0), edg.get('diameter', 0)))
            loss += l ** 2
    return loss
