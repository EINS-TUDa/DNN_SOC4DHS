"""
Author: Andreas Bott
year: 2024
mail: andreas.bott@eins.tu-darmstadt.de or andi-bott@web.de

This module defines the different parts of the Optimal Control problem for the automatic power control (APC) task.

The module is structured as follows:
    - state_estimation.py: contains the state estimation object and the state estimation model
    - optimisation_problem.py: contains the optimisation problem object and the optimisation interface
    - linear_state_model.py: contains the linear state model for the optimisation problem
    - DNN_lib: contains the DNN model for the state mapping

The OptimisationInterface is designed as a wrapper around the optimisation problem to handle external access.
The other classes should be treated as internal classes and should not be accessed directly.

"""


import tensorflow as tf
import numpy as np
import time
from tensorflow import keras
from colorama import Fore


# cost functions:
class Costs(object):
    def __init__(self, q1h, q1l, q2h, q2l, t1h, t1l, name='cost_function', *args, **kwargs):
        """
            quadratic cost function in the form:
            c = q1 * Delta Q + q2 * (Delta Q)^2 + t * Delta T
            h: absolute value higher than schedules_adj, i.e,. higher temp, higher production
            n: absolute value lower than schedules_adj, i.e., lower temp, lower production
        """
        self.name = name
        self.q1h = tf.cast(q1h, tf.float64)
        self.q2h = tf.cast(q2h, tf.float64)
        self.q1l = tf.cast(q1l, tf.float64)
        self.q2l = tf.cast(q2l, tf.float64)
        self.t1h = tf.cast(t1h, tf.float64)
        self.t1l = tf.cast(t1l, tf.float64)

    def __call__(self, Q_is, Q_sched, T_is, T_sched):
        delta_Q = tf.abs(Q_is) - tf.abs(Q_sched)
        delta_T = T_is - T_sched
        return tf.where(tf.greater(delta_Q, 0), self.q1h * tf.abs(delta_Q) + self.q2h * delta_Q ** 2,
                                                self.q1l * tf.abs(delta_Q) + self.q2l * delta_Q ** 2) + \
                tf.where(tf.greater(delta_T, 0), self.t1h * tf.abs(delta_T), self.t1l * tf.abs(delta_T))

# boundary condition - relaxed to a quadratic penalty
class Bound(object):
    def __init__(self, barrier, direction='leq', lmbda=100., name='bound'):
        """
        Boundary condition for the optimisation problem:

        direction: leq or geq
            leq (<=): value <= barrier
            geq (>=): value >= barrier

            lower: geq
            upper: leq
        """
        if not direction in ['leq', 'geq', '>=', '<=']:
            raise Exception(f'invalid argument for direction in bound {name}. Valid arguments for direction are '
                            f'"leq", "geq", ">=" and "<=". You passed "{direction}" instead')
        self.name = name
        self.direction = 'leq' if (direction == 'leq' or direction == '<=') else 'geq'
        self.barrier = tf.expand_dims(tf.cast(barrier, tf.float64), axis=0)
        self.lmbda = tf.cast(lmbda, tf.float64)

    def __call__(self, x):
        # apply the penalty function -> returning a squared penalty term
        _batch_dim = tf.shape(x)[0]
        if self.direction == 'leq':
            val = tf.clip_by_value(x - tf.repeat(self.barrier, _batch_dim, axis=0), 0, 1.e15) ** 2 * self.lmbda
        else:
            val = tf.clip_by_value(x - tf.repeat(self.barrier, _batch_dim, axis=0), -1.e15, 0.) ** 2 * self.lmbda
        return val

    def eval(self, x):
        # evaluate boundary violation -> return scalar violation value
        _batch_dim = tf.shape(x)[0]
        if self.direction == 'leq':
            val = tf.clip_by_value(x - tf.repeat(self.barrier, _batch_dim, axis=0), 0, 1.e15)
        else:
            val = tf.clip_by_value(x - tf.repeat(self.barrier, _batch_dim, axis=0), -1.e15, 0.)
        return val

class APC_problem(keras.layers.Layer):
    '''
        automatic power control (APC) optimisation problem:
        The class inherits from keras.layers.Layer. It is designed to be used as the only layer in the model to be used
        in combination with the keras.model.fit() function.

        structure:
            -  decision variables:
                - power at each power plant: self.q_scale * self.q
                - temperature at each power plant: self.T_q_scale * self.T_q
            - boundary conditions:
                - minimal / maximal supply powers: q_pp_min / q_pp_max
                - minimal / maximal supply temperatures: T_pp_min / T_pp_max
                - Grid constrains:
                    - minimal temperatures (T_dem_min for supply side, ambient temperature for return side)
                    - currently no constrains for pressures and mass flows
    '''

    def __init__(self, SE, powerplant_params, T_dem_min, state_mapping, lmbda, batch_size=10,
                 q_init=None, T_q_init=None, normalise_gradients=True, name='APC_problem'):
        """
        :param SE: state estimation object
        :param powerplant_params: power plant parameters
        :param T_dem_min: minimal temperature of the demand side
        :param state_mapping: state mapping object
        :param lmbda: penalty factor for boundary conditions
        :param batch_size: batch size for optimisation
        :param q_init: typical value for power plant power (used for normalisation)
        :param T_q_init: typical value for power plant temperature (used for normalisation)
        :param normalise_gradients: if True, gradients are normalised
        :param name: name of the layer
        """
        super().__init__(name=name)
        self.SE = SE
        pp_order = SE.pp_order
        self.batch_size = batch_size
        self.lmbda = lmbda
        self.normalise_gradients = normalise_gradients
        self.n_actives = SE.n_active_edges

        # normalise decision variables for numerical stability: skip last entry for q_init (slack pp)
        self.q_scale = q_init[:, :-1] if q_init is not None else -350 * tf.ones(shape=[1, len(pp_order)-1], dtype=tf.float64)
        self.T_q_scale = T_q_init if T_q_init is not None else +110 * tf.ones(shape=[1, len(pp_order)], dtype=tf.float64)
        self.q = tf.Variable(tf.ones_like(self.q_scale), dtype=tf.float64, trainable=True, name='power c')
        self.T_q = tf.Variable(tf.ones_like(self.T_q_scale), dtype=tf.float64, trainable=True, name='temp c')

        # state_mapping: DNN mapping the demands'/supplies' power values (d, q) and feed in temperatures (T_d, T_q)
        # to the grid state.
        # weights of the state mapping are not to be changed during optimisation
        state_mapping.trainable = False
        # repeat decision variables q and T_q, as they are 1D by definition. d and T_d arrays of n_samples (estimated)
        self.state_mapping = lambda d, q, T_d, T_q: \
            state_mapping([d,
                           tf.repeat(q, tf.shape(d)[0], axis=0),
                           T_d,
                           tf.repeat(T_q, tf.shape(d)[0], axis=0)])

        """
        setup cost function and boundary conditions for the optimisation problem  
        """
        # cost function:
        self.cost_fn = [Costs(name=pp, **powerplant_params[pp]) for pp in pp_order]

        # power plant restrictions for power and temperatures
        # note: powers are negative for heat supplies (more negative = more power), -> q_pp_max <= q_pp <= q_pp_min
        q_pp_min = tf.stack([powerplant_params[pp]['q_min'] for pp in pp_order], axis=0)
        self.B_min_sup = Bound(direction='<=', barrier=q_pp_min, name='minimum_generation', lmbda=lmbda)
        q_pp_max = tf.stack([powerplant_params[pp]['q_max'] for pp in pp_order], axis=0)
        self.B_max_sup = Bound(direction='>=', barrier=q_pp_max, name='maximum_generation', lmbda=lmbda)
        T_pp_min = tf.stack([powerplant_params[pp]['T_min'] for pp in pp_order], axis=0)
        self.B_min_T_sup = Bound(direction='>=', barrier=T_pp_min, name='minimal_supply_temperature', lmbda=lmbda)
        T_pp_max = tf.stack([powerplant_params[pp]['T_max'] for pp in pp_order], axis=0)
        self.B_max_T_sup = Bound(direction='<=', barrier=T_pp_max, name='maximum_supply_temperature', lmbda=lmbda)

        # state variables boundaries
        T_sup_nodes = [SE.find_node[SE.edges[SE.find_edge[ind]]['from']]
                       for ind in SE.demands.keys() if SE.demands[ind]['Power'] > 0]
        T_bounds_l = np.array([T_dem_min if i in T_sup_nodes else SE.Ta for i in range(SE.n_nodes)])
        mf_bounds_l = - np.inf * np.ones(SE.n_edges)
        p_bounds_l = - np.inf * np.ones(SE.n_nodes)
        T_end_bounds_l = - np.inf * np.ones(SE.n_edges)
        state_bound_l = np.concatenate([T_bounds_l, mf_bounds_l, p_bounds_l, T_end_bounds_l])
        self.B_state_low = Bound(direction='>=', barrier=state_bound_l, name='minimum_state_restrictions', lmbda=lmbda)

    def evaluate_opteq(self, sched_q, sched_T_q, estimated_d, estimated_T_d, q_n, T_q_n):
        """
        evaluates the equation of the optimisation problem and returns values for further processing
        :param sched_q: scheduled values for power plant power
        :param sched_T_q: scheduled values for power plant temperature
        :param estimated_d: estimated values for heat demands
        :param estimated_T_d: estimated values for heat demand temperatures
        :param q_n: normalised values for power plant power
        :param T_q_n: normalised values for power plant temperature
        """

        n_samples = tf.shape(estimated_d)[0]
        # rescale normalised values for cost function and state mapping

        with tf.GradientTape(persistent=True) as tape:
            q_c = q_n * self.q_scale
            T_c = T_q_n * self.T_q_scale

            tape.watch(q_c)
            tape.watch(T_c)

            # state mapping
            state, powers = self.state_mapping(estimated_d, q_c, estimated_T_d, T_c)
            # extract heat powers:
            hp = tf.gather(powers, self.SE.pp_pos, axis=-1)

            cost = tf.reduce_sum([cf(hp[..., i], sched_q[..., i], T_c[..., i], sched_T_q[0, i]) for i, cf in
                                  enumerate(self.cost_fn)], axis=0)

        B0 = tf.reduce_sum(self.B_max_sup(hp), axis=1)
        B1 = tf.reduce_sum(self.B_min_sup(hp), axis=1)
        B2 = tf.reduce_sum(self.B_min_T_sup(T_c))
        B3 = tf.reduce_sum(self.B_max_T_sup(T_c))
        B4 = tf.reduce_sum(self.B_state_low(state), axis=1)

        return cost, (B0, B1, tf.repeat(B2, n_samples), tf.repeat(B3, n_samples), B4), (state, powers)

    def get_loss(self, sched_q, sched_T_q, estimated_d, estimated_T_d, q_n, T_q_n):
        """
        calculates the loss of the optimisation problem.
        :parameter: see self.evaluate_opteq

        returns:
            loss: total loss including boundary conditions
            cost: value of the cost function, not including boundary violations
        """
        cost, boundaries, *(_) = \
            self.evaluate_opteq(sched_q, sched_T_q, estimated_d, estimated_T_d, q_n, T_q_n)
        loss = tf.reduce_sum([cost, *boundaries], axis=0)
        return cost, loss

    def call(self, inputs):
        """ wrapper around self.get_loss to be in line with keras.layers.Layer format requirements  """
        [estimated_q_f, estimated_T_f, sched_q, sched_T_c] = inputs
        cost, loss = self.get_loss(sched_q, sched_T_c, estimated_q_f, estimated_T_f, self.q, self.T_q)
        return loss, cost

    def get_insight(self, sched_q, sched_T_q, estimated_d, estimated_T_d, q=None, T_q=None, verbose=True, color=None):
        """
        convenience function - operations match self.get_loss,
        returns intermediate values i.e. cost, loss, boundary-penalties, estimated state and estimated power values

        :parameter: see self.evaluate_opteq
        :param verbose: print results to the console
            if verbose==1 or True: print cost, loss and boundaries
            if verbose==2 only print cost and loss no boundaries
            color: color for printing to the console
        """

        if q is not None:
            assert tf.reduce_all(q[:, :-1] == q[0, :-1]), \
                'q for not-slack power plants must be constant for all samples'
        color = Fore.LIGHTWHITE_EX if color is None else color

        q_n = q / self.q_scale if q is not None else self.q
        T_q_n = T_q / self.T_q_scale if T_q is not None else self.T_q

        if tf.shape(q_n)[-1] == tf.shape(sched_q)[-1]:
            q_n = q_n[:, :-1]  # remove slack variable

        n_samples = tf.shape(estimated_d)[0]
        # rescale all inputs such that they have the same batch size
        sched_q = tf.repeat(sched_q, n_samples, axis=0) if tf.shape(sched_q)[0] == 1 else sched_q
        sched_T_q = tf.repeat(sched_T_q, n_samples, axis=0) if tf.shape(sched_T_q)[0] == 1 else sched_T_q
        estimated_d = tf.repeat(estimated_d, n_samples, axis=0) if tf.shape(estimated_d)[0] == 1 else estimated_d
        estimated_T_d = tf.repeat(estimated_T_d, n_samples, axis=0) if tf.shape(estimated_T_d)[0] == 1 else estimated_T_d

        cost_exp, (B0, B1, B2, B3, B4), (state, powers) = \
            self.evaluate_opteq(sched_q, sched_T_q, estimated_d, estimated_T_d, q_n, T_q_n)

        # if a value of q is given, calculate the cost for this value (opteq uses the DNN prediction for q)
        if q is not None:
            T_is = T_q_n*self.T_q_scale
            q = tf.repeat(q, n_samples, axis=0) if tf.shape(q)[0] == 1 else q
            cost_act = tf.reduce_sum([cf(q[..., i], sched_q[..., i], T_is[..., i], sched_T_q[..., i])
                                  for i, cf in enumerate(self.cost_fn)], axis=0)

            loss_act = tf.reduce_sum([cost_act, B0, B1, B2, B3, B4], axis=0)
        loss_exp = tf.reduce_sum([cost_exp, B0, B1, B2, B3, B4], axis=0)

        verbose = 1 if type(verbose) == bool and verbose else verbose if type(verbose) == int else 0
        if verbose:
            slack_power = q[:, -1] if q is not None else tf.zeros(n_samples)
            res_string_0 = f'------------------\n' \
                           f'expected costs:  {tf.reduce_mean(cost_exp)}\n' \
                           f'{f"actual costs:    {tf.reduce_mean(cost_act)}" if "cost_act" in locals() else ""}\n' \
                           f'mean loss value: {tf.reduce_mean(loss_act) if "loss_act" in locals() else tf.reduce_mean(loss_exp)}\n' \
                           '------------------\n' \
                           f'regular power plant schedules_adj: \n' \
                           f'{f"    power: {np.round(q[0, :-1], 2)}kW" if q is not None else ""}\n'  \
                           f'    temp:  {np.round(T_q_n[:, :-1] * self.T_q_scale[:, :-1], 2)}°C\n' \
                           f'slack power plant schedules_adj: \n    ' \
                           f'power: (low) {np.round(tf.reduce_min(slack_power), 2)}kW; '\
                           f'(mean) {np.round(tf.reduce_mean(slack_power), 2)}kW; ' \
                           f'(max) {np.round(tf.reduce_max(slack_power),2)}kW \n    ' \
                           f'temp:  {np.round(T_q_n[:,-1]*self.T_q_scale[:,-1],2)}°C \n' \
                           '------------------'
            print(color + res_string_0)
            if verbose == 1:
                res_string_1 = f'------------------\n' \
                               'boundary functions: \n    ' \
                               f'min supply power: {tf.shape(B1)}; value: {tf.reduce_mean(B1)}\n    ' \
                               f'max supply power: {tf.shape(B0)}; value: {tf.reduce_mean(B0)}\n    ' \
                               f'min supply temp:  {tf.shape(B2)}; value: {tf.reduce_mean(B2)}\n    ' \
                               f'max supply temp:  {tf.shape(B3)}; value: {tf.reduce_mean(B3)}\n    ' \
                               f'grid boundaries:  {tf.shape(B4)}; value: {tf.reduce_mean(B4)}\n    ' \
                               '------------------'
                print(color + res_string_1)

        cost = cost_act if 'cost_act' in locals() else cost_exp
        loss = loss_act if 'cost_act' in locals() else loss_exp
        return cost, loss, B0, B1, B2, B3, B4, state, powers


class OptimisationInterface(object):
    """
    Wrapper around the optimisation problem to handle external access and include the optimisation scheme
    """
    def __init__(self, SE, q, T_q, d, T_d, state_model, powerplant_params, T_dem_min,
                 optimiser, n_steps_per_opt, boundary_lambda, lr, clip_results=False, batchsize=10, verbose=False,
                 **kwargs):
        """
        :param SE: state estimation object
        :param q: power schedules_adj for the power plants
        :param T_q: temperature schedules_adj for the power plants
        :param d: power schedules_adj for the demands
        :param T_d: temperature schedules_adj for the demands
        :param state_model: state mapping model from heat demand and temperatures to state
        :param powerplant_params: power plant parameters
        :param T_dem_min: minimum temperature of the demands
        :param optimiser: sgd-optimisation algorithm to be used
        :param n_steps_per_opt: number of steps to perform per optimisation;
            if int: constant number of steps per opt run (default: 500)
            if 'auto': number of steps is determined by the optimisation scheme
        :param boundary_lambda: lambda for the boundary constraints
        :param lr: initial learning rate for the optimisation scheme
        :param clip_results: if True, the results are clipped to the boundaries
        :param batchsize: batchsize for the optimisation scheme
        :param verbose: verbosity level for the optimisation scheme
        """
        # order of the power plants as they appear in the SE object - used to assign the powers/temps to the right plant
        self.sched_q = q
        self.sched_T_q = T_q
        self.sched_d = d
        self.sched_T_d = T_d
        self.n_steps_per_opt = n_steps_per_opt if type(n_steps_per_opt) is int else 500
        self.n_steps_done = 0  # tracker variable for debugging and optimisation
        self.batchsize = batchsize
        self.clip_results = clip_results
        if clip_results:
            self._compute_clip_values(powerplant_params, SE.pp_order)
        self.opttime = None  # time required for optimisation
        self.verbose = 0 if not verbose else verbose if type(verbose) is int else 'auto'  # passed to keras.model.fit(.)

        # save optimisation parameters, required to re-initialise the optimisation model if one is changed
        self.opt_params = {'SE': SE,
                           'powerplant_params': powerplant_params,
                           'T_dem_min': T_dem_min,
                           'state_model': state_model,
                           'boundary_lambda': boundary_lambda,
                           'sched_q': q,
                           'sched_T_q': T_q,
                           'optimiser': optimiser,
                           'batchsize': batchsize,
                           'lr': lr,
                           'n_steps_per_opt': n_steps_per_opt,
                           }
        self._init_opt_model(**self.opt_params)

    def _init_opt_model(self, SE, powerplant_params, T_dem_min, state_model, boundary_lambda,
                        sched_q, sched_T_q, batchsize, lr, n_steps_per_opt, optimiser):
        """ initialise the optimisation model """
        # Formulate the new_schedule optimisation model by wrapping the APC optimisation layer in an keras model:
        self.APC = APC_problem(SE, powerplant_params, T_dem_min, state_model, boundary_lambda,
                               q_init=sched_q, T_q_init=sched_T_q, batch_size=batchsize)
        # estimated_d, T_d, sched_q, sched_T_q
        inputs_q_f = keras.Input(shape=self.sched_d.get_shape()[-1], dtype=tf.float64, name='inputs_d')
        inputs_T_f = keras.Input(shape=self.sched_T_d.get_shape()[-1], dtype=tf.float64, name='inputs_T_d')
        inputs_q_c = keras.Input(shape=self.sched_q.get_shape()[-1], dtype=tf.float64, name='inputs_sched_q')
        inputs_T_c = keras.Input(shape=self.sched_T_q.get_shape()[-1], dtype=tf.float64, name='inputs_sched_T_q')
        loss, cost = self.APC([inputs_q_f, inputs_T_f, inputs_q_c, inputs_T_c])

        # automatic determination of number of steps via EarlyStopping callback
        if n_steps_per_opt == 'auto':
            self.callback = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)]
        else:
            self.callback = []

        self.opt_model = keras.Model([inputs_q_f, inputs_T_f, inputs_q_c, inputs_T_c], [loss, cost])
        optimiser_inst = keras.optimizers.get({'class_name': optimiser, 'config': {'learning_rate': lr}})
        self.opt_model.compile(loss=[self._mean_loss, None],
                               optimizer=optimiser_inst,
                               metrics=[])

    @staticmethod
    def _get_optimiser(identifier, **kwargs):
        """ get optimiser by identifier """
        if identifier == 'Adam':
            return keras.optimizers.Adam(**kwargs)
        else:
            raise Exception(f'optimiser {identifier} not found')

    def __setattr__(self, key, value):
        """ special behaviour for setting attributes
            -> if an attribute regarding the opt-model params is changed, the optimisation model is re-initialised
            -> if clip_results is changed to true, the corresponding clip values are calculated and stored
        """
        super().__setattr__(key, value)       # set attribute
        if hasattr(self, 'opt_params'):       # check if object is initialised (not in __init__ to avoid infinite loop)
            if key in self.opt_params.keys():
                self.opt_params[key] = value
                self._init_opt_model(**self.opt_params)

            if key == 'clip_results':
                if value:
                    self._compute_clip_values(self.opt_params['powerplant_params'], self.opt_params['SE'].pp_order)

    def _compute_clip_values(self, pp_params, pp_order):
        """ compute clip values for the optimisation results """
        # compute clip values for the optimisation results
        # -> clip values are the boundaries of the power plants
        # -> clip values are used to clip the optimisation results to the boundaries
        # -> clip values are stored in self.clip_values
        clip_values_q = np.array([[pp_params[pp]['q_min'], pp_params[pp]['q_max']] for pp in pp_order])
        clip_values_T = np.array([[pp_params[pp]['T_min'], pp_params[pp]['T_max']] for pp in pp_order])
        self.clip_values = (clip_values_q, clip_values_T)

    @staticmethod
    def _mean_loss(_, cost):
        return tf.reduce_mean(cost)

    def run_optimisation(self, d_estimates):
        """
        Main functionality of the optimisation interface;
        runs the optimisation for the given estimated demands, i.e. 'fit' the model defined in self.APC

        :param d_estimates:
        :return:
            q_c: optimised power plant power values
            T_c: optimised power plant temperature values
            cost: expected cost of the optimisation problem
            opttime: time required for the optimisation
        """

        # repeat each constant input to match dimension of sampled heat powers
        n_samples = tf.shape(d_estimates)[0]
        tic = time.localtime()
        hist = self.opt_model.fit([d_estimates,
                                   tf.repeat(self.sched_T_d, n_samples, axis=0),
                                   tf.repeat(self.sched_q, n_samples, axis=0),
                                   tf.repeat(self.sched_T_q, n_samples, axis=0)],
                                  tf.zeros_like(d_estimates),  # "true" solution, i.e. optimal loss is zero
                                  # self.n_steps_per_opt = 500 if early stopping callback is active
                                  epochs=self.n_steps_per_opt, callbacks=self.callback,
                                  batch_size=self.batchsize, verbose=0)
        toc = time.localtime()
        # extract the solution of the optimisation problem: - final cost, using the adapted schedules_adj:

        cost, *_, powers = self.APC.get_insight(self.sched_q, self.sched_T_q, d_estimates, self.sched_T_d, verbose=False)

        # decision variables are stored as normalised weights:
        q_c, T_c = self.opt_model.get_weights()
        q_c *= self.APC.q_scale
        q_c_slack = powers[:, -1]
        T_c *= self.APC.T_q_scale

        # save history and number of steps for debugging purpose
        self.history = hist
        self.n_steps_done = len(hist.history['loss'])
        ret_qs = tf.concat([tf.repeat(q_c, n_samples, axis=0), tf.expand_dims(q_c_slack, axis=1)], axis=1)
        self.opttime = time.mktime(toc) - time.mktime(tic)
        if self.clip_results:
            ret_qs = tf.clip_by_value(ret_qs, self.clip_values[0][..., 1], self.clip_values[0][..., 0])
            T_c = tf.clip_by_value(T_c, self.clip_values[1][..., 0], self.clip_values[1][..., 1])
        return ret_qs, T_c, cost, self.opttime


    """ 
    define different functions to acces the internal states and results fo the optimisation problem
    """

    def get_insight(self, d_estimates):
        """
        prints and returns intermediate values of the optimisation-problem,
            i.e. cost, loss, boundary-penalties, estimated state and estimated power values
        """
        self.APC.get_insight(self.sched_q, self.sched_T_q, d_estimates, self.sched_T_d, verbose=True)

    def report_optimisation_results(self, schedule, estimated_d, q, T_q, verbose=True, color=None):
        """
        prints and returns intermediate values of the optimisation-problem,
            i.e. cost, loss, boundary-penalties, estimated state and estimated power values
        """
        sched_q, sched_T_q, sched_T_d, = schedule
        return self.APC.get_insight(sched_q, sched_T_q, estimated_d, sched_T_d, q, T_q, verbose, color=color)

    def get_true_cost(self, schedule, actual, verbose=False):
        """
        computes the actual costs for the given initial schedule and actual power plant values
        """
        sched_q, sched_T_q = schedule
        true_q, true_T_q, = actual
        c = [cf(true_q[..., i], sched_q[..., i], true_T_q[..., i], sched_T_q[0, i]) for i, cf in enumerate(self.APC.cost_fn)]
        if verbose:
            print('True cost: ', [f'{PP}: {np.round(c[i], 3)}' for i, PP in enumerate(self.APC.SE.pp_order)])
        return tf.reduce_sum(c)

    def eval_bound(self, bound, x):
        """
        evaluates a single boundary condition given as string for the given input x
        ! does not weight the value with the penalty factor
        """
        b = getattr(self.APC, bound)
        return b.eval(x)

    def test_boundaries(self, schedule, state, verbose=False):
        """
        evaluates all boundaries for the given schedule and state
        ! does weight the values with the penalty factor
        """
        sched_q, sched_T_q = schedule
        loss_state = self.APC.B_state_low(tf.transpose(state))
        loss_temp = self.APC.B_max_T_sup(sched_T_q) + self.APC.B_min_T_sup(sched_T_q)
        loss_power = self.APC.B_max_sup(sched_q) + self.APC.B_min_sup(sched_q)
        loss_ges = tf.reduce_sum(loss_state) + tf.reduce_sum(loss_temp) + tf.reduce_sum(loss_power)
        if verbose:
            print('Loss state: ', np.round(loss_state, 3))
            print('Loss temp: ', np.round(loss_temp, 3))
            print('Loss power: ', np.round(loss_power, 3))
        return loss_state, loss_temp, loss_power, loss_ges


