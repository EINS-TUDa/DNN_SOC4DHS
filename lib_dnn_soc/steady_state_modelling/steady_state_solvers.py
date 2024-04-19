"""
Author: Andreas Bott
year: 2024
mail: andreas.bott@eins.tu-darmstadt.de or andi-bott@web.de

This module contains solver for the steady-state equation. They are supposed to be used in combination with the
state_equations.py module.

The module features two main algorithms to solve the steady-state equations:
- A fix-point iteration algorithm, which iteratively solves the mass flow equations and the temperature equations
    until convergence is reached.
- A Newton-Raphson algorithm, which solves the complete set of equations in one step.

SE_solver combines both algorithms and is intended to be used as a general solver for the state equations.
SteadyStateSimulator and solve_SE are based on this solver and offer some additional convenience functions.
"""


import numpy as np
import tensorflow as tf
import scipy.sparse as sp
import scipy.sparse.linalg as slag
import warnings
from .state_equations import TemperatureProbagationException, StateEquations


class MaximumIterationException(Exception):
    def __init__(self, message = 'Maximum number of iteration steps reached without convergence'):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)


def _scipi_solve_linalg_wrapper(J, lhs):
    # wrapper for scipy sparse solver for linear equations, modified to resemble behaviour of tf.linalg.solve()
    J_sp = sp.csr_matrix(J.numpy())
    b = lhs.numpy()
    return tf.expand_dims(tf.constant(slag.spsolve(-J_sp, b), dtype=tf.float64), axis=1)

class my_newton_optimizer():
    """
    name might be misleading - this newton_optimizer ist not general and only works for solving the complete set of
    SS-equations using the StateEquations-Object. It is not intended to be used for other purposes.
    """
    def __init__(self):
        pass

    @tf.function
    def do_newton_step(self, function):
        # print('retracing')
        # tf.print('run')
        jacobian, cur_value = function.evaluate_state_equations('jacobian')
        desc_dir = self.get_desc_dir(jacobian, cur_value)
        self.armijo_step(function, cur_value, desc_dir, tf.concat(jacobian, axis=1))
        # print('step done')

    # @tf.function
    def get_desc_dir(self, jacobians, values):
        '''
            solve -jacobian @ desc_dir = value, returns desc_dir
            for numerical stabilty: if jacobian is not invertable, add identity matrix
        '''
        # concatenate jacobians  for different variables
        J = tf.concat(jacobians, axis=1)
        cond = lambda J: tf.math.less(tf.linalg.svd(J, compute_uv=False)[-1], 1.e-10)
        body = lambda J: [tf.add(J, tf.eye(tf.shape(J)[0], dtype=tf.float64))]
        var = J
        [J] = tf.while_loop(cond, body, [var], maximum_iterations=100)
        # usually less than 5 iterations are needed, max iter should not be reachable. However, there may be problems in
        # extreme settings, e.g. if all demands are zero. set max-iter = 100 to avoid infinite looping.
        # Sometimes it actually seems to be invertible anyways, deflecting the laws of mathematics
        # desc_dir = tf.linalg.solve(-J, values)
        desc_dir = tf.py_function(func=_scipi_solve_linalg_wrapper, inp=[J, values], Tout=tf.float64)
        return desc_dir

    # @tf.function
    def armijo_step(self, function, start_value, desc_dir, jacobian, sigma_0=1., gamma=1.e-4, max_iter=46):
        '''
            using backtracking together with the amijo-condition to find step-size
            start with stepsize sigma_0;
            gamma and max_iter for numerical stability, values seem fitting for our purposes

            armijo-condition:
            if: f(x + sigma*d) > f(x) + gamma * sigma * grad^T * d
                reduce step-size sigma (here: half)
            else:
                add step-size * times desc_dir to variables

            algorithmic: add max step-size and reduce for each subsequent step to save some calculations
        '''

        # apply max stepsize:
        function.add_to_variables(desc_dir)
        # execute loss function for point of interest
        val = function.evaluate_state_equations('forwardpass')
        # local step variables:
        sigma = tf.cast(sigma_0, dtype=tf.float64)
        gamma = tf.cast(gamma, dtype=tf.float64)

        # tf.while_loop is checked before it is run first
        cond = lambda val, sigma: \
            tf.math.greater(tf.norm(val), tf.norm(start_value + gamma * sigma * jacobian @ desc_dir))
        def body(val, sigma):
            sigma = sigma / 2
            function.add_to_variables(-desc_dir*sigma)
            val = function.evaluate_state_equations('forwardpass')
            return [val, sigma]

        val, sigma = tf.while_loop(cond, body, [val, sigma], maximum_iterations=max_iter)


class my_fixpoint_iteratror():
    """
    The solver works by assuming initial temperatures at the inlet of the active edges.
    Using these temperatures, the mass flows through the active edges are calculated.
    These are thn used to compute all other mass flows in the grid - either my matrix inversion or via NR, if cycles are present.
    The temperatures are then calculated using the mass flows and the heat balance equations.
    Using the newly computed temperatures, the mass flows are recalculated and the process is repeated till convergence.
    """
    def __init__(self, nr_prec=1.e-3, prior_fix_slack_inlet=False, prior_fix_temp=40):
        self.nr_prec = nr_prec
        self.prior_fix_slack_inlet = prior_fix_slack_inlet
        self.prior_slack_temp = prior_fix_temp

    def _true_slack_temp(self, function, slack_edge_idx, lead_edge_idx):
        # returns the temperature of the slack inlet
        mf = function.mf[slack_edge_idx].numpy()
        if mf > 0:
            return function.T_end[lead_edge_idx].numpy()
        else:
            return function.T_end[slack_edge_idx].numpy()

    def setup_inlet_skip(self, function):
        if self.prior_fix_slack_inlet:
            slack_edge = function.find_edge[list(function.heatings.keys())[0]]
            self.inlet_idx = function.find_node[function.edges[slack_edge]['from']]
            lead_edg = [e['index'] for e in function.edges if e['to'] == function.edges[slack_edge]['from']]
            if len(lead_edg) == 0:  # this occurs if the slack edge is inverted
                lead_edg = [e['index'] for e in function.edges if e['from'] == function.edges[slack_edge]['from']
                            and e != function.edges[slack_edge]]
            self.lead_edg_idx = function.find_edge[lead_edg[0]]
            self.true_slack_temp = lambda function: self._true_slack_temp(function, slack_edge, self.lead_edg_idx)

        else:
            self.inlet_idx = None  # pass 'None' instead of false as self.inlet_idx can be zero
            self.true_slack_temp = lambda function: self._true_slack_temp(function, slack_edge, slack_edge)

    def solve_mf(self, function, mf_vals=None, cycles=[], loop=None, alpha=1.):
        """ solves the mass flow equations for given supply temperatures or mass flows at demands
            function: SE-object the mf is solved for
            mf_vals: if None: determine mf_vals at demands from Q_heat and temperatures
                     else: pass tf.variable odered the same way as SE.Q_heat to pass fixed mf-values
            loop: if True, iteratively loop over the equations until mf does no longer change
                  if loop=None: set loop=True if cycles are in the grid
            alpha: if loop=False: exponential smoothing factor for demand mass flows
        """
        if loop is None:
            loop = not cycles == []
        if not loop:
            mf = function.solve_massflow_fixed_temp(mf_vals, cycles, alpha=alpha)
            function.mf.assign(mf)
        else:
            if mf_vals is None:
                # calculate mass flows from demand values:
                mf_vals = tf.Variable(np.zeros_like(function.Q_heat), dtype=tf.float64)
                for d in function.demands.keys():
                    edg = function.edges[function.find_edge[d]]
                    dt = function.T[function.find_node[edg['from']]] - function.T[function.find_node[edg['to']]]
                    mf = tf.maximum(tf.math.divide_no_nan(function.Q_heat[function.dem_ind[d]], (function.cp * dt)), 0)
                    # mass flows below zero are not accepted for active demands -> leads to problems in temp propagation
                    mf_vals[function.dem_ind[d]].assign(tf.squeeze(mf))
            # assign mass flow values to demand mass flows
            for d in function.demands.keys():
                function.mf[function.find_edge[d]].assign([mf_vals[function.dem_ind[d]]])
            # else: calculate mfs during evaluation from demands
            # last_mf = function.mf.numpy()
            for count in range(30):
                loss = self.NR_step_mf(function, mf_vals, cycles)
                # if np.max(np.abs(mf.numpy() - last_mf)) < 1.e-10:
                if tf.reduce_sum(loss**2) < self.nr_prec:
                    break
            else:
                print('max iterations NR for mf')

    def NR_step_mf(self, function, mf_vals, cycles=[]):
        if not hasattr(self, '_demMask'):
            dem_mask = np.eye(function.mf.get_shape()[0])
            for d in function.demands.keys():
                ind = function.find_edge[d]
                dem_mask[ind, ind] = 0
            self._demMask = tf.constant(dem_mask, dtype=tf.float64)

        values, J = function.evaluate_mf_equations('jacobian', mf_vals, cycles)

        # calculate descent direction, add ones to diagonal if J is not invertible
        cond = lambda J: tf.math.less(tf.linalg.svd(J, compute_uv=False)[-1], 1.e-10)
        body = lambda J: [tf.add(J, tf.eye(tf.shape(J)[0], dtype=tf.float64))]
        var = J
        [J] = tf.while_loop(cond, body, [var])
        # desc_dir = tf.linalg.solve(-J, values)
        desc_dir = tf.py_function(func=_scipi_solve_linalg_wrapper, inp=[J, values], Tout=tf.float64)

        # use armijo-algorithm to determine maximum step size - check my_newoton_optimizer for details
        # apply max stepsize:
        function.mf.assign_add(tf.matmul(self._demMask, desc_dir, a_is_sparse=True))
        # execute loss function for point of interest
        val = function.evaluate_mf_equations('forwardpass', mf_vals, cycles)
        # local step variables:
        sigma = tf.constant(1., dtype=tf.float64)
        gamma = tf.constant(1.e-4, dtype=tf.float64)
        max_iter = 46
        # tf.while_loop is checked before it is run first
        cond = lambda val, sigma: \
            tf.math.greater(tf.norm(val), tf.norm(values + gamma * sigma * J @ desc_dir))

        def body(val, sigma):
            sigma = sigma / 2
            function.mf.assign_add(-1 * tf.matmul(self._demMask, desc_dir, a_is_sparse=True) * sigma)
            val = function.evaluate_mf_equations('forwardpass', mf_vals, cycles)
            return [val, sigma]

        val, sigma = tf.while_loop(cond, body, [val, sigma], maximum_iterations=max_iter)
        return val

    def solve_temp(self, function):
        if not hasattr(self, 'inlet_idx'):
            # run only once:
            self.setup_inlet_skip(function)
        if self.prior_fix_slack_inlet:
            function.T[self.inlet_idx].assign([self.prior_slack_temp])
            function.solve_temperature_fixed_mf(slack_inlet=self.inlet_idx)
            # assign 'correct' temperature to slack inlet:
            function.T[self.inlet_idx].assign(self.true_slack_temp(function))
        else:
            function.solve_temperature_fixed_mf()

    def solve_p(self, function, grid):
        function.solve_pressures(grid)


class SE_solver(object):
    """
    combines the solvers above in one function to solve SE-objects
    """
    def __init__(self, nr_prec=1.e-5, mf_nr_prec=1.e-2, prior_fix_slack_inlet=False, prior_fix_temp=(40, 95)):
        self.FI = my_fixpoint_iteratror(mf_nr_prec,
                                        prior_fix_slack_inlet=prior_fix_slack_inlet, prior_fix_temp=prior_fix_temp[0])
        self.NR = my_newton_optimizer()
        self.nr_prec = tf.cast(nr_prec, tf.float64)
        self.mf_nr_prec = tf.cast(mf_nr_prec, tf.float64)
        self.fi_prec = tf.cast(max(1.e-2, nr_prec), tf.float64)
        self.prior_fix_temp = prior_fix_temp
        # there is no point in solving the fi step below the nr precision
        # 1.e-2 is experimentally proven to be a good choice for our systems

    def fixpoint_step(self, SE, cycles, alpha=1.):
        # executes one step of the fixpoint iteration algorithm
        try:
            self.FI.solve_mf(SE, cycles=cycles, alpha=alpha)
            self.FI.solve_temp(SE)
        except TemperatureProbagationException:
            # this might happen once in a while if the initial parameters were bad.
            # often, just trying again will work fine,
            # reasoning: in solve_temp some temperatures were fixed, leading to correct mass flows
            self.FI.solve_mf(SE, cycles=cycles)
            self.FI.solve_temp(SE)
            # if the error still raises, something is wrong (probably)
        loss = SE.evaluate_state_equations('forwardpass')
        return loss

    def fixpoint_step_internal_loop(self, SE, cycles, alpha=1., verbose=False):
        mf = SE.solve_massflow_fixed_temp(cycles=cycles, alpha=alpha)
        SE.mf.assign(mf)
        SE.solve_temperature_fixed_mf()
        prev_loss = tf.reduce_sum(SE.evaluate_state_equations('forwardpass')**2)

        alpha = 0.5
        for i in range(10):
            SE.mf.assign_add(-mf * alpha)
            SE.solve_temperature_fixed_mf()
            loss = tf.reduce_sum(SE.evaluate_state_equations('forwardpass')**2)
            if loss > prev_loss:
                SE.mf.assign_add(mf * alpha)
                SE.solve_temperature_fixed_mf()
                if verbose:
                    print(f'terminated after step: {i} with alpha: {alpha}')
                break
            else:
                if verbose:
                    print(f'alpha: {alpha}; loss: {loss}')
                prev_loss = loss
                alpha /= 2
        return prev_loss

    def newton_raphson_step(self, SE):
        # executes one step of the newton-raphson-algorithm
        self.NR.do_newton_step(SE)
        loss = SE.evaluate_state_equations('forwardpass')
        return loss

    def _flip_slack(self, SE):
        # flip slack edge:
        for e in SE.edges:
            if e['index'] == list(SE.heatings.keys())[0]:
                e['from'], e['to'] = e['to'], e['from']
                break
        # recalculate T_mask and T_end_mask:
        SE._set_masks()

        # save config and current solution
        config = SE.get_config()
        SE2 = StateEquations(**config)
        SE2.T.assign(SE.T)
        SE2.mf.assign(SE.mf)
        SE2.p.assign(SE.p)
        SE2.T_end.assign(SE.T_end)

        # redefine FixpointIterator with potentially new slack inlet:
        self.FI = my_fixpoint_iteratror(self.FI.nr_prec,
                                        prior_fix_slack_inlet=self.FI.prior_fix_slack_inlet,
                                        prior_fix_temp=self.prior_fix_temp[1])
        return SE2

    def __call__(self, SE, cycles, max_steps=100, verbose=False, alpha=1.):
        # FI-steps:
        loss = tf.cast(np.inf, tf.float64)
        step = 1
        gain = tf.Variable(np.inf, dtype=tf.float64)
        last_loss = tf.Variable(np.inf, dtype=tf.float64)
        inverted = False

        # condition for while-loop
        def cond(loss, gain, step):
            return tf.math.logical_and(tf.math.greater(gain, tf.cast(0.2, tf.float64)),
                                       tf.math.greater(loss, self.fi_prec))
        # body of while loop
        def body(loss, gain, step):
            try:
                loss = tf.reduce_sum(self.fixpoint_step(SE, cycles, alpha=alpha) ** 2)
            except TemperatureProbagationException:
                # reset state and try again:
                SE.set_init_state()
                loss = last_loss
            gain = (last_loss - loss) / loss
            last_loss.assign(loss)
            if verbose:
                print(f'FI-step: {step}; loss: {loss.numpy()}; gain: {gain.numpy()}')
            return loss, gain, step+1

        # first run:
        loss = self.fixpoint_step_internal_loop(SE, cycles, alpha=alpha, verbose=verbose)
        # loss = tf.reduce_sum(self.fixpoint_step(SE, cycles, alpha=1.) ** 2)
        gain = (last_loss - loss) / loss
        last_loss.assign(loss)
        if verbose:
            print(f'FI-step: {step}; loss: {loss.numpy()}; gain: {gain.numpy()}')
        step += 1
        # subsequent steps
        loss, gain, step = tf.while_loop(cond, body, (loss, gain, step), maximum_iterations=max_steps)

        # NR-steps:
        stale_counter = 0
        nr_step = 0
        def cond(loss, stale_counter, nr_step):
            return tf.math.greater(loss, self.nr_prec)
        def body(loss, stale_counter, nr_step):
            if stale_counter < 5:
                loss = tf.reduce_sum(self.newton_raphson_step(SE) ** 2)
                gain = (last_loss - loss) / loss
                if verbose:
                    print(f'NR-step: {nr_step}; stale_count: {stale_counter}; loss: {loss.numpy()}; gain: {gain.numpy()};')
                if tf.math.greater(5.e-2, gain):
                    stale_counter += 1
                else:
                    stale_counter = 0
                    last_loss.assign(loss)
            else:
                loss = tf.reduce_sum(self.fixpoint_step(SE, cycles, alpha=alpha) ** 2)
                if verbose:
                    gain = (last_loss - loss) / loss
                    print(f'FI-step: {nr_step}; loss: {loss.numpy()}; gain: {gain.numpy()}')
                last_loss.assign(loss)
                stale_counter = 0
            return loss, stale_counter, nr_step+1

        loss, stale_counter, nr_step = tf.while_loop(cond, body, (loss, stale_counter, nr_step), maximum_iterations=max_steps)
        if nr_step >= max_steps:
            raise MaximumIterationException()

        # flip slack back if it was inverted:
        if inverted:
            SE = self._flip_slack(SE)

class SteadyStateSimulator(object):
    def __init__(self, SE, dem_sign, cycles, verbose=False):
        self.SE = SE
        self.dem_sign = dem_sign
        self.temp_sign = tf.constant(np.append(dem_sign.numpy(), -1), dtype=dem_sign.dtype)
        self.cycles = cycles
        self.verbose = verbose
        self.solve_SE = SE_solver()

    def get_state(self, d, q, T_d, T_q, verbose=False, pass_exceptions=False):
        SE = self.SE
        SE.load_save_state()
        # join d and q into power vector
        if tf.shape(q)[1] == tf.shape(T_q)[1]:  # if q includes the slack power
            power = tf.tensor_scatter_nd_update(
                tf.scatter_nd(tf.where(self.dem_sign > 0), tf.transpose(d), shape=(self.dem_sign.shape[0], 1)),
                tf.where(self.dem_sign < 0), tf.transpose(q[:,:-1]))
        else:
            power = tf.tensor_scatter_nd_update(
                tf.scatter_nd(tf.where(self.dem_sign > 0), tf.transpose(d), shape=(self.dem_sign.shape[0], 1)),
                tf.where(self.dem_sign < 0), tf.transpose(q))
        temps = tf.tensor_scatter_nd_update(
            tf.scatter_nd(tf.where(self.temp_sign > 0), tf.transpose(T_d), shape=(self.temp_sign.shape[0], 1)),
            tf.where(self.temp_sign < 0), tf.transpose(T_q))

        SE.Q_heat.assign(power[:,0])
        for i, key in enumerate(SE.demands.keys()):
            SE.set_active_edge_temperature(key, temps[i, 0])
        SE.set_active_edge_temperature(list(SE.heatings.keys())[0], temps[-1, 0])

        try:
            self.solve_SE(SE, cycles=self.cycles, verbose=verbose)
        except MaximumIterationException as e:
            if pass_exceptions:
                raise e
            else:
                print('maximum iteration reached without convergence')
        return tf.concat([SE.T, SE.mf, SE.p, SE.T_end], axis=0)

    def init_state(self, d, q, T_d, T_q):
        return self.get_state(d, q, T_d, T_q)

    def get_supply_power(self):
        powers = self.SE.get_demand_from_grid(heating=True)
        return tf.transpose(tf.gather(powers, tf.where(self.dem_sign < 0)))

    def get_slack_power(self, d, q, T_d, T_q, verbose=False):
        self.get_state(d, q, T_d, T_q, verbose=verbose)
        powers = self.SE.get_demand_from_grid(heating=True)
        return powers[-1]

    def get_state_and_slack(self, d, q, T_d, T_q, verbose=False):
        state = self.get_state(d, q, T_d, T_q, verbose=verbose)
        powers = self.SE.get_demand_from_grid(heating=True)
        slack = powers[-1]
        return state, slack


def solve_SE(powers, temperatures, SE, cycles):
    """
    wrapper around the solve_SE function to catch exception occurring during the solving process and resting the solver
    :param powers: thermal power values
    :param temperatures: temperature values
    :param SE: State-equations object
    :param cycles: list of cycles in the grid
    :return: modified SE object in place
    """

    _solve_SE = SE_solver()
    def __solve_SE(powers, temperatures, SE, cycles):
        # set the parameter for the SE object and call the solver
        for dem in SE.demands.keys():
            ind = SE.dem_ind[dem]
            d, T = powers[ind], temperatures[ind]
            SE.Q_heat[ind].assign(d)
            SE.set_active_edge_temperature(dem, T)
        # set temperature for heating - last entry in T_vector
        ind = [k for k in SE.heatings.keys()][0]
        SE.set_active_edge_temperature(ind, temperatures[-1])
        _solve_SE(SE, cycles, verbose=False)

    # The solution process depends on the initial values for the state variables, thus on the previous solution.
    # This is usually faster than with the default values. If the solver does not converge the initial values are
    # reset to the default values of the SE object and the solver is called again.
    try:
        __solve_SE(powers, temperatures, SE, cycles)
    except (MaximumIterationException, TemperatureProbagationException) as e:
        warnings.warn(f'Exception occurred: {e} \n reset SE to init state and try solving again')
        SE.set_init_state()
        try:
            __solve_SE(powers, temperatures, SE, cycles)
        except (MaximumIterationException, TemperatureProbagationException) as e:
            warnings.warn(f'Exception occurred: {e}  \n Sample did not converge.')