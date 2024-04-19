"""
Author: Andreas Bott
year: 2024
mail: andreas.bott@eins.tu-darmstadt.de or andi-bott@web.de


This script is used to evaluate the perfect knowledge baseline.
    We assume to know the true demand and do a gridsearch to find the best set-points.
    The runtimes depend strongly on the grid resolution, i.e. the number of possible set-points.
    Different heuristics are used to exclude invalid / suboptimal set-points from the search space.
"""

# %% imports:
import numpy as np
import tensorflow as tf
import time
import lib_dnn_soc.utility as util
import lib_dnn_soc.DNN_lib as NN
import sys
import pandas as pd

import lib_dnn_soc.dispatch_opt.grid_search as grid_search


def __main__(grid_identifier, grid_settings, scenario_settings, state_estimator_settings, Setting,
             results_file, q_res=3, T_res=10, run_ramdom_tests=500, verbose=True, start=0):
    """
    run a perfect knowledge evaluation, i.e. assume to know the true demand and do a gridsearch fo find the best setpoints
    """
    # ------------------------------------------------------------------------------------------------------------------
    # %% Setup general settings and objects
    Results = util.ResFile(name=results_file, path='./results/', print=verbose)
    Results.log_testcase(grid_identifier, grid_settings, scenario_settings, state_estimator_settings, OPT_settings)
    SE, StateSimulator, Measurements, d_prior_dist, T_prior_dist, = util.setup_prior_settings(grid_identifier, grid_settings)
    state_model = NN.load_DNN(grid_identifier, grid_settings['d_prior_type'])  # required to setup OPT object
    [q_sched, T_sched], dhs_prior_schedules, _, OPT = util.setup_scenario(Setting, d_prior_dist, T_prior_dist, SE,
                                                                          Measurements, state_model, Results,
                                                                          state_estimator_settings, OPT_settings,
                                                                          verbose)
    # ------------------------------------------------------------------------------------------------------------------
    # setup grid-search
    # q_4, T_0, T_4: grid of potential set points
    # i0, j0, k0: indices of the schedule points in the grid
    # cost_no_slack: cost functions for a given set point ignoring the slack power
    # is_valid, analyse_validity: functions to check the validity of a set point and to check, which bound is invalid
    q_4, T_0, T_4, i0, j0, k0, cost_fn, cost_no_slack, is_valid, analyse_validtiy = grid_search.setup(q_res, T_res, OPT, Setting)

    # datafile for random tests - contains the true demands and states
    file = (f'./data/{grid_identifier}_{abs(int(dhs_prior_schedules["q"][0, 0]))}_'
            f'{int(dhs_prior_schedules["T_q"][0, 0])}_{int(dhs_prior_schedules["T_q"][0, 1])}/samples')
    data = util.random_input_generator(how='load', dem_sel=SE.dem_pos, start=start, file=file)

    # arrays to log the results:
    cost_log = np.zeros((run_ramdom_tests, 1))
    steps_log = np.zeros((run_ramdom_tests, 1))
    times_log = np.zeros((run_ramdom_tests, 1))
    # ------------------------------------------------------------------------------------------------------------------
    # %% run the random tests
    for run in range(run_ramdom_tests):
        tic = time.time()
        Results.log(f'Run random test {run+1}/{run_ramdom_tests}')
        demand_true, tfi_true, state_true = next(data)
        t_dem = tfi_true[0:1, 0:3]

        # initialise valid set and removal object:
        valid_points = grid_search.ValidSet(cost_no_slack)
        n_valid = valid_points.get_n_valid()
        best_cost = np.infty

        assert valid_points.cost_q_ub[i0, j0, k0] == 0, 'cost for the schedule point is not zero'
        assert valid_points.get_best_point() == (i0, j0, k0), 'initial point is not the best point'


        '''
        In order to exclude unnecessary evaluations, some heuristics are used to remove invalid or suboptimal set points
        from the search space. We refer to these heuristics as slicing, as they remove slices of the search space.
        
        valid set slicing: 
        axis: [T_0, T_4, q_4]
        edge: (i, j, k) - indices of the corner of the slice
        directions: lower, upper slice lower or higher indices
        
        - Slicing for the initial schedule: 
            - if the state is valid:
                - q4 is at a minimum -> might be traded to reduce costs for the slack power
                - --> remove all points for q4 where the slack power would deviate more from the schedule
                - --> remove that slice for all temperatures, as the temperatures are at the minimum as well
                
            - if the state is invalid:
                - vstate: state constraints violated (temperature at demand to low) 
                    - --> remove all points with lower temperature for both plants
                - vq_low/vq_high: violation of supply power boundaries
                    - only points with valid power values for q4 are tested
                    - violation for q0 (slack power)
                        - --> remove all points for which q0 is worse, analogous to the valid case
                        (theoretical exception: higher temperatures lead to higher grid losses, thus a higher slack 
                        power. If the slack power is to low, increasing the temperatures could lead to new valid points. 
                        However, it will allways be beneficial to trade some power to q4 to reduce costs.
                        Therefore, the slicing remains valid. 
                        - the same holds for lower temperatures and lower slack power)
        
        - Slicing for all other points:
            - if the state is valid:
                - if costs are lower than current best cost:
                    - --> set upper bound for costs, i.e. remove all points whose lower bound for the costs is higher
                    - --> remove all points with same power and strictly worse temperature for either plant
                        (!CAUTION: theoretically, this could lead to suboptimal results, which we accept for the sake of
                        computational efficiency and because the probability is very low for our cost function. 
                        Higher temperatures lead to higher grid losses, thus a higher slack power.
                        If the current slack power is below its schedule, increasing the temperatures may reduce the 
                        cost for the slack power plant, offsetting the higher temperature costs. 
                        This case is very unlikely due to the cost function and the limited resolution of the grid. 
                        We ignore this case. 
                        Change this if using higher costs for power changes and lower costs for temperatures)
            
            - if the state is invalid:
                - vstate: state constraints violated (temperature at demand to low)
                    - --> remove all points with lower temperature for both plants

                - vq_low/vq_high: violation of supply power boundaries
                    - only points with valid power values for q4 are tested
                    - violation for q0 (slack power)
                        - if the temperature at any plant is increased, the slack power increases and vice versa
                        - --> remove power values for 14 which worsen q0
                        - --> remove all temperature values for which the slack power worsens ase well
                        (we only slice one temperature direction due to the exception mentioned above)
            
            - in any case: 
                - remove the current point from the valid set (it has been evaluated)
        '''

        # first iteration:
        iteration = 0
        if verbose:
            print(f'iteration: {iteration}; valid points: {n_valid}')
            print(f'best point: {i0}, {j0}, {k0}')

        # first iteration - initial set point (i0, j0, k0) is evaluated:
        q_vals = tf.concat([[q_4[k0:k0 + 1]], [q_4[k0:k0 + 1]]],
                           axis=1)  # last entry (slack) is ignored by state simulator
        T_vals = tf.concat([[T_4[j0:j0 + 1]], [T_0[i0:i0 + 1]]], axis=1)
        state, q0 = StateSimulator.get_state_and_slack(demand_true, q_vals, t_dem, T_vals)

        if is_valid(state, q0):
            if q0 < q_sched[0, 1]:
                n_valid = valid_points.remove_slice(edge=(i0, j0, k0), dir=['upper', 'upper', 'lower'],  axis=[0, 1, 2])
                n_valid = valid_points.remove_slice(edge=(i0, j0, k0), dir=['lower', 'lower', 'lower'],  axis=[0, 1, 2])
            else:
                n_valid = valid_points.remove_slice(edge=(i0, j0, k0), dir=['lower', 'lower', 'upper'], axis=[0, 1, 2])
                n_valid = valid_points.remove_slice(edge=(i0, j0, k0), dir=['upper', 'upper', 'upper'], axis=[0, 1, 2])
        else:
            vstate, vq_low, vq_high = analyse_validtiy(state, q0)
            if vq_low:
                n_valid = valid_points.remove_slice(edge=(i0, j0, k0), dir=['upper', 'upper', 'lower'], axis=[0, 1, 2])
                n_valid = valid_points.remove_slice(edge=(i0, j0, k0), dir=['lower', 'lower', 'lower'], axis=[0, 1, 2])
            if vq_high:
                n_valid = valid_points.remove_slice(edge=(i0, j0, k0), dir=['lower', 'lower', 'upper'], axis=[0, 1, 2])
                n_valid = valid_points.remove_slice(edge=(i0, j0, k0), dir=['upper', 'upper', 'upper'], axis=[0, 1, 2])
            if vstate:
                n_valid = valid_points.remove_slice(edge=(i0, j0, k0), dir=['lower', 'lower'], axis=[0, 1])

        # subsequent iterations:
        while n_valid > 0:
            if verbose:
                print(f'iteration: {iteration}; valid points: {n_valid}')
            # choose the point with the lowest cost as next point to evaluate:
            i, j, k = valid_points.get_best_point()
            if verbose:
                print(f'best point: {i}, {j}, {k}')
            q_vals = tf.concat([[q_4[k:k+1]], [q_4[k:k+1]]], axis=1)  # last entry (slack) is ignored by state simulator
            T_vals = tf.concat([[T_4[j:j+1]], [T_0[i:i+1]]], axis=1)
            state, q0 = StateSimulator.get_state_and_slack(demand_true, q_vals , t_dem, T_vals)

            if is_valid(state, q0):
                cost_upper_bound = cost_fn(i, j, k, q0)
                if cost_upper_bound < best_cost:
                    best_cost = cost_upper_bound
                # mark points as invalid if costs are higher than the upper bound:
                n_valid = valid_points.set_upper_bound(cost_upper_bound)
                dir_i = 'lower' if i < i0 else 'upper'
                dir_j = 'lower' if j < j0 else 'upper'
                n_valid = valid_points.remove_slice(edge=(i, j, k), dir=[dir_i, dir_j], axis=[0, 1])

            else:
                vstate, vq_low, vq_high = analyse_validtiy(state, q0)
                if vq_low:
                    n_valid = valid_points.remove_slice(edge=(i, j, k), dir=['upper', 'upper', 'lower'], axis=[0, 1, 2])
                if vq_high:
                    n_valid = valid_points.remove_slice(edge=(i, j, k), dir=['lower', 'lower', 'upper'], axis=[0, 1, 2])
                if vstate:
                    n_valid = valid_points.remove_slice(edge=(i, j, k), dir=['lower', 'lower'], axis=[0, 1])

            n_valid = valid_points.remove_point(i, j, k)
            iteration += 1
        toc = time.time()

        # log the results:
        cost = best_cost
        cost_log[run] = cost
        steps_log[run] = iteration
        times_log[run] = toc - tic

        Results.log(f'Cost for true demand: {cost}')
        Results.log(f'Number of iterations: {iteration}')
        Results.log(f'Time for random test: {toc - tic}')

        if verbose:
            print(f'running average cost: {np.mean(cost_log[:run+1])}')
            print(f'running average steps: {np.mean(steps_log[:run+1])}')
            print(f'running average time: {np.mean(times_log[:run+1])}')

    # log the final results and save them to a file:
    Results.log(f'Average cost: {np.mean(cost_log)}')
    Results.log(f'Average steps: {np.mean(steps_log)}')
    Results.log(f'Average time: {np.mean(times_log)}')

    with open('results/perfect_baseline_cost.csv', 'a') as f:
        for c in cost_log:
            f.write(c[0].__str__())
            f.write('\n')

    with open('results/perfect_baseline_steps.csv', 'a') as f:
        for s in steps_log:
            f.write(s[0].__str__())
            f.write('\n')

    with open('results/perfect_baseline_times.csv', 'a') as f:
        for t in times_log:
            f.write(t[0].__str__())
            f.write('\n')

def read_result_files():
    """
        read the results from the files and print the average and standard deviation
    """
    file = lambda spec: f'results/perfect_baseline_{spec}.csv'
    for spec in ['cost', 'steps', 'times']:
        df = pd.read_csv(file(spec), header=None)
        average, std = df.mean()[0], df.std()[0]
        print(f'{spec}: {average} +- {std}')


if __name__ == '__main__':
    grid_identifier = 'ladder5'
    from Settings import grid_settings, scenario_settings, state_estimator_settings, OPT_settings
    from Settings import experiments

    n_per_test = 2
    start = int(sys.argv[1]) * n_per_test
    results_file = f'baseline_{sys.argv[1]}.out'
    # perfect knowledge baseline - gridsearch results:
    __main__(grid_identifier, grid_settings, scenario_settings, state_estimator_settings,
             experiments[0], results_file, start=start, run_ramdom_tests=n_per_test, verbose=True, q_res=1.5, T_res=2.5)
    read_result_files()

