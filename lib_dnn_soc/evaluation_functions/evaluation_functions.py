"""
Author: Andreas Bott
year: 2024
mail: andreas.bott@eins.tu-darmstadt.de or andi-bott@web.de

This module contains functions to evaluate the results of the optimisation process.

Only eval_results is intended to be called from outside this module. The other functions are helper functions.
"""

import tensorflow as tf
import pandas as pd
import itertools
import lib_dnn_soc.utility as util
import numpy as np

def _res_str(error_array, SE, tpe):
    # identifies violations and phrase them in a human-readable way
    violations = tf.where(tf.squeeze(error_array) != 0)
    string = ''
    for i in violations:
        values = error_array[:, i[0]]
        if tpe == 'state':
            string += f'violation for state {SE.identify_pos(i[0], verbose=False)} with value {values} \n'
        elif tpe == 'q_max':
            string += f'violation for max. supply of {SE.identify_sup(i[0], verbose=False)} with value {values} \n'
        elif tpe == 'q_min':
            string += f' violation for min. supply of {SE.identify_sup(i[0], verbose=False)} with value {values} \n'
        elif tpe == 'T_max':
            string += f'violation for max. temperature of {SE.identify_sup(i[0], verbose=False)} with value {values} \n'
        elif tpe == 'T_min':
            string += f'violation for min. temperature of {SE.identify_sup(i[0], verbose=False)} with value {values} \n'
    return string

def _write_violations(violations, SE, R):
    # order = [('state', 'state'), ('max supply', 'q_max'), ('min supply', 'q_min'),
    #          ('max temperature', 'T_max'), ('min temperature', 'T_min')]
    order = ['state', 'q_max', 'q_min', 'T_max', 'T_min']
    first = True
    for i, v in enumerate(violations):
        # _name, tpe = order[i]
        tpe = order[i]
        if tf.reduce_any(v != 0):
            if first:
                R.write(f'violations: \n')
                first = False
            R.write(_res_str(v, SE, tpe))
    if first:
        R.write(f'no violations \n')


def _print_report_schedule(SE, schedule, schedule_sl, demands, state, Resultsfile, name='', draw_graph=False):
    """
    creates a asci-style picture of the DHS and reports the most important values

    :param SE: StateEquations object
    :param schedule: schedule of the power plants without slack power
    :param schedule_sl: schedule of the power plants with slack power
    :param demands: demands of the DHS
    :param state: state vector of the DHS
    :param Resultsfile: Resultsfile object -> file to write into
    :param name: name of the schedule
    :param draw_graph: if True, a graph of the DHS is drawn

    Resulting picture:
    -> asci-style picture of the DHS (optional)
    -> power values for the plants and demands w.o. adjusting slack power
    -> power values for the plants and demands with adjusting slack power
    -> temperatures at the supply side nodes, i.e. for plants: feed-in temperature, for demands achieved supply temp.
    """
    width = 10

    n_active_edges = SE.n_active_edges
    pp_pos = SE.pp_order
    dem_pos = SE.dem_order
    # prepare active edges:
    dem_ind = {}
    pp_ind = {}
    for i, d in enumerate(dem_pos):
        dem_ind[d] = i
    for i, p in enumerate(pp_pos):
        pp_ind[p] = i
    ae_order = [f'DEM{i}' if f'DEM{i}' in dem_pos else f'PP{i}' for i in range(n_active_edges)]

    with Resultsfile as R:
        # draw grid:
        if draw_graph:
            R.write(3*' ' + '-' * (width+1) * (n_active_edges-1) + '-', end='\n')
            R.write(3*' ' + ('|' + ' ' * width) * n_active_edges, end='\n')
            R.write(3*' ' + ('|' + ' ' * width) * n_active_edges, end='\n')
            R.write(3*' ' + ''.join(f'{f"{ae_order[i]}":{width+1}}' for i in range(n_active_edges)), end='')
        R.write('\n' + name)
        R.write('\nQs ', end='')
        # print schedule:
        for ae in ae_order:
            if ae in dem_pos:
                ind = dem_ind[ae]
                R.write(f'{f"{demands[0][0, ind].numpy():2.2f}":{width+1}}', end='')
            else:
                ind = pp_ind[ae]
                R.write(f'{f"{schedule[0][0, ind].numpy():2.2f}":{width+1}}', end='')
        R.write('\nQa ', end='')
        # print schedule:
        for ae in ae_order:
            if ae in dem_pos:
                ind = dem_ind[ae]
                R.write(f'{f"{demands[0][0, ind].numpy():2.2f}":{width + 1}}', end='')
            else:
                ind = pp_ind[ae]
                R.write(f'{f"{schedule_sl[0][0, ind].numpy():2.2f}":{width + 1}}', end='')
        R.write('\nT  ', end='')
        # print temperatures:
        for ae in ae_order:
            if ae in dem_pos:
                node = [e['from'] for e in SE.edges if e['index'] == ae][0]
                value = state[SE.find_node[node]][0].numpy()
                R.write(f'{f"{value:2.2f}":{width+1}}', end='')
            else:
                ind = pp_ind[ae]
                R.write(f'{f"{schedule[1][0, ind].numpy():2.2f}":{width+1}}', end='')


def eval_results(SE, optimiser, StateSimulator, schedule, dispatch_list, dispatch_base, demands, Resultsfile,
                 powerplant_params, name, plot=False):
    """
    :param schedule: initial schedule of the power plants
    :param dispatch_list: list of dispatches for different estimated demands
    :param dispatch_base: baseline dispatch
    :param demands: true demands
    :param plot: if True, a plot of the results is created

    :return: dictionary with the results of the optimisation process

    Notation within this function:
        _sl   -> (correct) slack power
        _dp   -> dispatch
        _dq   -> dispatching powers only deprecated
        _sh   -> schedule
        _bl   -> baseline
    """

    # dispatch_list contains a constant value for each power plant in the first n-1 columns
    # the slack power in the last column depends on the estimated demands
    # -> use first row only as the real demand and therefore slack power is known here
    dispatch = [dispatch_list[0][0:1, :], dispatch_list[1]]
    dispatch_base = [dispatch_base[0][0:1, :], dispatch_base[1]] if dispatch_base is not None else None
    # calculate costs:
    state_sh_sl, slack_schedule = StateSimulator.get_state_and_slack(demands[0], schedule[0], demands[1], schedule[1])
    state_dp_sl, slack_dispatch = StateSimulator.get_state_and_slack(demands[0], dispatch[0], demands[1], dispatch[1])
    if dispatch_base is not None:
        state_bl_sl, slack_base = StateSimulator.get_state_and_slack(demands[0], dispatch_base[0], demands[1], dispatch_base[1])

    # update schedules with slack power:
    schedule_sl = [tf.tensor_scatter_nd_update(schedule[0], [[0, 1]], [slack_schedule]), schedule[1]]
    dispatch_sl = [tf.tensor_scatter_nd_update(dispatch[0], [[0, 1]], [slack_dispatch]), dispatch[1]]
    if dispatch_base is not None:
        dispatch_base_sl = [tf.tensor_scatter_nd_update(dispatch_base[0], [[0, 1]], [slack_base]), dispatch_base[1]]

    cost_schedule = optimiser.get_true_cost(schedule=schedule, actual=schedule_sl)
    *_, loss_schedule = optimiser.test_boundaries(schedule=schedule_sl, state=state_sh_sl)
    cost_dispatch = optimiser.get_true_cost(schedule=schedule, actual=dispatch_sl)
    *_, loss_dispatch = optimiser.test_boundaries(schedule=dispatch_sl, state=state_dp_sl)
    if dispatch_base is not None:
        cost_base = optimiser.get_true_cost(schedule=schedule, actual=dispatch_base_sl)
        *_, loss_base = optimiser.test_boundaries(schedule=dispatch_base_sl, state=state_bl_sl)

    def violations(schedule, state):
        state_violation = optimiser.eval_bound('B_state_low', tf.transpose(state))
        q_max_violation = optimiser.eval_bound('B_max_sup', schedule[0])
        q_min_violation = optimiser.eval_bound('B_min_sup', schedule[0])
        T_min_violation = optimiser.eval_bound('B_min_T_sup', schedule[1])
        T_max_violation = optimiser.eval_bound('B_max_T_sup', schedule[1])
        return state_violation, q_max_violation, q_min_violation, T_min_violation, T_max_violation

    violations_sh = violations(schedule_sl, state_sh_sl)
    violations_dp = violations(dispatch_sl, state_dp_sl)
    violations_bl = violations(dispatch_base_sl, state_bl_sl) if dispatch_base is not None else None

    # define result string:

    with Resultsfile as R:
        R.write(f'-' * 20 + '\n')
        R.write(f'Prior schedule: \n')
        _write_violations(violations_sh, SE, R)
        R.write(f'Linear Baseline: \n')
        _write_violations(violations_bl, SE, R) if dispatch_base is not None else R.write('no baseline\n')
        R.write(f'DNN Dispatch: \n')
        _write_violations(violations_dp, SE, R)
        R.write(f'-' * 20 + '\n')
        R.write('costs: \n')
        R.write(f'cost of Prior schedule: {cost_schedule} \n')
        R.write(f'cost of DNN dispatch: {cost_dispatch} \n')
        R.write(f'cost of Linear Baseline: {cost_base} \n') if dispatch_base is not None else R.write('no baseline\n')
        R.write(f'optimisation time: {optimiser.opttime} \n')
        R.write(f'-' * 20 + '\n')
        R.write(f'loss of Prior schedule: {loss_schedule} \n')
        R.write(f'loss of DNN dispatch: {loss_dispatch} \n')
        R.write(f'loss of Linear Baseline: {loss_base} \n') if dispatch_base is not None else R.write('no baseline\n')


    if plot:
        # Plot comparison:
        schedules_opt = [schedule, dispatch_base, dispatch] if dispatch_base is not None else [schedule, dispatch]
        schedules_opt_sl = [schedule_sl, dispatch_base_sl, dispatch_sl] if dispatch_base is not None else [schedule_sl, dispatch_sl]
        util.plots.comparisonplot(cost_fn=optimiser.APC.cost_fn, power_plants=SE.pp_order, pp_params=powerplant_params,
                                  schedule_orig=schedule, schedules_adj=schedules_opt, schedules_adj_sl=schedules_opt_sl,
                                  title=name, subtitles=['initial schedule', 'apply q-dp only', 'apply q and T dp'],
                                  show_boundaries=True, save=False, path='./results/', name=name)

    _print_report_schedule(SE, schedule, schedule_sl, demands, state_sh_sl, Resultsfile, name='default', draw_graph=True)
    _print_report_schedule(SE, dispatch, dispatch_sl, demands, state_dp_sl, Resultsfile, name='dispatch')
    if dispatch_base is not None:
        _print_report_schedule(SE, dispatch_base, dispatch_base_sl, demands, state_bl_sl, Resultsfile, name='baseline')
    with Resultsfile as R:
        R.write('\n' + f'-' * 20 + '\n'*3)

    violation_dict = lambda violations: dict((key, value.numpy()) for key, value in zip(['state', 'q_max', 'q_min', 'T_min', 'T_max'], violations)) \
        if violations is not None else {}

    results_dict = {
        'DNN':
            {'cost': cost_dispatch,
             'loss': loss_dispatch,
             **violation_dict(violations_dp),
             },
        'Lin':
            {'cost': cost_base if dispatch_base is not None else None,
             'loss': loss_base if dispatch_base is not None else None,
             **violation_dict(violations_bl)
             },
        'SC':
            {'cost': cost_schedule,
             'loss': loss_schedule,
             **violation_dict(violations_sh)
             }
    }
    return results_dict

def convert_results_to_df(results):
    """
    converts a list of results dictionaries to a pandas dataframe

    :param results: list of dictionaries with the results of the optimisation process
    :return: pandas dataframe
    """
    convert_to_numpy = lambda value: value.numpy() if tf.is_tensor(value) else value
    keys = results[0].keys()
    cols = pd.MultiIndex.from_tuples(itertools.product(keys, list(range(len(results)))))
    results = [pd.DataFrame.from_dict(res).map(convert_to_numpy) for res in results]
    results_df = pd.DataFrame(data=None, columns=cols)
    for i, res in enumerate(results):
        for select in keys:
            results_df.loc[:, (select, i)] = res.loc[:, select]
    return results_df


def log_results(results, logfile, SE):
    """
    logs the results of the optimisation process
    :param results: pd.DataFrame with the results of the optimisation process
    :param logfile: File to write into
    :param SE: StateEquations object
    :return: None
    """

    # aggregation functions for non-zero entries:
    agg_fun = {'n': np.count_nonzero,
               'mean': lambda x: np.mean(np.abs(x[x != 0])),
               'max': lambda x: np.max(np.abs(x[x != 0])),
               'min': lambda x: np.min(np.abs(x[x != 0])),
               }
    select = lambda i: lambda x: x[0, i]

    indices = results.index
    cols = results.columns

    # result
    log_idx = ['cost', 'loss', 'est_time', 'opt_time', 'total_time']    # can be logged directly
    sum_idx = ['q_min', 'q_max', 'T_min', 'T_max']                      # need to be aggregated
    sel_idx = ['state']                                                 # need to be aggregated and filtered

    with logfile as R:
        R.write(f'mean \n {results.loc[log_idx, :].T.groupby(level=0).agg("mean").T.to_string()} \n')
        R.write(f'min  \n {results.loc[log_idx, :].T.groupby(level=0).agg("min").T.to_string()} \n')
        R.write(f'max  \n {results.loc[log_idx, :].T.groupby(level=0).agg("max").T.to_string()} \n')

        # aggregate and log
        for idx in sum_idx:
            R.write(f'\nboundary violations for {idx} \n')
            for i, pp in enumerate(SE.pp_order):
                R.write(f'{pp}\n')
                for key, fun in agg_fun.items():
                    R.write(f'{key}\n{results.loc[idx, :].map(select(i)).groupby(level=0).agg(fun).to_string()} \n')

        for idx in sel_idx:
            R.write(f'\nstate violations for {idx} \n')
            for i in range(2 * (SE.n_nodes + SE.n_edges)):
                tpe, pos = SE.identify_pos(i, verbose=False)
                if not np.all(results.loc[idx, :].map(select(i)) == 0):
                    R.write(f'{tpe}, {pos}\n')
                    for key, fun in agg_fun.items():
                        R.write(f'{key}\n{results.loc[idx, :].map(select(i)).groupby(level=0).agg(fun).to_string()} \n')
