"""
Author: Andreas Bott
year: 2024
mail: andreas.bott@eins.tu-darmstadt.de or andi-bott@web.de


Main File to run randomised experiments for the paper
    "Stochastic Optimal Control for Nonlinear System based on Sampling & Deep Learning"
    by Andreas Bott, Kirill Kuroptev, Florain Steinke, currently under review.
"""

import itertools
# fix seed: (for reproducibility)
seed = 2345

# %% imports:
import tensorflow as tf
import time
import lib_dnn_soc.utility as util
import lib_dnn_soc.dispatch_opt as opt
import lib_dnn_soc.evaluation_functions as eval
import lib_dnn_soc.state_estimation as est
import lib_dnn_soc.DNN_lib as NN
from lib_dnn_soc.dispatch_opt.linear_state_model import LinearStateModel
tf.random.set_seed(seed)


def run_estimation(EST, dhs_prior_schedules, measurement_value, Results):
    tic = time.time()
    Results.log(f'State Estimation: {EST.__class__.__name__}')
    demand_estimate, state_estimate = EST.estimate_state(**dhs_prior_schedules, meas_vals=measurement_value)
    toc = time.time()
    est_time = toc - tic
    return demand_estimate, state_estimate, est_time

def run_optimisation(OPT, demand_estimate, Results):
    tic = time.time()
    Results.log(f'Run optimisation')
    *schedule_opt, cost, _opt_time = q_opt, T_q_opt, cost, opttime = OPT.run_optimisation(demand_estimate)
    toc = time.time()
    opt_time = toc - tic
    return schedule_opt, cost, opt_time


def __main__(grid_identifier, grid_settings, state_estimator_settings, OPT_settings,
             setting, results_file, n_ramdom_tests=1, load_demands=True, verbose=False):
    # ------------------------------------------------------------------------------------------------------------------
    # %% Setup-section: (load grid, setup state estimator, setup dispatch optimiser)
    LogFile = util.ResFile(name=results_file, path='./results/', print=verbose)
    LogFile.log_testcase(grid_identifier, grid_settings, setting.settings, state_estimator_settings, OPT_settings)
    # setup classes for state estimation:
    SE, StateSimulator, Measurements, d_prior_dist, T_prior_dist, = util.setup_prior_settings(grid_identifier,
                                                                                              grid_settings)
    # setup alias for schedules
    dhs_prior_schedules = setting.get_dhs_inputs()

    # setup state models:
    DNN_state_model = NN.load_DNN(grid_identifier, grid_settings['d_prior_type'])
    Lin_state_model = LinearStateModel(StateSimulator, dhs_prior_schedules)

    # setup state estimators:
    EST_DNN = est.state_estimation_selector('SIR', **state_estimator_settings, d_prior_dist=d_prior_dist,
                                            Measurements=Measurements, SE=SE, state_model=DNN_state_model)
    EST_Lin = est.state_estimation_selector('linear', **state_estimator_settings, d_prior_dist=d_prior_dist,
                                            Measurements=Measurements, SE=SE, StateSimulator=StateSimulator)

    # setup optimisers:
    OPT_DNN = opt.Optimiser(**setting.settings, **OPT_settings, **dhs_prior_schedules, SE=SE, state_model=DNN_state_model)
    OPT_Lin = opt.Optimiser(**setting.settings, **OPT_settings, **dhs_prior_schedules, SE=SE, state_model=Lin_state_model)

    # setup evaluation function and initialise constant parameter:
    def evaluate(demand_true, Td_true, schedule_opt_DNN, schedule_opt_Lin):
        return eval.eval_results(SE, OPT_DNN, StateSimulator, schedule=setting.get_schedules(),
                                dispatch_list=schedule_opt_DNN, dispatch_base=schedule_opt_Lin,
                                demands=[demand_true, Td_true],
                                Resultsfile=LogFile, powerplant_params=setting.settings['powerplant_params'],
                                name=f'{setting.name}', plot=False)

    # setup data pipeline:
    file = (f'./data/{grid_identifier}_{abs(int(dhs_prior_schedules["q"][0, 0]))}_'
            f'{int(dhs_prior_schedules["T_q"][0, 0])}_{int(dhs_prior_schedules["T_q"][0, 1])}/samples')
    if load_demands:
        data = util.random_input_generator(how='load', dem_sel=SE.dem_pos, start=0, file=file)
    else:
        data = util.random_input_generator(how='random', StateSimulator=StateSimulator,
                                           d_prior_dist=d_prior_dist, dhs_sched=dhs_prior_schedules)
    # ------------------------------------------------------------------------------------------------------------------
    """
    Run randomised results to compare both state estimators and optimisers: 
    
    1.) generate random true demand and corresponding true state 
        -> for reproducibility: load sample from file; control this via the load_demands argument
        -> Data file is created by the script "generate_samples_fixed_pp.py"
    2.) generate measurements
    3.) run DNN state estimation
    4.) run DNN optimisation
    5.) run linear state estimation
    6.) run linear optimisation
    7.) evaluate and compare results
    
    --> repeat n_ramdom_tests times
    8.) summarise and report results
    """
    # ------------------------------------------------------------------------------------------------------------------
    results = list()
    for _ in range(n_ramdom_tests):
        # 1.) generate random true demand and corresponding true state
        demand_true, tfi_true, state_true = next(data)
        Td_true = tf.gather(tfi_true, SE.dem_pos, axis=1).numpy()
        LogFile.log(f'true demand: {demand_true}')
        # 2.) generate measurements
        measurement_value = Measurements.generate_measurement_values(state_true)
        # 3.) run DNN state estimation
        tic = time.time()
        demand_estimate_DNN, state_estimate_DNN, est_time_DNN = run_estimation(EST_DNN, dhs_prior_schedules, measurement_value, LogFile)
        # 4.) run DNN optimisation
        schedule_opt_DNN, cost_DNN, opt_time_DNN = run_optimisation(OPT_DNN, demand_estimate_DNN, LogFile)
        toc = time.time()
        total_time_DNN = toc - tic
        # 5.) run linear state estimation
        tic = time.time()
        demand_estimate_Lin, state_estimate_Lin, est_time_Lin = run_estimation(EST_Lin, dhs_prior_schedules, measurement_value, LogFile)
        # 6.) run linear optimisation
        schedule_opt_Lin, cost_Lin, opt_time_Lin = run_optimisation(OPT_Lin, demand_estimate_Lin, LogFile)
        toc = time.time()
        total_time_Lin = toc - tic
        # 7.) evaluate and compare results, add times to results:
        _results = evaluate(demand_true, Td_true, schedule_opt_DNN, schedule_opt_Lin)
        _results['DNN'] = {**_results['DNN'], **{'est_time': est_time_DNN, 'opt_time': opt_time_DNN, 'total_time': total_time_DNN}}
        _results['Lin'] = {**_results['Lin'], **{'est_time': est_time_Lin, 'opt_time': opt_time_Lin, 'total_time': total_time_Lin}}
        results.append(_results)

    # 8.) summarise and report results
    results = eval.convert_results_to_df(results)
    results.to_csv('.'.join((LogFile.path + LogFile.name).split('.')[:-1]) + '.csv', index=False)
    eval.log_results(results, LogFile, SE)




if __name__ == '__main__':
    grid_identifier = 'ladder5'
    logfile = 'logs.out'
    from Settings import grid_settings, state_estimator_settings, OPT_settings, setting

    __main__(grid_identifier, grid_settings, state_estimator_settings, OPT_settings, setting, logfile,
             n_ramdom_tests=500, verbose=True)


