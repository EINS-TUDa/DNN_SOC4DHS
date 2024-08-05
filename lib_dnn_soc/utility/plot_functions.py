"""
Author: Andreas Bott
year: 2024
mail: andreas.bott@eins.tu-darmstadt.de or andi-bott@web.de

This script executes a showcase of the state estimation and optimisation framework. It is used to generate the plots for
    "Stochastic Optimal Control for Nonlinear System based on Sampling & Deep Learning"
    by Andreas Bott, Kirill Kuroptev, Florain Steinke, currently under review.
"""

# fix seed: (for reproducibility)
seed = 2345

# %% imports:
import os
import numpy as np
import tensorflow as tf
import lib_dnn_soc.utility as util
import lib_dnn_soc.dispatch_opt as opt
import lib_dnn_soc.state_estimation as est
import lib_dnn_soc.DNN_lib as NN
import lib_dnn_soc.utility.plot_functions as plot
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import time
import tensorflow_probability as tfp
tfd = tfp.distributions
tf.random.set_seed(seed)




def __main__(grid_identifier, grid_settings, state_estimator_settings, setting, log_file, T_nodes='dems', m_selector=0,
             compute_gradients=True, plot_mf_T=True, plot_feasible=True, plot_hist_prior=True, plot_hist_opt=False,
             verbose=False, **kwargs):
    # ------------------------------------------------------------------------------------------------------------------
    plot_hist_opt = plot_hist_opt if plot_hist_opt else []
    # %% Setup-section: (load grid, setup state estimator, setup dispatch optimiser)
    LogFile = util.ResFile(name=log_file, path='./results/', print=verbose)
    LogFile.log_testcase(grid_identifier, grid_settings, setting.settings, state_estimator_settings, OPT_settings)
    # setup classes for state estimation:
    SE, StateSimulator, Measurements, d_prior_dist, T_prior_dist, = util.setup_prior_settings(grid_identifier,
                                                                                              grid_settings)
    # setup alias for schedules
    dhs_prior_schedules = setting.get_dhs_inputs()

    # setup state model:
    DNN_state_model = NN.load_DNN(grid_identifier, grid_settings['d_prior_type'])
    True_state_model = lambda d, q, T_d, T_q: StateSimulator.get_state(d, q, T_d, T_q)

    # setup state estimator:
    EST_DNN = est.state_estimation_selector('SIR', **state_estimator_settings, d_prior_dist=d_prior_dist,
                                            Measurements=Measurements, SE=SE, state_model=DNN_state_model)

    # setup optimiser:
    OPT_DNN = opt.Optimiser(**setting.settings, **OPT_settings, **dhs_prior_schedules, SE=SE,
                            state_model=DNN_state_model)

    # import data:
    file = (f'./data/{grid_identifier}_{abs(int(dhs_prior_schedules["q"][0, 0]))}_'
            f'{int(dhs_prior_schedules["T_q"][0, 0])}_{int(dhs_prior_schedules["T_q"][0, 1])}/samples')
    demand_true, tfi_true, states_true = NN.import_training_data(state_estimator_settings['SIR_settings']['n_results'], file)

    # helper functions:
    T_nodes = [SE.edges[SE.find_edge[list(SE.demands.keys())[i]]]['from'] for i in SE.dem_pos] if T_nodes == 'dems'\
        else T_nodes
    T_nodes_ind = [SE.find_node[n] for n in T_nodes]

    # ------------------------------------------------------------------------------------------------------------------
    """
    Run different analyses: 
    1.) plot T at demands over mf at PP0 for varying powers at demands
    2.) plot the prior distribution of the thermal demands
    3.) Generate measurements and run state estimation
    4.) Run the optimisation & Plot the results
    5.) Test the gradients of the state model 
    """

    # 1.) plot T over mf
    if plot_mf_T:
        fig, axs = util.plots.plot_mf_T(models=[True_state_model], SE=SE, prior=dhs_prior_schedules, T_nodes=T_nodes,
                                        **kwargs)
        plt.savefig(f'./results/{grid_identifier}_T_mf.pdf')

    # 2.) plot the prior distribution of the thermal demands
    if plot_hist_prior:
        n_hist_samples = EST_DNN.n_results
        fig, axs = plt.subplots(len(T_nodes), 1, figsize=(2*len(T_nodes)+1, 4))
        states_scheduled = StateSimulator.get_state(**dhs_prior_schedules)
        for i, ax in enumerate(axs):
            sns.histplot(states_true[:n_hist_samples, T_nodes_ind[i]], ax=ax, kde=True, bins=30)
            ax.axvline(states_scheduled[T_nodes_ind[i]], color='k', linestyle='--', linewidth=2.5)
            if 'lables_hist' in kwargs:
                ax.set_title(f'{kwargs["lables_hist"][i]}')
        plt.savefig(f'./results/{grid_identifier}_hist.png')

    # 3.) Generate measurements and run state estimation
    # Precompute several measurements to increase consistency during the selection of a measurement to be plotted for
    # the paper
    m_vals = [Measurements.generate_measurement_values(tf.transpose(states_true[i:i + 1])) for i in range(m_selector+1)]
    m_val = m_vals[m_selector]
    demand_est, states_est = EST_DNN.estimate_state(**dhs_prior_schedules, meas_vals=m_val)


    # 4.) Run the optimisation & Plot the results
    q_opt, T_q_opt, cost, _opt_time = OPT_DNN.run_optimisation(demand_est)
    states_opt = DNN_state_model([demand_est, q_opt[:, :-1],
                                 tf.repeat(dhs_prior_schedules['T_d'], EST_DNN.n_results, axis=0),
                                 tf.repeat(T_q_opt, EST_DNN.n_results, axis=0)])[0]

    # 4.1) Histogram of the initial and optimised temperature distribution
    if plot_hist_opt:
        for node in plot_hist_opt:
            index = SE.find_node[node]

        fig, axs = plt.subplots(3, 1, figsize=(7, 4), sharex=True)
        # histogram of all states
        # add red line at 80° (lower limit)
        # add black line for true value

        n_hist_samples = EST_DNN.n_results
        sns.histplot(x=states_true[:n_hist_samples, index], kde=True, ax=axs[0], bins=30)
        axs[0].axvline(80, color='red', linestyle='--', linewidth=2.5)
        axs[0].axvline(states_true[m_selector, index], color='black', linestyle='--', linewidth=2.5)
        axs[0].set_ylim(0, 100)
        axs[0].set_title(f'Original model; default control setpoints')

        sns.histplot(x=states_est[:, index], kde=True, ax=axs[1], bins=30)
        d = tf.transpose(tf.gather_nd(demand_true, [[[m_selector, i]] for i in SE.dem_pos]))
        q = dhs_prior_schedules['q']
        T_d = dhs_prior_schedules['T_d']
        T_q = dhs_prior_schedules['T_q']
        axs[1].axvline(80, color='red', linestyle='--', linewidth=2.5)
        state_dnn = DNN_state_model([d, q[:, :-1], T_d, T_q])[0]
        axs[1].axvline(state_dnn[0, index], color='black', linestyle='--', linewidth=2.5)
        axs[1].set_ylim(0, 100)
        axs[1].set_title(f'DNN model; default control setpoints')

        sns.histplot(x=states_opt[:, index], kde=True, ax=axs[2], bins=30)
        axs[2].axvline(80, color='red', linestyle='--', linewidth=2.5)
        state_dnn_opt = DNN_state_model([d, q_opt[0:1, :-1], T_d, T_q_opt])[0]
        axs[2].axvline(state_dnn_opt[0, index], color='black', linestyle='--', linewidth=2.5)
        axs[2].set_ylim(0, 100)
        axs[2].set_title(f'DNN model, optimised control setpoints')

        axs[2].set_xlabel('Temperature [°C]')
        axs[2].set_xlim(70, 102)
        plt.savefig(f'./results/{grid_identifier}_hist_opt.png')

    # 4.2.) Plot the feasible region
    if plot_feasible:
        sched_q = setting.settings['schedules']['sched_q']
        sched_T = setting.settings['schedules']['sched_T_q']
        q_min = OPT_DNN.APC.B_min_sup.barrier
        q_max = OPT_DNN.APC.B_max_sup.barrier
        mean_q_opt = tf.reduce_mean(q_opt, axis=0)
        q_10 = tfp.stats.percentile(q_opt, 10, axis=0)
        q_90 = tfp.stats.percentile(q_opt, 90, axis=0)
        errorbar = np.abs([mean_q_opt - q_10, mean_q_opt - q_90])
        price_level = np.linspace(0, 1000, 11)
        def cost_fn_init(q4, q0):
            c4 = OPT_DNN.APC.cost_fn[0](Q_is=q4, Q_sched=sched_q[0, 0], T_sched=sched_T[0, 0], T_is=sched_T[0, 0])
            c0 = OPT_DNN.APC.cost_fn[1](Q_is=q0, Q_sched=sched_q[0, 1], T_sched=sched_T[0, 1], T_is=sched_T[0, 1])
            return c4 + c0

        def cost_fn_opt(q4, q0):
            c4 = OPT_DNN.APC.cost_fn[0](Q_is=q4, Q_sched=sched_q[0, 0], T_sched=sched_T[0, 0], T_is=T_q_opt[0, 0])
            c0 = OPT_DNN.APC.cost_fn[1](Q_is=q0, Q_sched=sched_q[0, 1], T_sched=sched_T[0, 1], T_is=T_q_opt[0, 1])
            return c4 + c0


        fig, axs = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(9, 4.5))
        axs[0].sharex = axs[1]
        axs[0].sharey = axs[1]

        plot.feasibilty_plot(demand_est, cost_fn_init, q_min, q_max, DNN_state_model, SE, OPT_DNN, tpe='imshow',
                             td=dhs_prior_schedules['T_d'], tq=dhs_prior_schedules['T_q'],
                             titles=[f'Temp. A: {dhs_prior_schedules["T_q"].numpy()[0, 1]}; '
                                     f'Temp. E: {dhs_prior_schedules["T_q"].numpy()[0, 0]}'],
                             fig=fig, ax=[axs[0]], levels=price_level, add_legend=False, add_colors=False,
                             add_axis_label=False)
        plot.feasibilty_plot(demand_est, cost_fn_opt, q_min, q_max, DNN_state_model, SE, OPT_DNN, tpe='imshow',
                             td=dhs_prior_schedules['T_d'], tq=T_q_opt,
                             marker=[mean_q_opt.numpy(), errorbar[:, 0], errorbar[:, 1]],
                             titles=[f'Temp. A: {T_q_opt.numpy()[0, 1]:.2f}; Temp. E: {T_q_opt.numpy()[0, 0]:.2f}'],
                             add_axis_label=False,
                             fig=fig, ax=[axs[1]], levels=price_level, add_legend=False, add_colourbar=False,
                             add_colors=False)

        # add colorbar:
        c_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        c_ax.imshow(np.vstack([np.linspace(1, 0, 100) for _ in range(10)]).T, cmap='winter',
                    extent=(0, 10, 100, 0))
        c_ax.set_xticks([])
        c_ax.set_yticks([i for i in np.linspace(0, 100, 6)])
        c_ax.set_yticklabels([f'{int(np.round(i * 100)):d}%' for i in np.linspace(1, 0, 6)])
        # change hight:
        height = axs[1].get_position().p1[1] - axs[1].get_position().p0[1]
        c_ax.set_position([axs[1].get_position().p1[0], axs[1].get_position().p0[1], 0.05, height])

        # axs[2].set_ylim(250, 0)
        # pos = axs[2].get_position().p0
        axs[0].set_xlabel('Power plant E [kW]')
        axs[1].set_xlabel('Power plant E [kW]')
        axs[0].set_ylabel('Power plant A [kW]')
        axs[1].set_yticks([])

        plt.show(block=False)

    # 5.) Test the gradients of the state model
    if compute_gradients:
        # select grid-state for which the gradient should be computed:
        tpe = 'Temp' if not 'grad_tpe' in kwargs.keys() else kwargs['grad_tpe']
        pos = 'NS_DEM2' if not 'grad_pos' in kwargs.keys() else kwargs['grad_pos']
        grad_batch_size = 100 if not 'grad_batch_size' in kwargs.keys() else kwargs['grad_batch_size']
        offset = {'Temp': 0, 'p': SE.n_nodes, 'mf': SE.n_nodes + SE.n_edges, 'T_end': 2 * SE.n_nodes + SE.n_edges}
        idx = SE.find_node[pos] if tpe == 'Temp' or tpe == 'p' else SE.find_edge[pos]
        idx += offset[tpe]

        # generate random samples for the thermal demands & setup lists to collect the results:
        ds = d_prior_dist.sample(compute_gradients)
        schedules = setting.settings['schedules']
        grad_q_SE_list = []
        grad_T_SE_list = []
        grad_q_NN_list = []
        grad_T_NN_list = []

        # function to compute batched gradients for the DNN model:
        @tf.function()
        def est_gradients(d_batched, q_batched, T_d_batched, T_q_batched):
            with tf.GradientTape(persistent=True) as tape:
                state, powers = DNN_state_model([d_batched, q_batched, T_d_batched, T_q_batched])
            grad_q_NN = tf.gather(tape.batch_jacobian(state, q_batched), idx, axis=1)
            grad_T_NN = tf.gather(tape.batch_jacobian(state, T_q_batched), idx, axis=1)
            return grad_q_NN, grad_T_NN

        # compute gradients for true model:
        tic = time.time()
        for i, d in enumerate(ds):
            print(i)
            d = tf.expand_dims(d, axis=0)
            StateSimulator.get_state(q=schedules['sched_q'], d=d, T_d=schedules['sched_T_d'],
                                     T_q=schedules['sched_T_q'])
            # compute gradients for q and T_q:
            jac = SE.evaluate_state_equations('demand and temperature jacobian')
            # strip-out critical gradient:
            grad_q = tf.gather_nd(jac, [[idx, 3]])
            grad_T = tf.gather_nd(jac, [[idx, 7], [idx, 8]])
            grad_q_SE_list.append(grad_q)
            grad_T_SE_list.append(grad_T)
        toc = time.time()
        time_grad_true = toc - tic

        # compute gradients for DNN model:
        tic = time.time()
        n_batches = compute_gradients // grad_batch_size

        for b in range(n_batches):
            print(f'{b}/{n_batches}')
            s = slice(b * grad_batch_size, (b + 1) * grad_batch_size, 1)
            d_batched = tf.Variable(d[s])
            q_batched = tf.Variable(q[s])
            T_d_batched = tf.Variable(T_d[s])
            T_q_batched = tf.Variable(T_q[s])
            grad_q_NN, grad_T_NN = est_gradients(d_batched, q_batched, T_d_batched, T_q_batched)
            grad_q_NN_list.append(grad_q_NN)
            grad_T_NN_list.append(grad_T_NN)
        toc = time.time()
        time_grad_NN = toc - tic

        # compare the results:  (normalise gradients and compute scalar product)
        n_grad_cl = tf.linalg.normalize(tf.concat([grad_q_SE_list, grad_T_SE_list], axis=1), axis=1)[0]
        n_grad_nn = tf.linalg.normalize(tf.concat([tf.concat(grad_q_NN_list, axis=0), tf.concat(grad_T_NN_list, axis=0)], axis=1), axis=1)[0]
        sp = tf.reduce_sum(n_grad_nn * n_grad_cl, axis=1)
        with LogFile as f:
            f.write(f'mean sp: {tf.reduce_mean(sp).numpy():4f}, std: {tf.math.reduce_std(sp).numpy():4f}')
            f.write(f'time classic computation: {time_grad_true}')
            f.write(f'time DNN computation: {time_grad_NN}')

if __name__ == '__main__':
    grid_identifier = 'ladder5'
    log_file = 'logs.out'
    from Settings import grid_settings, state_estimator_settings, OPT_settings, setting

    __main__(grid_identifier, grid_settings, state_estimator_settings, setting, log_file, m_selector=7,
             compute_gradients=False, plot_mf_T=True, plot_feasible=False, plot_hist_prior=False,
             plot_hist_opt=['NS_DEM2'],
             verbose=True,
             d_modified=['DEM1', 'DEM3'], lables_mfT=['Change B', 'Change D'], lables_hist=['B', 'C', 'D'])
