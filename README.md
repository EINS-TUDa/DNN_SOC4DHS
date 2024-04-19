# Stochastic Optimal Control for Nonlinear Systems based on Sampling \& Deep Learning

This repository contains the code for the paper 
"Stochastic Optimal Control for Nonlinear Systems based on Sampling \& Deep Learning" 
by [Andreas Bott](https://scholar.google.de/citations?hl=de&user=9ETRA44AAAAJ), 
[Kirill Kuroptev](https://www.eins.tu-darmstadt.de/eins/team), and 
[Florian Steinke](https://scholar.google.de/citations?hl=de&user=5XqE3bQAAAAJ).
The paper is currently under Review for the 63rd IEEE Conference on Decision and Control (CDC 2024).

For questions regarding this conde, please contact Andreas Bott (andreas.bott@eins.tu-darmstadt.de / andi-bott@web.de)

## Abstract
Stochastic optimal control requires an adequate representation of state uncertainty. For stochastic nonlinear
systems, the probability distribution over the states given measurements can often not be represented in closed form. 
In this paper, we thus propose to address this control task based on Monte-Carlo sampling, integrating the state 
estimation step with stochastic gradient descent-based control optimisation. A deep neural network approximation 
of the nonlinear system is the key to speeding up both parts. We motivate and demonstrate the approach for district 
heating systems, where the security of supply shall be guaranteed with high probability in the face of uncertain 
consumer demands. Our conceptually simple approach enables representing multimodal distributions and achieving
computation times feasible for the online operation of district heating systems.

## Citing
The paper which belongs to this code is currently under review. If you use the code, please contact 
[Andreas Bott](andreas.bott@eins.tu-darmstadt.de) (andreas.bott@eins.tu-darmstadt.de). 

## File structure of the repository 

|-- utilities/            -> contains the requirements.txt file for the python environment\
|-- grids/               -> stores topography files of the heating grid\
| \
|--Settings.py            -> Main configuration file\
|--generate_training_data.py:     -> script to generate training data; designed to run multiple instances in parallel\
|--generate_samples_fixed_pp.py   -> script to generate samples with constant pp setpoints; designed to run multiple instances in parallel\
|--train_DNN.py           -> script to train the DNN\
|--perfect_knowledge_baseline.py   -> Evaluates the true best scheduling via gridsearch\
|--plot_control_process.py     -> Exexcutes the different steps of the proposed algorithm and plot intermediate reults\
|--run_randomised_experiments.py -> Execute the proposed algorithm with random inputs and report the results\
|\
|--lib_dnn_soc: \
 |-- utility/\
  |-- plot_functions.py    -> Functions for plotting the results \
  |-- _plot_functions/      -> Belongs to plot_functions.py  \
  |-- setup.py          -> Setup functions for the experiments \
  |-- _setup_functions.py  -> Belongs to setup.py \
  |-- utility.py           -> Utility functions for the experiments \
 |-- steady_state_modelling/   \
  |-- state_equations.py    -> Contains the modelling equations  \
  |-- steady_state_solvers.py -> Contains solvers for the state equations  \
 |-- DNN_lib/           ->\
  |-- se_NN_lib.py        -> Contains the custom layers, the neural network class, etc.   \
  |-- _DNN_utility.py     -> Defines class for data handling   \
  |-- importance_sampling.py -> Implements the importance sampling algorithm  used for training data generation \
 |-- state_estimation/      \
  |-- utility.py         -> Selector function for different state estimators \
  |-- state_estimator.py   -> Parent class of all state estimators \
  |-- det_linear.py       -> Linear state estimator \
  |-- sir.py              -> Sampling importace resampling estimator (proposed algorithm) \
  |-- mcmc.py        -> Markov Chain Monte Carlo estimator \
  |-- settings_estimator.py  -> Load predefined estimation from settings file  \
 |-- dispatch_opt/         \
  |-- optimisation_problem.py  &nbsp;-> Defines the optimisation problem  \
  |-- optimisation_problem.py  &nbsp;-> Defines the optimisation problem  \
  |-- linear_state_model.py   -> Defines a linear state model to replace the DNN in the OPT problem  \
  |-- grid_search.py       -> Implements the grid search to find the true best solution  \
 |-- evaluation_functions/ \
  |-- evaluation_functions.py -> Function to evaluate the performance of the algorithms \
 \
\














