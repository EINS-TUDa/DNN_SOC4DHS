"""
Author: Andreas Bott
year: 2024
mail: andreas.bott@eins.tu-darmstadt.de or andi-bott@web.de

The script initialises and trains a DNN model. The model is a surrogate model for the thermal power flow computation in
a given district heating grid.

If the existing training samples are not sufficient, the script generates new samples using the importance sampling
method described in the paper:
    "Efficient Training of Learning-Based Thermal Power Flow for 4th Generation District Heating Grids"
    by Andreas Bott, Mario Beykirch, and Florian Steinke
    currently under review at the journal "Energy", preprint available at: https://arxiv.org/abs/2403.11877

The script "generate_training_data.py" generates the same samples as this script but is designed to be run in parallel.
"""

import tensorflow as tf
import lib_dnn_soc.utility as util
import time
import lib_dnn_soc.DNN_lib as NN_lib
import os


def _number_of_existing_samples(training_data_file):
    files = 0
    samples = 0
    while os.path.exists(training_data_file(files)):
        with open(training_data_file(files), 'r') as f:
            for line in f:
                samples += 1  # count lines
        files += 1
    return samples

def generate_training_data(grid_identifier, d_prior_dist, T_prior_dist, SE, cycles, grid, file_spec, n_samples, verbose=False):
    Results = util.ResFile(name='generate_training_data.out', path='./results/', print=verbose)
    Results.log(f'Generate training data for {grid_identifier}, current time: {time.ctime()}')


    # Generate training data
    IS = NN_lib.ImportanceSampler(d_prior_dist, T_prior_dist, SE, cycles, grid, results_file=file_spec)
    IS.setup()
    IS.generate_training_samples(n_samples, include_slack=True, file_spec=file_spec, verbose=verbose)
    Results.log(f'Finished generating {n_samples} training data for {grid_identifier}, current time: {time.ctime()}')

def train_DNN(SE, cycles, d_dist, T_dist, Data, model_file,  layers_spec, n_training, n_test, n_val, batch_size,
              Results, **kwargs):
    """
    Train a DNN model for the given scenario
    :param layers_spec: List of nodes per layer in the DNN
    :param Data:        Data object to get the training data from
    :param model_file:  file to store the trained model in
    :return: training history object
    """
    # Load the model:
    x_input, y_input = Data.get_data(1000, section='train')
    model = NN_lib.build_DNN(SE, SE.dem_order, SE.pp_order, *x_input, y_input[0],
                             normalise_output_on_data=True,
                             d_prior_dist=d_dist, T_prior_dist=T_dist,
                             layer_spec=layers_spec,
                             add_demands_output=True, add_slack_output=True, cycles=cycles)

    model.save(f'{model_file("init")}')

    """ load optimiser, loss function, callbacks and compile the model  """
    opt = tf.keras.optimizers.get({'class_name': 'Adam', 'config': {'learning_rate': 0.01}})
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    # not specified lambda = 1, weight mf to compensate for the difference in scale
    loss_state = NN_lib.LossWeightedMSE(SE.n_nodes, SE.n_edges, lambda_mf=500)
    loss_power = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=opt, loss=[loss_state, loss_power], loss_weights=[1, 1])

    """ load and prepare training and validation data """
    # load training and validation data
    x_t, y_t = Data.get_data(n_training, section='train')
    x_v, y_v = Data.get_data(n_val, section='validation')
    # transform data into dataset for faster training
    x_t = tf.data.Dataset.from_tensor_slices(tuple(x_t))
    y_t = tf.data.Dataset.from_tensor_slices(tuple(y_t))
    x_v = tf.data.Dataset.from_tensor_slices(tuple(x_v))
    y_v = tf.data.Dataset.from_tensor_slices(tuple(y_v))
    data_train = tf.data.Dataset.zip((x_t, y_t))
    data_val = tf.data.Dataset.zip((x_v, y_v))
    Results.log(f'Loaded training and validation data for {grid_identifier}, start training; current time: {time.ctime()}')
    # start training process: -> 1000 epochs should never be reached due to the early stopping
    history = model.fit(data_train.batch(batch_size), validation_data=data_val.batch(batch_size),
                        epochs=1000, callbacks=[early_stopping], verbose=2)

    model.save(model_file(f'final_MSE'))
    model.save_weights(model_file(f'final_MSE'))
    Results.log(f'Finished training for {grid_identifier}, current time: {time.ctime()}')

    # evaluate the model on the test data
    x_test, y_test = Data.get_data(n_test, section='test')
    loss, state_loss, power_loss = model.evaluate(x_test, y_test, verbose=0)
    Results.log(f'final loss: {loss}, state_loss: {state_loss}, power_loss: {power_loss} for {grid_identifier}')
    return history

def __main__(grid_identifier, results_file, grid_settings, DNN_settings, verbose=False):
    Results = util.ResFile(name=results_file, path='./results/', print=verbose)
    Results.log(f'Run DNN training script for {grid_identifier}, current time: {time.ctime()}')
    SE, d_prior_dist, T_prior_dist, cycles, grid = util.setup_training(grid_identifier, grid_settings)

    # file to store the training samples in:
    training_data_file = lambda idx: f'DNN_model/data/{grid_identifier}_{grid_settings["d_prior_type"]}/{grid_identifier}_{idx}.csv'
    model_file = lambda spec: f'DNN_model/{grid_identifier}_{grid_settings["d_prior_type"]}/{grid_identifier}_{spec}'

    if not os.path.exists(f'DNN_model/data/{grid_identifier}_{grid_settings["d_prior_type"]}/'):
        os.makedirs(f'DNN_model/data/{grid_identifier}_{grid_settings["d_prior_type"]}/')

    # cont existing samples:
    n_existing_samples = _number_of_existing_samples(training_data_file)
    Results.log(f'Found {n_existing_samples} existing training samples for {grid_identifier}')

    n_samples = DNN_settings.get('n_training', 0) + DNN_settings.get('n_val', 0) + DNN_settings.get('n_test', 0)
    if n_samples > n_existing_samples:
        n_new_samples = n_samples - n_existing_samples
        Results.log(f'Compute {n_new_samples} new samples')
        save_file = lambda idx: training_data_file((n_existing_samples + idx) // DNN_settings['n_per_file'])
        generate_training_data(grid_identifier, d_prior_dist, T_prior_dist, SE, cycles, grid, save_file, n_new_samples, verbose)

    Data = NN_lib.DnnData(training_data_file(''), SE, include_slack_input=False, include_slack_output=True)
    history = train_DNN(SE, cycles, d_prior_dist, T_prior_dist, Data, model_file, **DNN_settings, Results=Results)


if __name__ == "__main__":
    grid_identifier = 'ladder5'
    results_file = 'results.out'
    from Settings import grid_settings, DNN_settings
    __main__(grid_identifier, results_file, grid_settings, DNN_settings, verbose=True)