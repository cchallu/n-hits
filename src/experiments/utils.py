# Cell
ENV_VARS = dict(OMP_NUM_THREADS='2',
                OPENBLAS_NUM_THREADS='2',
                MKL_NUM_THREADS='3',
                VECLIB_MAXIMUM_THREADS='2',
                NUMEXPR_NUM_THREADS='3')

# Cell
import os
import pickle
# Limit number of threads in numpy and others to avoid throttling
os.environ.update(ENV_VARS)
import random
import time
from functools import partial

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch as t
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from torch.utils.data import DataLoader

from ..data.scalers import Scaler
from ..data.tsdataset import TimeSeriesDataset, WindowsDataset, IterateWindowsDataset
from ..data.tsloader import TimeSeriesLoader
from ..models.esrnn.esrnn import ESRNN
from ..models.nbeats.nbeats import NBEATS
from ..models.nhits.nhits import NHITS
from ..models.transformer.autoformer import Autoformer

# Cell
def get_mask_dfs(Y_df, ds_in_val, ds_in_test):
    # train mask
    train_mask_df = Y_df.copy()[['unique_id', 'ds']]
    train_mask_df.sort_values(by=['unique_id', 'ds'], inplace=True)
    train_mask_df.reset_index(drop=True, inplace=True)

    train_mask_df['sample_mask'] = 1
    train_mask_df['available_mask'] = 1

    idx_out = train_mask_df.groupby('unique_id').tail(ds_in_val+ds_in_test).index
    train_mask_df.loc[idx_out, 'sample_mask'] = 0

    # test mask
    test_mask_df = train_mask_df.copy()
    test_mask_df['sample_mask'] = 0
    idx_test = test_mask_df.groupby('unique_id').tail(ds_in_test).index
    test_mask_df.loc[idx_test, 'sample_mask'] = 1

    # validation mask
    val_mask_df = train_mask_df.copy()
    val_mask_df['sample_mask'] = 1
    val_mask_df['sample_mask'] = val_mask_df['sample_mask'] - train_mask_df['sample_mask']
    val_mask_df['sample_mask'] = val_mask_df['sample_mask'] - test_mask_df['sample_mask']

    assert len(train_mask_df)==len(Y_df), \
        f'The mask_df length {len(train_mask_df)} is not equal to Y_df length {len(Y_df)}'

    return train_mask_df, val_mask_df, test_mask_df

# Cell
def get_random_mask_dfs(Y_df, ds_in_test,
                        n_val_windows, n_ds_val_window,
                        n_uids, freq):
    """
    Generates train, test and random validation mask.
    Train mask begins by avoiding ds_in_test

    Validation mask: 1) samples n_uids unique ids
                     2) creates windows of size n_ds_val_window
    Parameters
    ----------
    ds_in_test: int
        Number of ds in test.
    n_uids: int
        Number of unique ids in validation.
    n_val_windows: int
        Number of windows for validation.
    n_ds_val_window: int
        Number of ds in each validation window.
    periods: int
        ds_in_test multiplier.
    freq: str
        string that determines datestamp frequency, used in
        random windows creation.
    """
    np.random.seed(1)
    #----------------------- Train mask -----------------------#
    # Initialize masks
    train_mask_df, val_mask_df, test_mask_df = get_mask_dfs(Y_df=Y_df,
                                                            ds_in_val=0,
                                                            ds_in_test=ds_in_test)

    assert val_mask_df['sample_mask'].sum()==0, 'Muerte'

    #----------------- Random Validation mask -----------------#
    # Overwrite validation with random windows
    uids = train_mask_df['unique_id'].unique()
    val_uids = np.random.choice(uids, n_uids, replace=False)

    # Validation avoids test
    idx_test = train_mask_df.groupby('unique_id').tail(ds_in_test).index
    available_ds = train_mask_df.loc[~train_mask_df.index.isin(idx_test)]['ds'].unique()
    val_init_ds = np.random.choice(available_ds, n_val_windows, replace=False)

    # Creates windows
    val_ds = [pd.date_range(init, periods=n_ds_val_window, freq=freq) for init in val_init_ds]
    val_ds = np.concatenate(val_ds)

    # Cleans random windows from train mask
    val_idx = train_mask_df.query('unique_id in @val_uids & ds in @val_ds').index
    train_mask_df.loc[val_idx, 'sample_mask'] = 0
    val_mask_df.loc[val_idx, 'sample_mask'] = 1

    return train_mask_df, val_mask_df, test_mask_df

# Cell
def scale_data(Y_df, X_df, mask_df, normalizer_y, normalizer_x):
    mask = mask_df['available_mask'].values * mask_df['sample_mask'].values

    if normalizer_y is not None:
        scaler_y = Scaler(normalizer=normalizer_y)
        Y_df['y'] = scaler_y.scale(x=Y_df['y'].values, mask=mask)
    else:
        scaler_y = None

    if normalizer_x is not None:
        X_cols = [col for col in X_df.columns if col not in ['unique_id','ds']]
        for col in X_cols:
            scaler_x = Scaler(normalizer=normalizer_x)
            X_df[col] = scaler_x.scale(x=X_df[col].values, mask=mask)

    return Y_df, X_df, scaler_y

# Cell
def create_datasets(mc, S_df, Y_df, X_df, f_cols,
                    ds_in_test, ds_in_val):
    #------------------------------------- Available and Validation Mask ------------------------------------#
    train_mask_df, valid_mask_df, test_mask_df = get_mask_dfs(Y_df=Y_df,
                                                              ds_in_val=ds_in_val,
                                                              ds_in_test=ds_in_test)

    #---------------------------------------------- Scale Data ----------------------------------------------#
    Y_df, X_df, scaler_y = scale_data(Y_df=Y_df, X_df=X_df, mask_df=train_mask_df,
                                      normalizer_y=mc['normalizer_y'], normalizer_x=mc['normalizer_x'])

    #----------------------------------------- Declare Dataset and Loaders ----------------------------------#

    if mc['mode'] == 'simple':
        train_dataset = WindowsDataset(S_df=S_df, Y_df=Y_df, X_df=X_df,
                                       mask_df=train_mask_df, f_cols=f_cols,
                                       input_size=int(mc['n_time_in']),
                                       output_size=int(mc['n_time_out']),
                                       sample_freq=int(mc['idx_to_sample_freq']),
                                       complete_windows=mc['complete_windows'],
                                       verbose=True)

        valid_dataset = WindowsDataset(S_df=S_df, Y_df=Y_df, X_df=X_df,
                                       mask_df=valid_mask_df, f_cols=f_cols,
                                       input_size=int(mc['n_time_in']),
                                       output_size=int(mc['n_time_out']),
                                       sample_freq=int(mc['val_idx_to_sample_freq']),
                                       complete_windows=True,
                                       verbose=True)

        test_dataset = WindowsDataset(S_df=S_df, Y_df=Y_df, X_df=X_df,
                                      mask_df=test_mask_df, f_cols=f_cols,
                                      input_size=int(mc['n_time_in']),
                                      output_size=int(mc['n_time_out']),
                                      sample_freq=int(mc['val_idx_to_sample_freq']),
                                      complete_windows=True,
                                      verbose=True)
    if mc['mode'] == 'iterate_windows':
        train_dataset = IterateWindowsDataset(S_df=S_df, Y_df=Y_df, X_df=X_df,
                                              mask_df=train_mask_df, f_cols=f_cols,
                                              input_size=int(mc['n_time_in']),
                                              output_size=int(mc['n_time_out']),
                                              verbose=True)

        valid_dataset = IterateWindowsDataset(S_df=S_df, Y_df=Y_df, X_df=X_df,
                                              mask_df=valid_mask_df, f_cols=f_cols,
                                              input_size=int(mc['n_time_in']),
                                              output_size=int(mc['n_time_out']),
                                              verbose=True)

        test_dataset = IterateWindowsDataset(S_df=S_df, Y_df=Y_df, X_df=X_df,
                                             mask_df=test_mask_df, f_cols=f_cols,
                                             input_size=int(mc['n_time_in']),
                                             output_size=int(mc['n_time_out']),
                                             verbose=True)

    if mc['mode'] == 'full':
        train_dataset = TimeSeriesDataset(S_df=S_df, Y_df=Y_df, X_df=X_df,
                                          mask_df=train_mask_df, f_cols=f_cols,
                                          input_size=int(mc['n_time_in']),
                                          output_size=int(mc['n_time_out']),
                                          verbose=True)

        valid_dataset = TimeSeriesDataset(S_df=S_df, Y_df=Y_df, X_df=X_df,
                                          mask_df=valid_mask_df, f_cols=f_cols,
                                          input_size=int(mc['n_time_in']),
                                          output_size=int(mc['n_time_out']),
                                          verbose=True)

        test_dataset = TimeSeriesDataset(S_df=S_df, Y_df=Y_df, X_df=X_df,
                                         mask_df=test_mask_df, f_cols=f_cols,
                                         input_size=int(mc['n_time_in']),
                                         output_size=int(mc['n_time_out']),
                                         verbose=True)

    if ds_in_test == 0:
        test_dataset = None

    return train_dataset, valid_dataset, test_dataset, scaler_y

# Cell
def instantiate_loaders(mc, train_dataset, val_dataset, test_dataset):
    if mc['mode'] in ['simple', 'full'] :
        train_loader = TimeSeriesLoader(dataset=train_dataset,
                                        batch_size=int(mc['batch_size']),
                                        n_windows=int(mc['n_windows']),
                                        eq_batch_size=False,
                                        shuffle=True)
        if val_dataset is not None:
            val_loader = TimeSeriesLoader(dataset=val_dataset,
                                        batch_size=1,
                                        shuffle=False)
        else:
            val_loader = None

        if test_dataset is not None:
            test_loader = TimeSeriesLoader(dataset=test_dataset,
                                        batch_size=1,
                                        shuffle=False)
        else:
            test_loader = None

    elif mc['mode'] == 'iterate_windows':
        train_loader =DataLoader(dataset=train_dataset,
                                 batch_size=int(mc['batch_size']),
                                 shuffle=True,
                                 drop_last=True)

        if val_dataset is not None:
            val_loader = DataLoader(dataset=val_dataset,
                                    batch_size=1,
                                    shuffle=False)
        else:
            val_loader = None

        if test_dataset is not None:
            test_loader = DataLoader(dataset=test_dataset,
                                     batch_size=1,
                                     shuffle=False)
        else:
            test_loader = None

    return train_loader, val_loader, test_loader

# Cell
def instantiate_nbeats(mc):
    mc['n_theta_hidden'] = len(mc['stack_types']) * [ [int(mc['n_hidden']), int(mc['n_hidden'])] ]

    if mc['max_epochs'] is not None:
        lr_decay_step_size = int(mc['max_epochs'] / mc['n_lr_decays'])
    elif mc['max_steps'] is not None:
        lr_decay_step_size = int(mc['max_steps'] / mc['n_lr_decays'])

    model = NBEATS(n_time_in=int(mc['n_time_in']),
                   n_time_out=int(mc['n_time_out']),
                   n_x=mc['n_x'],
                   n_s=mc['n_s'],
                   n_s_hidden=int(mc['n_s_hidden']),
                   n_x_hidden=int(mc['n_x_hidden']),
                   shared_weights=mc['shared_weights'],
                   initialization=mc['initialization'],
                   activation=mc['activation'],
                   stack_types=mc['stack_types'],
                   n_blocks=mc['n_blocks'],
                   n_layers=mc['n_layers'],
                   n_theta_hidden=mc['n_theta_hidden'],
                   n_harmonics=int(mc['n_harmonics']),
                   n_polynomials=int(mc['n_polynomials']),
                   batch_normalization = mc['batch_normalization'],
                   dropout_prob_theta=mc['dropout_prob_theta'],
                   learning_rate=float(mc['learning_rate']),
                   lr_decay=float(mc['lr_decay']),
                   lr_decay_step_size=lr_decay_step_size,
                   weight_decay=mc['weight_decay'],
                   loss_train=mc['loss_train'],
                   loss_hypar=float(mc['loss_hypar']),
                   loss_valid=mc['loss_valid'],
                   frequency=mc['frequency'],
                   seasonality=int(mc['seasonality']),
                   random_seed=int(mc['random_seed']))
    return model

# Cell
def instantiate_esrnn(mc):
    if mc['max_epochs'] is not None:
        lr_decay_step_size = int(mc['max_epochs'] / mc['n_lr_decays'])
    elif mc['max_steps'] is not None:
        lr_decay_step_size = int(mc['max_steps'] / mc['n_lr_decays'])
    model = ESRNN(# Architecture parameters
                  n_series=mc['n_series'],
                  n_x=mc['n_x'],
                  n_s=mc['n_s'],
                  sample_freq=int(mc['sample_freq']),
                  input_size=int(mc['n_time_in']),
                  output_size=int(mc['n_time_out']),
                  es_component=mc['es_component'],
                  cell_type=mc['cell_type'],
                  state_hsize=int(mc['state_hsize']),
                  dilations=mc['dilations'],
                  add_nl_layer=mc['add_nl_layer'],
                  # Optimization parameters
                  learning_rate=mc['learning_rate'],
                  lr_scheduler_step_size=lr_decay_step_size,
                  lr_decay=mc['lr_decay'],
                  per_series_lr_multip=mc['per_series_lr_multip'],
                  gradient_eps=mc['gradient_eps'],
                  gradient_clipping_threshold=mc['gradient_clipping_threshold'],
                  rnn_weight_decay=mc['rnn_weight_decay'],
                  noise_std=mc['noise_std'],
                  level_variability_penalty=mc['level_variability_penalty'],
                  testing_percentile=mc['testing_percentile'],
                  training_percentile=mc['training_percentile'],
                  loss=mc['loss_train'],
                  val_loss=mc['loss_valid'],
                  seasonality=mc['seasonality'])
    return model

# Cell
def instantiate_nhits(mc):
    mc['n_theta_hidden'] = len(mc['stack_types']) * [ mc['n_layers'][0] * [int(mc['n_hidden'])] ]

    if mc['max_epochs'] is not None:
        lr_decay_step_size = int(mc['max_epochs'] / mc['n_lr_decays'])
    elif mc['max_steps'] is not None:
        lr_decay_step_size = int(mc['max_steps'] / mc['n_lr_decays'])

    model = NHITS(n_time_in=int(mc['n_time_in']),
                      n_time_out=int(mc['n_time_out']),
                      n_y=mc['n_y'],
                      n_x=mc['n_x'],
                      n_s=mc['n_s'],
                      n_s_hidden=int(mc['n_s_hidden']),
                      n_x_hidden=int(mc['n_x_hidden']),
                      shared_weights = mc['shared_weights'],
                      initialization=mc['initialization'],
                      activation=mc['activation'],
                      stack_types=mc['stack_types'],
                      n_blocks=mc['n_blocks'],
                      n_layers=mc['n_layers'],
                      n_theta_hidden=mc['n_theta_hidden'],
                      n_pool_kernel_size=mc['n_pool_kernel_size'],
                      n_freq_downsample=mc['n_freq_downsample'],
                      pooling_mode=mc['pooling_mode'],
                      interpolation_mode=mc['interpolation_mode'],
                      batch_normalization = mc['batch_normalization'],
                      dropout_prob_theta=mc['dropout_prob_theta'],
                      learning_rate=float(mc['learning_rate']),
                      lr_decay=float(mc['lr_decay']),
                      lr_decay_step_size=lr_decay_step_size,
                      weight_decay=mc['weight_decay'],
                      loss_train=mc['loss_train'],
                      loss_hypar=float(mc['loss_hypar']),
                      loss_valid=mc['loss_valid'],
                      frequency=mc['frequency'],
                      seasonality=int(mc['seasonality']),
                      random_seed=int(mc['random_seed']))
    return model

# Cell
def instantiate_autoformer(mc):

    if mc['max_epochs'] is not None:
        lr_decay_step_size = int(mc['max_epochs'] / mc['n_lr_decays'])
    elif mc['max_steps'] is not None:
        lr_decay_step_size = int(mc['max_steps'] / mc['n_lr_decays'])

    model = Autoformer(seq_len=int(mc['seq_len']),
                       label_len=int(mc['label_len']),
                       pred_len=int(mc['pred_len']),
                       output_attention=mc['output_attention'],
                       enc_in=int(mc['enc_in']),
                       dec_in=int(mc['dec_in']),
                       d_model=int(mc['d_model']),
                       c_out=int(mc['c_out']),
                       embed = mc['embed'],
                       freq=mc['freq'],
                       dropout=mc['dropout'],
                       factor=mc['factor'],
                       n_heads=int(mc['n_heads']),
                       d_ff=int(mc['d_ff']),
                       moving_avg=int(mc['moving_avg']),
                       activation=mc['activation'],
                       e_layers=int(mc['e_layers']),
                       d_layers=int(mc['d_layers']),
                       learning_rate=float(mc['learning_rate']),
                       lr_decay=float(mc['lr_decay']),
                       lr_decay_step_size=lr_decay_step_size,
                       weight_decay=mc['weight_decay'],
                       loss_train=mc['loss_train'],
                       loss_hypar=float(mc['loss_hypar']),
                       loss_valid=mc['loss_valid'],
                       random_seed=int(mc['random_seed']))

    return model

# Cell
def instantiate_model(mc):
    MODEL_DICT = {'nbeats': instantiate_nbeats,
                  'esrnn': instantiate_esrnn,
                  'nhits': instantiate_nhits,
                  'autoformer': instantiate_autoformer}
    return MODEL_DICT[mc['model']](mc)

# Cell
def predict(mc, model, trainer, loader, scaler_y):
   outputs = trainer.predict(model, loader)
   y_true, y_hat, mask = [t.cat(output).cpu().numpy() for output in zip(*outputs)]
   meta_data = loader.dataset.meta_data

   # Scale to original scale
   if mc['normalizer_y'] is not None:
        y_true_shape = y_true.shape
        y_true = scaler_y.inv_scale(x=y_true.flatten())
        y_true = np.reshape(y_true, y_true_shape)

        y_hat = scaler_y.inv_scale(x=y_hat.flatten())
        y_hat = np.reshape(y_hat, y_true_shape)

   return y_true, y_hat, mask, meta_data

# Cell
def model_fit_predict(mc, S_df, Y_df, X_df, f_cols, evaluate_train, ds_in_val, ds_in_test):

    # Protect inplace modifications
    Y_df = Y_df.copy()
    if X_df is not None:
        X_df = X_df.copy()
    if S_df is not None:
        S_df = S_df.copy()

    #----------------------------------------------- Datasets -----------------------------------------------#
    train_dataset, val_dataset, test_dataset, scaler_y = create_datasets(mc=mc,
                                                                         S_df=S_df, Y_df=Y_df, X_df=X_df,
                                                                         f_cols=f_cols,
                                                                         ds_in_val=ds_in_val,
                                                                         ds_in_test=ds_in_test)
    mc['n_y'], mc['n_x'], mc['n_s'] = train_dataset.get_n_variables()

    #------------------------------------------- Instantiate & fit -------------------------------------------#
    train_loader, val_loader, test_loader = instantiate_loaders(mc=mc,
                                                                train_dataset=train_dataset,
                                                                val_dataset=val_dataset,
                                                                test_dataset=test_dataset)
    model = instantiate_model(mc=mc)
    callbacks = []
    if mc['early_stop_patience']:
        early_stopping = pl.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4,
                                                    patience=mc['early_stop_patience'],
                                                    verbose=True,
                                                    mode='min')
        callbacks=[early_stopping]

    gpus = -1 if t.cuda.is_available() else 0
    trainer = pl.Trainer(max_epochs=mc['max_epochs'],
                         max_steps=mc['max_steps'],
                         check_val_every_n_epoch=mc['eval_freq'],
                         progress_bar_refresh_rate=1,
                         gpus=gpus,
                         callbacks=callbacks,
                         checkpoint_callback=False,
                         logger=False)
    trainer.fit(model, train_loader, val_loader)

    #------------------------------------------------ Predict ------------------------------------------------#
    results = {}

    if evaluate_train:
        y_true, y_hat, mask, meta_data = predict(mc, model, trainer, train_loader, scaler_y)
        train_values = (('train_y_true', y_true), ('train_y_hat', y_hat), ('train_mask', mask), ('train_meta_data', meta_data))
        results.update(train_values)

        print(f"TRAIN y_true.shape: {y_true.shape}")
        print(f"TRAIN y_hat.shape: {y_hat.shape}")
        print("\n")

    if ds_in_val > 0:
        y_true, y_hat, mask, meta_data = predict(mc, model, trainer, val_loader, scaler_y)
        val_values = (('val_y_true', y_true), ('val_y_hat', y_hat), ('val_mask', mask), ('val_meta_data', meta_data))
        results.update(val_values)

        print(f"VAL y_true.shape: {y_true.shape}")
        print(f"VAL y_hat.shape: {y_hat.shape}")
        print("\n")

    # Predict test if available
    if ds_in_test > 0:
        y_true, y_hat, mask, meta_data = predict(mc, model, trainer, test_loader, scaler_y)
        test_values = (('test_y_true', y_true), ('test_y_hat', y_hat), ('test_mask', mask), ('test_meta_data', meta_data))
        results.update(test_values)

        print(f"TEST y_true.shape: {y_true.shape}")
        print(f"TEST y_hat.shape: {y_hat.shape}")
        print("\n")
    
    assert 1<0, 'STOP'

    return results

# Cell
def evaluate_model(mc, loss_function_val, loss_functions_test,
                   S_df, Y_df, X_df, f_cols,
                   evaluate_train,
                   ds_in_val, ds_in_test,
                   return_forecasts,
                   save_progress,
                   trials,
                   results_file,
                   loss_kwargs):

    if (save_progress) and (len(trials) % 5 == 0):
        with open(results_file, "wb") as f:
            pickle.dump(trials, f)

    print(47*'=' + '\n')
    print(pd.Series(mc))
    print(47*'=' + '\n')

    # Some asserts due to work in progress
    n_series = Y_df['unique_id'].nunique()
    if n_series > 1:
        assert mc['normalizer_y'] is None, 'Data scaling not implemented with multiple time series'
        assert mc['normalizer_x'] is None, 'Data scaling not implemented with multiple time series'

    assert ds_in_test % mc['val_idx_to_sample_freq']==0, 'outsample size should be multiple of val_idx_to_sample_freq'

    # Make predictions
    start = time.time()
    results = model_fit_predict(mc=mc,
                                S_df=S_df,
                                Y_df=Y_df,
                                X_df=X_df,
                                f_cols=f_cols,
                                evaluate_train=evaluate_train,
                                ds_in_val=ds_in_val,
                                ds_in_test=ds_in_test)
    run_time = time.time() - start

    # Evaluate predictions
    val_loss = loss_function_val(y=results['val_y_true'], y_hat=results['val_y_hat'], weights=results['val_mask'], **loss_kwargs)

    results_output = {'loss': val_loss,
                      'mc': mc,
                      'run_time': run_time,
                      'status': STATUS_OK}

    # Evaluation in test (if provided)
    if evaluate_train > 0:
        train_loss_dict = {}
        for loss_name, loss_function in loss_functions_test.items():
            train_loss_dict[loss_name] = loss_function(y=results['train_y_true'], y_hat=results['train_y_hat'], weights=results['train_mask'])
        results_output['train_losses'] = train_loss_dict

    # Evaluation in test (if provided)
    if ds_in_test > 0:
        test_loss_dict = {}
        for loss_name, loss_function in loss_functions_test.items():
            test_loss_dict[loss_name] = loss_function(y=results['test_y_true'], y_hat=results['test_y_hat'], weights=results['test_mask'])
        results_output['test_losses'] = test_loss_dict

    if return_forecasts:
        forecasts_test = {}
        test_values = (('test_y_true', results['test_y_true']), ('test_y_hat', results['test_y_hat']),
                        ('test_mask', results['test_mask']), ('test_meta_data', results['test_meta_data']))
        forecasts_test.update(test_values)
        results_output['forecasts_test'] = forecasts_test

    return results_output

# Cell
def hyperopt_tunning(space, hyperopt_max_evals, loss_function_val, loss_functions_test,
                     S_df, Y_df, X_df, f_cols,
                     evaluate_train,
                     ds_in_val, ds_in_test,
                     return_forecasts,
                     save_progress,
                     results_file,
                     loss_kwargs=None):
    assert ds_in_val > 0, 'Validation set is needed for tunning!'

    trials = Trials()
    fmin_objective = partial(evaluate_model, loss_function_val=loss_function_val, loss_functions_test=loss_functions_test,
                             S_df=S_df, Y_df=Y_df, X_df=X_df, f_cols=f_cols,
                             evaluate_train=evaluate_train, ds_in_val=ds_in_val, ds_in_test=ds_in_test,
                             return_forecasts=return_forecasts, save_progress=save_progress, trials=trials,
                             results_file=results_file,
                             loss_kwargs=loss_kwargs or {})

    fmin(fmin_objective, space=space, algo=tpe.suggest, max_evals=hyperopt_max_evals, trials=trials, verbose=True)

    return trials