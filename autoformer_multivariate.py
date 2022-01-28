import os
import pickle
import glob
import time
import numpy as np
import pandas as pd
import argparse
import platform

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from src.losses.numpy import mae, mse
from src.experiments.utils import hyperopt_tunning

def get_experiment_space(args):
    if args.model == 'autoformer':
        space= {# Architecture parameters
                'model':'autoformer',
                'mode': 'iterate_windows',
                'seq_len': hp.choice('seq_len', [96]),
                'label_len': hp.choice('label_len', [48]),
                'pred_len': hp.choice('pred_len', [args.horizon]),
                'output_attention': hp.choice('output_attention', [False]),
                'enc_in': hp.choice('enc_in', [7]),
                'dec_in': hp.choice('dec_in', [7]),
                'c_out': hp.choice('c_out', [7]),
                'e_layers': hp.choice('e_layers', [2]),
                'd_layers': hp.choice('d_layers', [1]),
                'd_model': hp.choice('d_model', [512]),
                'embed': hp.choice('embed', ['timeF']),
                'freq': hp.choice('freq', ['h']),
                'dropout': hp.choice('dropout', [0.05]),
                'factor': hp.choice('factor', [1]),
                'n_heads': hp.choice('n_heads', [8]),
                'd_ff': hp.choice('d_ff', [2_048]),
                'moving_avg': hp.choice('moving_avg', [25]),
                'activation': hp.choice('activation', ['gelu']),
                # Regularization and optimization parameters
                'learning_rate': hp.choice('learning_rate', [0.001]),
                'lr_decay': hp.choice('lr_decay', [0.5]),
                'n_lr_decays': hp.choice('n_lr_decays', [5]), 
                'weight_decay': hp.choice('weight_decay', [0]), 
                'max_epochs': hp.choice('max_epochs', [10]),
                'max_steps': hp.choice('max_steps', [None]),
                'early_stop_patience': hp.choice('early_stop_patience', [3]),
                'eval_freq': hp.choice('eval_freq', [1]),
                'loss_train': hp.choice('loss', ['MSE']),
                'loss_hypar': hp.choice('loss_hypar', [0.5]),                
                'loss_valid': hp.choice('loss_valid', ['MSE']),
                # Data parameters
                'n_time_in': hp.choice('n_time_in', [96]),
                'n_time_out': hp.choice('n_time_out', [args.horizon]),
                'normalizer_y': hp.choice('normalizer_y', [None]),
                'normalizer_x': hp.choice('normalizer_x', [None]),
                'val_idx_to_sample_freq': hp.choice('val_idx_to_sample_freq', [1]),
                'batch_size': hp.choice('batch_size', [32]),
                'random_seed': hp.choice('random_seed', [1])}
    else:
        assert 1<0, 'Model not defined'
    return space

def main(args):

    #----------------------------------------------- Load Data -----------------------------------------------#
    Y_df = pd.read_csv(f'./data/{args.dataset}/M/df_y.csv')
    X_df = pd.read_csv(f'./data/{args.dataset}/M/df_x.csv')
    print(Y_df)
    print(X_df)

    #raise Exception

    X_df = X_df.drop_duplicates(subset=['ds'])

    X_df = Y_df[['unique_id', 'ds']].merge(X_df, how='left', on=['ds'])
    
    S_df = None

    print('Y_df: ', Y_df.head())
    if args.dataset == 'ETTm2':
        len_val = 11520
        len_test = 11520
    if args.dataset == 'Exchange':
        len_val = 760
        len_test = 1517
    if args.dataset == 'Electricity':
        len_val = 2632
        len_test = 5260
    if args.dataset == 'Traffic':
        len_val = 1756
        len_test = 3508
    if args.dataset == 'Weather':
        len_val = 5270
        len_test = 10539
    if args.dataset == 'ILI':
        len_val = 97
        len_test = 193

    space = get_experiment_space(args)

    output_dir = f'./results/multivariate/{args.dataset}_{args.horizon}/{args.model}/'

    os.makedirs(output_dir, exist_ok = True)
    assert os.path.exists(output_dir), f'Output dir {output_dir} does not exist'

    hyperopt_file = output_dir + f'hyperopt_{args.experiment_id}.p'

    if not os.path.isfile(hyperopt_file):
        print('Hyperparameter optimization')
        #----------------------------------------------- Hyperopt -----------------------------------------------#
        trials = hyperopt_tunning(space=space, hyperopt_max_evals=args.hyperopt_max_evals, loss_function_val=mae,
                                  loss_functions_test={'mae':mae, 'mse': mse},
                                  Y_df=Y_df, X_df=X_df, S_df=S_df, f_cols=[],
                                  ds_in_val=len_val, ds_in_test=len_test,
                                  return_forecasts=False,
                                  results_file = hyperopt_file,
                                  save_progress=True,
                                  loss_kwargs={})

        with open(hyperopt_file, "wb") as f:
            pickle.dump(trials, f)
    else:
        print('Hyperparameter optimization already done!')

def parse_args():
    desc = "Example of hyperparameter tuning"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--horizon', type=int, required=True, help='Horizon')
    parser.add_argument('--model', type=str, help='Model name', default='autoformer')
    parser.add_argument('--hyperopt_max_evals', type=int, help='hyperopt_max_evals', default=1)
    parser.add_argument('--experiment_id', default=None, required=False, type=str, help='string to identify experiment')
    return parser.parse_args()

if __name__ == '__main__':

    # parse arguments
    args = parse_args()
    if args is None:
        exit()
    
    main(args)

