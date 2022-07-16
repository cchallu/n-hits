import os
import pickle
import time
import argparse
import pandas as pd

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from src.losses.numpy import mae, mse
from src.experiments.utils import hyperopt_tunning

def get_experiment_space(args):
    space= {# Architecture parameters
            'model':'nhits',
            'mode': 'simple',
            'n_time_in': hp.choice('n_time_in', [5*args.horizon]),
            'n_time_out': hp.choice('n_time_out', [args.horizon]),
            'n_x_hidden': hp.choice('n_x_hidden', [0]),
            'n_s_hidden': hp.choice('n_s_hidden', [0]),
            'shared_weights': hp.choice('shared_weights', [False]),
            'activation': hp.choice('activation', ['ReLU']),
            'initialization':  hp.choice('initialization', ['lecun_normal']),
            'stack_types': hp.choice('stack_types', [ 3*['identity'] ]),
            'n_blocks': hp.choice('n_blocks', [ 3*[1]]),
            'n_layers': hp.choice('n_layers', [ 9*[2] ]),
            'n_hidden': hp.choice('n_hidden', [ 512 ]),
            'naive_seasonality': hp.choice('naive_seasonality', [ 0, args.naive_seasonality ]),
            'n_pool_kernel_size': hp.choice('n_pool_kernel_size', [ 3*[1], 3*[2], 3*[4], 3*[8], [8, 4, 1], [16, 8, 1] ]),
            'n_freq_downsample': hp.choice('n_freq_downsample', [ [168, 24, 1], [24, 12, 1],
                                                                  [180, 60, 1], [60, 8, 1],
                                                                  [40, 20, 1]
                                                                ]),
            'pooling_mode': hp.choice('pooling_mode', [ 'max' ]),
            'interpolation_mode': hp.choice('interpolation_mode', ['linear']),
            # Regularization and optimization parameters
            'batch_normalization': hp.choice('batch_normalization', [False]),
            'dropout_prob_theta': hp.choice('dropout_prob_theta', [ 0 ]),
            'dropout_prob_exogenous': hp.choice('dropout_prob_exogenous', [0]),
            'learning_rate': hp.choice('learning_rate', [0.001]),
            'lr_decay': hp.choice('lr_decay', [0.5] ),
            'n_lr_decays': hp.choice('n_lr_decays', [3]), 
            'weight_decay': hp.choice('weight_decay', [0] ),
            'max_epochs': hp.choice('max_epochs', [None]),
            'max_steps': hp.choice('max_steps', [1_000]),
            'early_stop_patience': hp.choice('early_stop_patience', [10]),
            'eval_freq': hp.choice('eval_freq', [50]),
            'loss_train': hp.choice('loss', ['MAE']),
            'loss_hypar': hp.choice('loss_hypar', [0.5]),                
            'loss_valid': hp.choice('loss_valid', ['MAE']),
            'l1_theta': hp.choice('l1_theta', [0]),
            # Data parameters
            'normalizer_y': hp.choice('normalizer_y', [None]),
            'normalizer_x': hp.choice('normalizer_x', [None]),
            'complete_windows':  hp.choice('complete_windows', [True]),
            'frequency': hp.choice('frequency', ['H']),
            'seasonality': hp.choice('seasonality', [24]),      
            'idx_to_sample_freq': hp.choice('idx_to_sample_freq', [1]),
            'val_idx_to_sample_freq': hp.choice('val_idx_to_sample_freq', [1]),
            'batch_size': hp.choice('batch_size', [1]),
            'n_windows': hp.choice('n_windows', [256]),
            'random_seed': hp.quniform('random_seed', 1, 10, 1)}
    return space

def main(args):

    #----------------------------------------------- Load Data -----------------------------------------------#
    Y_df = pd.read_csv(f'./data/{args.dataset}/M/df_y.csv')

    X_df = None
    S_df = None

    print('Y_df: ', Y_df.head())
    if args.dataset == 'ETTm2':
        len_val = 11520
        len_test = 11520
        args.naive_seasonality = 96
    if args.dataset == 'Exchange':
        len_val = 760
        len_test = 1517
        args.naive_seasonality = 7
    if args.dataset == 'ECL':
        len_val = 2632
        len_test = 5260
        args.naive_seasonality = 96
    if args.dataset == 'traffic':
        len_val = 1756
        len_test = 3508
        args.naive_seasonality = 168
    if args.dataset == 'weather':
        len_val = 5270
        len_test = 10539
        args.naive_seasonality = 144
    if args.dataset == 'ili':
        len_val = 97
        len_test = 193
        args.naive_seasonality = 52

    space = get_experiment_space(args)

    #---------------------------------------------- Directories ----------------------------------------------#
    output_dir = f'./results/multivariate/{args.dataset}_{args.horizon}/NHITS/'

    os.makedirs(output_dir, exist_ok = True)
    assert os.path.exists(output_dir), f'Output dir {output_dir} does not exist'

    hyperopt_file = output_dir + f'hyperopt_{args.experiment_id}.p'

    if not os.path.isfile(hyperopt_file):
        print('Hyperparameter optimization')
        #----------------------------------------------- Hyperopt -----------------------------------------------#
        trials = hyperopt_tunning(space=space, hyperopt_max_evals=args.hyperopt_max_evals, loss_function_val=mae,
                                  loss_functions_test={'mae':mae, 'mse': mse},
                                  Y_df=Y_df, X_df=X_df, S_df=S_df, f_cols=[],
                                  evaluate_train=True,
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
    parser.add_argument('--hyperopt_max_evals', type=int, help='hyperopt_max_evals')
    parser.add_argument('--experiment_id', default=None, required=False, type=str, help='string to identify experiment')
    return parser.parse_args()

if __name__ == '__main__':

    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    horizons = [96, 192, 336, 720]
    ILI_horizons = [24, 36, 48, 60]
    datasets = ['ETTm2', 'Exchange', 'weather', 'ili', 'ECL', 'traffic']

    for dataset in datasets:
        # Horizon
        if dataset == 'ili':
            horizons_dataset = ILI_horizons
        else:
            horizons_dataset = horizons
        for horizon in horizons_dataset:
            print(50*'-', dataset, 50*'-')
            print(50*'-', horizon, 50*'-')
            start = time.time()
            args.dataset = dataset
            args.horizon = horizon
            main(args)
            print('Time: ', time.time() - start)

    main(args)

# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate nixtla
# CUDA_VISIBLE_DEVICES=0 python nhits_multivariate.py --hyperopt_max_evals 30 --experiment_id "2022_07_16"
