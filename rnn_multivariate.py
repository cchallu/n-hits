import os
import pickle
import time
import argparse
import pandas as pd

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from src.losses.numpy import mae, mse
from neuralforecast.experiments.utils import hyperopt_tunning

def get_experiment_space(args):
    space= {# Architecture parameters
            'model':'rnn',
            'mode': 'full',
            'n_time_in': hp.choice('n_time_in', [1*horizon]),
            'n_time_out': hp.choice('n_time_out', [horizon]),
            'cell_type': hp.choice('cell_type', ['LSTM']),
            'state_hsize': hp.choice('state_hsize', [10, 20, 50]),
            'dilations': hp.choice('dilations', [ [[1, 2]], [[1, 2, 4, 8]], [[1,2],[4,8]] ]),
            'add_nl_layer': hp.choice('add_nl_layer', [ False ]),
            'n_pool_kernel_size': hp.choice('n_pool_kernel_size', [ args.pooling ]),
            'n_freq_downsample': hp.choice('n_freq_downsample', [ args.interpolation ]),
            'sample_freq': hp.choice('sample_freq', [1]),
            # Regularization and optimization parameters
            'learning_rate': hp.choice('learning_rate', [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1]),
            'lr_decay': hp.choice('lr_decay', [0.5] ),
            'n_lr_decays': hp.choice('n_lr_decays', [3]), 
            'gradient_eps': hp.choice('gradient_eps', [1e-8]),
            'gradient_clipping_threshold': hp.choice('gradient_clipping_threshold', [10]),
            'weight_decay': hp.choice('weight_decay', [0]),
            'noise_std': hp.choice('noise_std', [0.001]),
            'max_epochs': hp.choice('max_epochs', [None]),
            'max_steps': hp.choice('max_steps', [500]),
            'early_stop_patience': hp.choice('early_stop_patience', [10]),
            'eval_freq': hp.choice('eval_freq', [50]),
            'loss_train': hp.choice('loss', ['MAE']),
            'loss_hypar': hp.choice('loss_hypar', [0.5]),
            'loss_valid': hp.choice('loss_valid', ['MAE']),
            # Data parameters
            'normalizer_y': hp.choice('normalizer_y', [None]),
            'normalizer_x': hp.choice('normalizer_x', [None]),
            'complete_windows':  hp.choice('complete_windows', [True]),
            'idx_to_sample_freq': hp.choice('idx_to_sample_freq', [1]),
            'val_idx_to_sample_freq': hp.choice('val_idx_to_sample_freq', [1]),
            'batch_size': hp.choice('batch_size', [8, 16, 32]),
            'n_windows': hp.choice('n_windows', [None]),
            'frequency': hp.choice('frequency', ['D']),
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
        window_sampling_limit = 4000+len_val+len_test
    if args.dataset == 'Exchange':
        len_val = 760
        len_test = 1517
        window_sampling_limit = 1517+len_val+len_test
    if args.dataset == 'ECL':
        len_val = 2632
        len_test = 5260
        window_sampling_limit = 4000+len_val+len_test
    if args.dataset == 'traffic':
        len_val = 1756
        len_test = 3508
        window_sampling_limit = 3508+len_val+len_test
    if args.dataset == 'weather':
        len_val = 5270
        len_test = 10539
        window_sampling_limit = 4000+len_val+len_test
    if args.dataset == 'ili':
        len_val = 97
        len_test = 193
        window_sampling_limit = 193+len_val+len_test

    Y_df = Y_df.groupby('unique_id').tail(window_sampling_limit).reset_index(drop=True)

    space = get_experiment_space(args)

    #---------------------------------------------- Directories ----------------------------------------------#
    output_dir = f'./results/multivariate/{args.dataset}_{args.horizon}/RNN_{args.pooling}_{args.interpolation}/{args.experiment_id}'

    os.makedirs(output_dir, exist_ok = True)
    assert os.path.exists(output_dir), f'Output dir {output_dir} does not exist'

    #----------------------------------------------- Hyperopt -----------------------------------------------#
    hyperopt_tunning(space=space, hyperopt_max_evals=args.hyperopt_max_evals, loss_function_val=mae,
                                loss_functions_test={'mae':mae, 'mse': mse},
                                Y_df=Y_df, X_df=X_df, S_df=S_df, f_cols=[],
                                ds_in_val=len_val, ds_in_test=len_test,
                                return_forecasts=False,
                                return_model=False,
                                save_trials=True,
                                results_dir=output_dir,
                                verbose=True)


def parse_args():
    desc = "Example of hyperparameter tuning"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--hyperopt_max_evals', type=int, help='hyperopt_max_evals')
    parser.add_argument('--pooling', type=int, help='pooling')
    parser.add_argument('--interpolation', type=int, help='interpolation')
    parser.add_argument('--experiment_id', default=None, required=False, type=str, help='string to identify experiment')
    return parser.parse_args()

if __name__ == '__main__':

    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    horizons = [96, 192, 336, 720]
    ILI_horizons = [24, 36, 48, 60]
    datasets = ['ETTm2', 'Exchange', 'ili', 'ECL', 'traffic', 'weather']

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
# CUDA_VISIBLE_DEVICES=2 python rnn_multivariate.py --pooling 2 --interpolation 2 --hyperopt_max_evals 5 --experiment_id "2022_05_15"
