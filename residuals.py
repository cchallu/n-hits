import argparse
import os

import numpy as np
import pandas as pd

from evaluation import get_score_min_val
from src.experiments.utils import model_fit_predict

def main(args):

    #----------------------------------------------- Load Data -----------------------------------------------#
    Y_df = pd.read_csv(f'./data/{args.dataset}/M/df_y.csv')

    X_df = None
    S_df = None

    print('Y_df: ', Y_df.head())
    if args.dataset == 'ETTm2':
        len_val = 11520
        len_test = 11520
    if args.dataset == 'Exchange':
        len_val = 760
        len_test = 1517
    if args.dataset == 'ECL':
        len_val = 2632
        len_test = 5260
    if args.dataset == 'traffic':
        len_val = 1756
        len_test = 3508
    if args.dataset == 'weather':
        len_val = 5270
        len_test = 10539
    if args.dataset == 'ili':
        len_val = 97
        len_test = 193

    output_dir = f'./results/multivariate/{args.dataset}_{args.horizon}/NHITS/'

    os.makedirs(output_dir, exist_ok = True)
    assert os.path.exists(output_dir), f'Output dir {output_dir} does not exist'

    hyperopt_file = output_dir + f'hyperopt_{args.experiment_id}.p'
    *_, mc = get_score_min_val(hyperopt_file)
    results = model_fit_predict(mc=mc, S_df=S_df, 
				Y_df=Y_df, X_df=X_df, 
				f_cols=[], ds_in_val=len_val, 
				ds_in_test=len_test, 
				insample=True)

    n_series = Y_df['unique_id'].nunique()
    for data_kind in ['insample', 'val', 'test']:
        for y_kind in ['true', 'hat']:
            name = f'{data_kind}_y_{y_kind}'
            result_name = results[name].reshape((n_series, -1, mc['n_time_out']))
            np.save(output_dir + f'{name}.npy', result_name)

def parse_args():
    desc = "Example of hyperparameter tuning"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--experiment_id', default=None, required=False, type=str, help='string to identify experiment')
    return parser.parse_args()

if __name__ == '__main__':

    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    horizons = [96, 192, 336, 720]
    ILI_horizons = [24, 36, 48, 60]
    datasets = ['ETTm2', 'weather', 'Exchange']#['ECL', 'Exchange', 'traffic', 'weather', 'ili']

    for dataset in datasets:
        # Horizon
        if dataset == 'ili':
            horizons_dataset = ILI_horizons
        else:
            horizons_dataset = horizons
        for horizon in horizons_dataset:
            print(50*'-', dataset, 50*'-')
            print(50*'-', horizon, 50*'-')
            args.dataset = dataset
            args.horizon = horizon
            main(args)

    main(args)

# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate nixtla
# CUDA_VISIBLE_DEVICES=0 python nhits_multivariate.py --hyperopt_max_evals 10 --experiment_id "test"
