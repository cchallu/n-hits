from pathlib import Path

import pickle
import argparse
import numpy as np

from src.losses.numpy import mae, mse


def get_score_min_val(dir):
    print(dir)
    result = pickle.load(open(dir, 'rb'))
    min_mae = 100
    mc = {}
    for i in range(len(result)):
        val_mae = result.trials[i]['result']['loss']
        if val_mae < min_mae:
            mae_best = result.trials[i]['result']['test_losses']['mae']
            mse_best = result.trials[i]['result']['test_losses']['mse']
            min_mae = val_mae
            mc = result.trials[i]['result']['mc']
    return mae_best, mse_best, mc

def main(args):

    if args.horizon<0:
        if args.dataset == 'ili':
            horizons = [24, 36, 48, 60]
        else:
            horizons = [96, 192, 336, 720]
    else:
        horizons = [args.horizon]

    for horizon in horizons:
        result_dir = f'./results/{args.setting}/{args.dataset}_{horizon}/{args.model}/'
        result_dir = Path(result_dir)
        files = list(result_dir.glob(f'hyperopt_{args.experiment}*.p'))
        maes = []
        mses = []
        for file_ in files:
            mae_data, mse_data = get_score_min_val(file_)
            maes.append(mae_data)
            mses.append(mse_data)

        print(f'Horizon {horizon}')
        print(f'MSE: {np.mean(mses)}')
        print(f'MAE: {np.mean(maes)}')

def parse_args():
    desc = "Example of hyperparameter tuning"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--dataset', type=str, help='Name of the dataset')
    parser.add_argument('--setting', type=str, help='Multivariate or univariate', default='multivariate')
    parser.add_argument('--horizon', type=int, help='Horizon')
    parser.add_argument('--model', type=str, help='Model name')
    parser.add_argument('--experiment', type=str, help='string to identify experiment')
    return parser.parse_args()

if __name__ == '__main__':

    # parse arguments
    args = parse_args()
    if args is None:
        exit()
    
    main(args)
