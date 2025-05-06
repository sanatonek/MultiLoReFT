# create different datasets and run baseline analysis on all

import os
import sys
import torch
import torch.nn.functional as F
import argparse
import random
import numpy as np
import pandas as pd
# add parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.sim_data import generate_multimodal_data
from src.eval_metrics import evaluate_predictability, evaluate_regression, evaluate_classification

parser = argparse.ArgumentParser(description='Run baseline analysis on simulated data')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
args = parser.parse_args()

device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
# Set random seed
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

def eval_baseline(h1, h2, x1, x2, labels):
    
    # Get representations
    h1 = F.normalize(torch.Tensor(h1).float(), dim=1).to(device)
    h2 = F.normalize(torch.Tensor(h2).float(), dim=1).to(device)
    z_n = [[torch.Tensor(x1[:,:-2]).to(device), torch.Tensor(x1[:,-2:]).to(device)], [torch.Tensor(x2[:,:-2]).to(device), torch.Tensor(x2[:,-2:]).to(device)]]
    
    out_df = []
    # Evaluate predictability for each label
    for label_idx in range(labels.shape[1]):
        temp_dict = evaluate_predictability(z_n, labels, label_idx)
        temp_df = pd.DataFrame(temp_dict)
        temp_df['eval'] = 'v1'
        temp_df['label'] = label_idx
        out_df.append(temp_df)
    # I would not use the sum as the labels for regression but the use (z_s, z_m) with a linear layer to predict h1
    temp_dict = evaluate_regression(z_n[0], h1, i=0)
    temp_df = pd.DataFrame(temp_dict)
    temp_df['eval'] = 'v2'
    temp_df['label'] = 'regression'
    out_df.append(temp_df)
    temp_dict = evaluate_regression(z_n[1], h2, i=1)
    temp_df = pd.DataFrame(temp_dict)
    temp_df['eval'] = 'v2'
    temp_df['label'] = 'regression'
    out_df.append(temp_df)
    # also not entirely sure if the classification worked, since zs gets higher scores than zm
    temp_dict = evaluate_classification(z_n, labels[:, 0])
    temp_df = pd.DataFrame(temp_dict)
    temp_df['eval'] = 'v2'
    temp_df['label'] = 'classification'
    out_df.append(temp_df)

    out_df = pd.concat(out_df, axis=0)

    return out_df

n_sample_options = [100000] # I can subset for training
input_dims_0 = [5, 10]
input_dims_1 = [5, 10]
data_dims_0 = [5, 10, 50]
data_dims_1 = [5, 10, 50]
shared_dim_options = [2, 4, 8]
n_classes = 2
class_location_options = ['shared', 'specific']

for n_samples in n_sample_options:
    for input_dim0 in input_dims_0:
        for input_dim1 in input_dims_1:
            for data_dim0 in data_dims_0:
                if data_dim0 < input_dim0:
                    continue
                for data_dim1 in data_dims_1:
                    if data_dim1 < input_dim1:
                        continue
                    for shared_dim in shared_dim_options:
                        if shared_dim > data_dim0 or shared_dim > data_dim1:
                            continue
                        for class_location in class_location_options:
                            data_name = f"sim_{n_samples}_in{input_dim0}-{input_dim1}_data{data_dim0}-{data_dim1}_shared{shared_dim}_c{n_classes}_{class_location}.npz"
                            h1, h2, x1, x2, labels = generate_multimodal_data(n_samples, save_path='./data/', input_dims=[input_dim0, input_dim1], data_dims=[data_dim0, data_dim1], shared_dim=shared_dim, n_classes=n_classes, class_location=class_location)
                            temp_df = eval_baseline(h1, h2, x1, x2, labels)
                            temp_df['n_samples'] = n_samples
                            temp_df['input_dim0'] = input_dim0
                            temp_df['input_dim1'] = input_dim1
                            temp_df['data_dim0'] = data_dim0
                            temp_df['data_dim1'] = data_dim1
                            temp_df['shared_dim'] = shared_dim
                            temp_df['class_location'] = class_location

                            if os.path.exists('./results/sim_baselines.csv'):
                                temp_df.to_csv('./results/sim_baselines.csv', mode='a', header=False, index=False)
                            else:
                                temp_df.to_csv('./results/sim_baselines.csv', mode='w', header=True, index=False)