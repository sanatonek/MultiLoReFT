import sys
import os as _os
_script_dir = _os.path.dirname(_os.path.abspath(__file__))
_project_root = _os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
from torch.utils.data import DataLoader
import os
import random
import numpy as np
import argparse
import wandb

from src.multimodal_projector import MultiLoReFT
from src.data.base import MultimodalDataset


def main(dataset_name, seed_id, lr, bs, rank, prune_th):
    """Main function to run the training pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Generate and load data
    if dataset_name == "simulated_apollo":
        loaded_data = np.load("./data/simulated_data_apollo.npz")
        input_dims = [80,40]
        shared_rank, specific_rank = rank, rank#40, 40
        early_stopping_config = {
            "shared": {
                "patience": 20,
                "min_improvement_ratio": 0.001,
                "max_epochs": 200
            },
            "private": {
                "patience": 20,
                "min_improvement_ratio": 0.001,
                "max_epochs": 200
            },
            "joint": {
                "patience": 150,
                "min_improvement_ratio": 0.001,
                "max_epochs": 5000
            }
        }
    elif dataset_name == "simulated":
        loaded_data = np.load("./data/simplest_sim_nongaussian.npz")
        input_dims = [10,10]
        shared_rank, specific_rank = rank, rank#10, 10
        early_stopping_config = {
                "shared": {
                    "patience": 20,
                    "min_improvement_ratio": 0.001,
                    "max_epochs": 200
                },
                "private": {
                    "patience": 20,
                    "min_improvement_ratio": 0.001,
                    "max_epochs": 200
                },
                "joint": {
                    "patience": 150,
                    "min_improvement_ratio": 0.001,
                    "max_epochs": 10000
                }
            }
    h1, h2, x1, x2, labels = loaded_data["h1"], loaded_data["h2"], loaded_data["x1"], loaded_data["x2"], loaded_data["labels"]
    # Create datasets
    n_train = int(0.8*len(h1))
    n_val = int(0.1*len(h1))
    n_test = len(h1) - n_train - n_val
    dataset = MultimodalDataset(h1[:n_train], h2[:n_train], x1[:n_train], x2[:n_train], labels[:n_train])
    val_dataset = MultimodalDataset(h1[n_train:n_train+n_val], h2[n_train:n_train+n_val], x1[n_train:n_train+n_val], x2[n_train:n_train+n_val], labels[n_train:n_train+n_val])

    # Create dataloaders
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=bs, shuffle=True)
    
        # Early stopping configuration
    

    # Initialize model
    projection_model = MultiLoReFT(
        dataset_name=dataset_name,
        input_dims=input_dims, 
        shared_rank=shared_rank, 
        specific_rank=specific_rank, 
        pruning_threshold=prune_th,
        staging=False,
        pruning=True,
        device=device,
        shared_R_mode="pad"
    ).to(device)
    

    # Train model
    projection_model.train_projection(dataloader, val_dataloader, early_stopping_config, lr=lr, epochs=5000, exp_name='multi_loreft_lr%.4f_bs%d_rank%d_prune%.2f_%d_no_stage'%(lr, bs, rank, prune_th, seed_id), dataset_name=dataset_name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run multimodal projection training.')
    parser.add_argument('--dataset', type=str, choices=['simulated', 'simulated_apollo'], default='simulated',
                        help='Type of dataset to use: either "simulated" or "simulated_apollo".')
    args = parser.parse_args()
    dataset_name = args.dataset
    for lr in [1e-3]:
        for bs in [256]:
            for rank in [40 if dataset_name == "simulated_apollo" else 10]:#[40]
                for prune_th in [0.1]:
                    run_name = '%s_multi_loreft_lr%.4f_bs%d_rank%d_prune%.2f'%(dataset_name, lr, bs, rank, prune_th)
                    wandb.init(project="MultiLoReFT", name=run_name, config={"dataset": dataset_name, "lr": lr, "bs": bs, "rank": rank, "prune_th": prune_th})
                    for seed_id in range(3):
                        main(dataset_name, seed_id, lr, bs, rank, prune_th)
                        wandb.log({"seed": seed_id})
                    wandb.finish()
