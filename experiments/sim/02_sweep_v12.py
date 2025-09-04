import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
import wandb
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tqdm import tqdm
from src.utils import *
from src.losses import *
#from transformers import BertTokenizer, AutoTokenizer
from src.visualization import *
#from src.utils import setup_wandb, log_wandb, custom_weight_init
from src.eval_metrics import calc_corrs_and_ranks, evaluate_validation_loss, eval_model, reeval_model
from src.model import MultiLoReFT
from src.data import MultimodalDataset

# ignore warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def main(dev_id=1, seed=0, out_dir="./results/", file_name="sweep_v2"):
    """Main function to run the training pipeline."""
    device = torch.device(f"cuda:{dev_id}" if torch.cuda.is_available() else "cpu")

    # set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    file_name = f"{file_name}_seed{seed}"
    
    # Generate and load data
    loaded_data = np.load("./data/simplest_sim_nongaussian.npz")
    h1, h2, x1, x2, labels = loaded_data["h1"], loaded_data["h2"], loaded_data["x1"], loaded_data["x2"], loaded_data["labels"]
    # Create datasets
    dataset = MultimodalDataset(h1[:4000], h2[:4000], x1[:4000], x2[:4000], labels[:4000])
    val_dataset = MultimodalDataset(h1[4000:5000], h2[4000:5000], x1[4000:5000], x2[4000:5000], labels[4000:5000])
    
    hyperparameters = {
        "batch_size": [256],
        "learning_rate": [1e-4],
        "lr_schedule": ['constant'],
        "weight_decay": [1e-3],
        "n_specific_rank": [10],
        "n_shared_rank": [10],
        "weight_init": ['kaiming_normal'],
        "weight_depth": [1],
        "r_uniform_gain": [0.1],
        "patience1": 10,
        "patience2": [10, 50, 100],
        "min_improvement_ratio1": [0.0001, 0.001, 0.01],
        "max_epochs": 1000,
        "staging": [True],
        "pruning": [True],
        "single_prune": [True, False],
        "pruning_threshold": [0.01, 0.05, 0.1, 0.2], # make the pruning condition dependent on this instead of fixed value
        #"early_stopping": [True, False],
        "early_stopping": [True],
        "warmup": [30], # add as variable
    }
    #epochs = 3000
    n_combinations = np.prod([len(v) for v in hyperparameters.values() if isinstance(v, list)])
    print(f"Total combinations: {n_combinations}")
    
    run_iter = 0
    for bs in hyperparameters["batch_size"]:
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=bs, shuffle=True)
        for lr in hyperparameters["learning_rate"]:
            for wd in hyperparameters["weight_decay"]:
                for lr_anneal in hyperparameters["lr_schedule"]:
                    for gain in hyperparameters["r_uniform_gain"]:
                        for p2 in hyperparameters["patience2"]:
                            #for p2 in hyperparameters["patience2"]:
                            for staging in hyperparameters["staging"]:
                                for single_prune in hyperparameters["single_prune"]:
                                    for n_specific_rank in hyperparameters["n_specific_rank"]:
                                        for n_shared_rank in hyperparameters["n_shared_rank"]:
                                            for w_init in hyperparameters["weight_init"]:
                                                for w_depth in hyperparameters["weight_depth"]:
                                                    for min_improvement_ratio1 in hyperparameters["min_improvement_ratio1"]:
                                                        #for min_improvement_ratio2 in hyperparameters["min_improvement_ratio2"]:
                                                        for pruning_threshold in hyperparameters["pruning_threshold"]:
                                                            for early_stopping in hyperparameters["early_stopping"]:
                                                                for warmup in hyperparameters["warmup"]:

                                                                    # Print current hyperparameters
                                                                    print(f"Running iteration {run_iter + 1}/{n_combinations}")

                                                                    # Create early stopping config
                                                                    early_stopping_config = {
                                                                        "shared": {
                                                                            "patience": hyperparameters["patience1"],
                                                                            "max_epochs": hyperparameters["max_epochs"],
                                                                            "min_improvement_ratio": min_improvement_ratio1
                                                                        },
                                                                        "private": {
                                                                            "patience": hyperparameters["patience1"],
                                                                            "max_epochs": hyperparameters["max_epochs"],
                                                                            "min_improvement_ratio": min_improvement_ratio1
                                                                        },
                                                                        "joint": {
                                                                            "patience": p2,
                                                                            "max_epochs": hyperparameters["max_epochs"],
                                                                            "min_improvement_ratio": min_improvement_ratio1
                                                                        }
                                                                    }

                                                                    model_hyperparameters = {
                                                                        "batch_size": bs,
                                                                        "learning_rate": lr,
                                                                        "weight_decay": wd,
                                                                        "weight_init": w_init,
                                                                        "r_gain": gain,
                                                                        "lr_schedule": lr_anneal,
                                                                        "n_specific_rank": n_specific_rank,
                                                                        "n_shared_rank": n_shared_rank,
                                                                        "patience1": hyperparameters["patience1"],
                                                                        "patience2": p2,
                                                                        "min_improvement_ratio1": min_improvement_ratio1,
                                                                        "min_improvement_ratio2": min_improvement_ratio1,
                                                                        "single_prune": single_prune,
                                                                        "max_epochs": hyperparameters["max_epochs"],
                                                                        "staging": staging,
                                                                        "pruning": hyperparameters["pruning"][0],
                                                                        "pruning_value_threshold": pruning_threshold,
                                                                        "model_number": run_iter,
                                                                        "seed": seed,
                                                                        "weight_depth": w_depth,
                                                                        "early_stopping": early_stopping,
                                                                        "warmup": warmup
                                                                    }

                                                                    setup_wandb(
                                                                        run_name=f"{file_name}_{run_iter}",
                                                                        hyperparams=model_hyperparameters,
                                                                        project_name="multimodal_loreft",
                                                                        entity="vschuster-broad-institute"
                                                                    )

                                                                    # Initialize model
                                                                    projection_model = MultiLoReFT(
                                                                        input_dims=[10,10], 
                                                                        shared_rank=n_shared_rank,
                                                                        specific_rank=n_specific_rank,
                                                                        staging=staging,
                                                                        pruning=model_hyperparameters["pruning"],
                                                                        pruning_value_threshold=model_hyperparameters["pruning_value_threshold"],
                                                                        w_init=model_hyperparameters["weight_init"],
                                                                        r_init_gain=gain,
                                                                        verbose=False,
                                                                        wandb_log=True,
                                                                        device=device).to(device)
                                                                    
                                                                    # Train model
                                                                    all_epoch_losses, all_loss_names, all_epoch_stages = projection_model.train_projection(dataloader, val_dataloader, early_stopping_config, epochs=hyperparameters["max_epochs"], hyperparameters=model_hyperparameters, warmup=warmup)

                                                                    # get eval metrics and plot
                                                                    h1 = F.normalize(torch.Tensor(val_dataloader.dataset.h1).float(), dim=1).to(device)
                                                                    h2 = F.normalize(torch.Tensor(val_dataloader.dataset.h2).float(), dim=1).to(device)
                                                                    labels = val_dataloader.dataset.labels
                                                                    phis = projection_model.forward([h1,h2])
                                                                    z_n = projection_model.decouple(phis, full=True, th=projection_model.pruning_threshold)
                                                                    try:
                                                                        plot_representations_wandb(z_n, labels)
                                                                        plot_projection_matrices_wandb(projection_model)
                                                                    except Exception as e:
                                                                        print(f"Error occurred while plotting: {e}")
                                                                    # return all training losses
                                                                    all_epoch_losses = np.array(all_epoch_losses)
                                                                    train_df = pd.DataFrame(all_epoch_losses, columns=all_loss_names)
                                                                    train_df['stage'] = all_epoch_stages
                                                                    train_df['epoch'] = np.arange(len(train_df))
                                                                    train_df['run_iter'] = run_iter
                                                                    train_df['seed'] = seed

                                                                    #"""
                                                                    n_train = 4000
                                                                    n_val = 1000
                                                                    regression_df, classification_df = eval_model(projection_model, h1, h2, labels.numpy(), device)
                                                                    regression_df['run_iter'] = run_iter
                                                                    regression_df['seed'] = seed
                                                                    classification_df['run_iter'] = run_iter
                                                                    classification_df['seed'] = seed
                                                                    eval_df = reeval_model(projection_model, h1, h2, labels, device)
                                                                    eval_df['run_iter'] = run_iter
                                                                    eval_df['seed'] = seed

                                                                    hyperparam_df = pd.DataFrame([model_hyperparameters])
                                                                    if os.path.exists(f"{out_dir}/{file_name}_train.csv"):
                                                                        train_df.to_csv(f"{out_dir}/{file_name}_train.csv", mode='a', header=False, index=False)
                                                                    else:
                                                                        train_df.to_csv(f"{out_dir}/{file_name}_train.csv", header=True, index=False)
                                                                    if os.path.exists(f"{out_dir}/{file_name}_hyperparams.csv"):
                                                                        hyperparam_df.to_csv(f"{out_dir}/{file_name}_hyperparams.csv", mode='a', header=False, index=False)
                                                                    else:
                                                                        hyperparam_df.to_csv(f"{out_dir}/{file_name}_hyperparams.csv", header=True, index=False)
                                                                    if os.path.exists(f"{out_dir}/{file_name}_analysis.csv"):
                                                                        eval_df.to_csv(f"{out_dir}/{file_name}_analysis.csv", mode='a', header=False, index=False)
                                                                    else:
                                                                        eval_df.to_csv(f"{out_dir}/{file_name}_analysis.csv", header=True, index=False)
                                                                    if os.path.exists(f"{out_dir}/{file_name}_regression.csv"):
                                                                        regression_df.to_csv(f"{out_dir}/{file_name}_regression.csv", mode='a', header=False, index=False)
                                                                    else:
                                                                        regression_df.to_csv(f"{out_dir}/{file_name}_regression.csv", header=True, index=False)
                                                                    if os.path.exists(f"{out_dir}/{file_name}_classification.csv"):
                                                                        classification_df.to_csv(f"{out_dir}/{file_name}_classification.csv", mode='a', header=False, index=False)
                                                                    else:
                                                                        classification_df.to_csv(f"{out_dir}/{file_name}_classification.csv", header=True, index=False)
                                                                    #"""
                                                                    # Finish wandb run
                                                                    wandb.finish()
                                                                    run_iter += 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train multimodal projection model")
    parser.add_argument("--gpu", type=int, default=1, 
                      help="GPU device ID to use (default: 1)")
    parser.add_argument("--seed", type=int, default=0,
                      help="Random seed for reproducibility (default: 0)")
    parser.add_argument("--out_dir", type=str, default="./results/",
                      help="Output directory for results (default: './results/')")
    #parser.add_argument("--file_name", type=str, default="sim_sweep",
    #                  help="Base name for output files (default: 'sweep')")
    args = parser.parse_args()
    #file_name = f"sweep_v9_methodParams_seed{args.seed}_noES"
    #file_name = f"sweep_v9_methodParams_seed{args.seed}"
    file_name = f"sweep_v12" # here I changed part of the training objective to use the forward pass again
    main(dev_id=args.gpu, seed=args.seed, out_dir=args.out_dir, file_name=file_name)