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
    loaded_data = np.load("./data/simulated_data.npz")
    h1, h2, x1, x2, labels = loaded_data["h1"], loaded_data["h2"], loaded_data["x1"], loaded_data["x2"], loaded_data["labels"]
    # Create datasets
    dataset = MultimodalDataset(h1[:4000], h2[:4000], x1[:4000], x2[:4000], labels[:4000])
    val_dataset = MultimodalDataset(h1[4000:5000], h2[4000:5000], x1[4000:5000], x2[4000:5000], labels[4000:5000])
    
    hyperparameters = {
        "batch_size": [128],
        "learning_rate": [1e-5, 1e-4, 1e-3],
        "lr_schedule": ['cosine', 'linear'],
        "weight_decay": [1e-4],
        "n_specific_rank": [4],
        "n_shared_rank": [4],
        "weight_init": ['uniform', 'kaiming_uniform', 'kaiming_normal'],
        "patience1": 10,
        "patience2": 10,
        "min_improvement_ratio1": 0.001,
        "min_improvement_ratio2": 0.001,
        "max_epochs": 1000,
        "staging": [True, False],
        "pruning": [True],
    }
    n_combinations = np.prod([len(v) for v in hyperparameters.values() if isinstance(v, list)])
    print(f"Total combinations: {n_combinations}")
    epochs = 3000

    run_iter = 0
    for bs in hyperparameters["batch_size"]:
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=bs, shuffle=True)
        for lr in hyperparameters["learning_rate"]:
            for wd in hyperparameters["weight_decay"]:
                for weight_init in hyperparameters["weight_init"]:
                    for lr_anneal in hyperparameters["lr_schedule"]:
                        for staging in hyperparameters["staging"]:
                            for pruning in hyperparameters["pruning"]:

                                # Print current hyperparameters
                                print(f"Running iteration {run_iter + 1}/{n_combinations}")

                                # Create early stopping config
                                early_stopping_config = {
                                    "shared": {
                                        "patience": hyperparameters["patience1"],
                                        "max_epochs": hyperparameters["max_epochs"],
                                        "min_improvement_ratio": hyperparameters["min_improvement_ratio1"]
                                    },
                                    "private": {
                                        "patience": hyperparameters["patience1"],
                                        "max_epochs": hyperparameters["max_epochs"],
                                        "min_improvement_ratio": hyperparameters["min_improvement_ratio1"]
                                    },
                                    "joint": {
                                        "patience": hyperparameters["patience2"],
                                        "max_epochs": hyperparameters["max_epochs"],
                                        "min_improvement_ratio": hyperparameters["min_improvement_ratio2"]
                                    }
                                }

                                model_hyperparameters = {
                                    "batch_size": bs,
                                    "learning_rate": lr,
                                    "weight_decay": wd,
                                    "weight_init": weight_init,
                                    "lr_schedule": lr_anneal,
                                    "n_specific_rank": hyperparameters["n_specific_rank"][0],
                                    "n_shared_rank": hyperparameters["n_shared_rank"][0],
                                    "patience1": hyperparameters["patience1"],
                                    "patience2": hyperparameters["patience2"],
                                    "min_improvement_ratio1": hyperparameters["min_improvement_ratio1"],
                                    "min_improvement_ratio2": hyperparameters["min_improvement_ratio2"],
                                    "max_epochs": hyperparameters["max_epochs"],
                                    "staging": staging,
                                    "pruning": pruning,
                                    "model_number": run_iter,
                                    "seed": seed,
                                }
                                
                                setup_wandb(
                                    run_name=f"{file_name}_{run_iter}",
                                    hyperparams=model_hyperparameters,
                                    project_name="multimodal_loreft",
                                    entity="vschuster-broad-institute"
                                )

                                # Initialize model
                                projection_model = MultiLoReFT(
                                    input_dims=[5,5], 
                                    shared_rank=4,
                                    specific_rank=4, 
                                    staging=staging,
                                    pruning=pruning,
                                    r_init=weight_init,
                                    verbose=False,
                                    wandb_log=True,
                                    device=device).to(device)
                                
                                # Train model
                                train_df = projection_model.train_projection(dataloader, val_dataloader, early_stopping_config, epochs=epochs,hyperparameters=model_hyperparameters)
                                train_df['run_iter'] = run_iter
                                train_df['seed'] = seed

                                n_train = 4000
                                n_val = 1000
                                regression_df, classification_df = eval_model(projection_model, h1[n_train:n_train+n_val], h2[n_train:n_train+n_val], labels[n_train:n_train+n_val], device)
                                regression_df['run_iter'] = run_iter
                                regression_df['seed'] = seed
                                classification_df['run_iter'] = run_iter
                                classification_df['seed'] = seed
                                eval_df = reeval_model(projection_model, h1[n_train:n_train+n_val], h2[n_train:n_train+n_val], labels[n_train:n_train+n_val], device)
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
    parser.add_argument("--file_name", type=str, default="sim_sweep",
                      help="Base name for output files (default: 'sweep')")
    args = parser.parse_args()
    main(dev_id=args.gpu, seed=args.seed, out_dir=args.out_dir, file_name=args.file_name)