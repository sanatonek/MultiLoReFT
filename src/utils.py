import wandb
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score

def setup_wandb(run_name, hyperparams, project_name, entity):
    try:
        wandb.init(project=project_name, entity=entity, config=hyperparams)
        wandb.run.name = run_name
    except:
        raise ValueError("Could not initialize wandb. Please check your settings.")

def log_wandb(history, mode="train"):
    if mode == "eval":
        wandb.run.summary.update(history)
    else:
        wandb.log(history)

import torch
import torch.nn.init as init
def custom_weight_init(m, init_option='kaiming_normal'):
    if isinstance(m, torch.nn.Linear):
        if init_option == 'xavier_uniform':
            init.xavier_uniform_(m.weight)
        elif init_option == 'xavier_normal':
            init.xavier_normal_(m.weight)
        elif init_option == 'uniform':
            init.uniform_(m.weight, -0.9, 0.9)
        elif init_option == 'normal':
            init.normal_(m.weight, 0, 0.1)
        elif init_option == 'kaiming_uniform':
            init.kaiming_uniform_(m.weight)
        elif init_option == 'kaiming_normal':
            init.kaiming_normal_(m.weight)
        else:
            raise ValueError('Invalid weight initialization scheme')
        
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, torch.nn.Sequential):
        for sub_m in m:
            custom_weight_init(sub_m, init_option)
    elif isinstance(m, torch.nn.Parameter):
        if init_option == 'xavier_uniform':
            init.xavier_uniform_(m)
        elif init_option == 'xavier_normal':
            init.xavier_normal_(m)
        elif init_option == 'uniform':
            init.uniform_(m, -0.9, 0.9)
        elif init_option == 'normal':
            init.normal_(m, 0, 0.1)
        elif init_option == 'kaiming_uniform':
            init.kaiming_uniform_(m)
        elif init_option == 'kaiming_normal':
            init.kaiming_normal_(m)
        else:
            raise ValueError('Invalid weight initialization scheme')

def _patch_linear_layer(layer, new_out_features):
    in_features = layer.in_features
    new_layer = nn.Linear(in_features, new_out_features, dtype=layer.weight.dtype)
    return new_layer

def load_checkpoint(filepath, model, optimizer=None):
    checkpoint = torch.load(filepath)
    state_dict = checkpoint["model_state_dict"]

    # Patch R_s and matching W_s0, W_s1
    if "R_s" in state_dict:
        new_rank = state_dict["R_s"].shape[0]
        if model.R_s.shape[0] != new_rank:
            print(f"Patching R_s → {new_rank}")
            model.R_s = nn.Parameter(torch.empty_like(state_dict["R_s"]))
            model.register_parameter("R_s", model.R_s)
            model.W_s0[-1] = _patch_linear_layer(model.W_s0[-1], new_rank)
            model.W_s1[-1] = _patch_linear_layer(model.W_s1[-1], new_rank)

    # Patch R_m1 and W_m0
    if "R_m1" in state_dict:
        new_rank = state_dict["R_m1"].shape[0]
        if model.R_m1.shape[0] != new_rank:
            print(f"Patching R_m1 → {new_rank}")
            model.R_m1 = nn.Parameter(torch.empty_like(state_dict["R_m1"]))
            model.register_parameter("R_m1", model.R_m1)
            model.W_m0[-1] = _patch_linear_layer(model.W_m0[-1], new_rank)

    # Patch R_m2 and W_m1
    if "R_m2" in state_dict:
        new_rank = state_dict["R_m2"].shape[0]
        if model.R_m2.shape[0] != new_rank:
            print(f"Patching R_m2 → {new_rank}")
            model.R_m2 = nn.Parameter(torch.empty_like(state_dict["R_m2"]))
            model.register_parameter("R_m2", model.R_m2)
            model.W_m1[-1] = _patch_linear_layer(model.W_m1[-1], new_rank)

    # Now load the weights
    model.load_state_dict(state_dict, strict=False)

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Checkpoint loaded from {filepath} (Epoch {checkpoint['epoch']})")
    return checkpoint


class SklearnTrainer:
    def __init__(self, model, task_type="regression"):
        """
        Wrapper for training a Scikit-Learn model with k-fold cross-validation.

        Args:
            model: A Scikit-Learn model instance (e.g., LinearRegression, RandomForestClassifier).
            task_type (str): "regression" or "classification".
        """
        self.model = model
        self.task_type = task_type

    def train_and_evaluate(self, X, y, k=5):
        """
        Trains and evaluates the model using k-fold cross-validation.

        Args:
            X (numpy array): Input features (N, d).
            y (numpy array): Labels (N,).
            k (int): Number of folds for cross-validation.

        Returns:
            float: Mean validation score (MSE for regression, accuracy for classification).
        """
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        all_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            # Standardize features
            scaler = StandardScaler().fit(X_train)
            X_train, X_val = scaler.transform(X_train), scaler.transform(X_val)
            # Train model
            self.model.fit(X_train, y_train)
            # Predict on validation set
            y_pred = self.model.predict(X_val)
            # Compute validation score
            if self.task_type == "regression":
                score = mean_squared_error(y_val, y_pred)  # MSE for regression
            else:
                y_pred = (y_pred > 0.5).astype(int)  # Convert to binary for classification
                score = accuracy_score(y_val, y_pred)  # Accuracy for classification
            all_scores.append(score)

        mean_score = np.mean(all_scores)
        return mean_score, np.var(all_scores)