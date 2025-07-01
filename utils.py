import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score
import torchvision.transforms as transforms
from skorch import NeuralNetRegressor


import matplotlib.pyplot as plt
import seaborn as sns
sns.set() 

class MultiHeadRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def get_multihead_regressor(input_dim, output_dim, device='cpu'):
    return NeuralNetRegressor(
        module=MultiHeadRegressor,
        module__input_dim=input_dim,
        module__output_dim=output_dim,
        max_epochs=20,
        lr=1e-3,
        device=device,
        verbose=0,
    )

def get_dino_preprocess(image_size=518):
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def _patch_linear_layer(layer, new_out_features):
    in_features = layer.in_features
    new_layer = nn.Linear(in_features, new_out_features, dtype=layer.weight.dtype)
    return new_layer

def load_checkpoint(filepath, model, optimizer=None):
    checkpoint = torch.load(filepath, map_location="cpu")
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
    return model


class SklearnTrainer:
    def __init__(self, model, task_type="regression"):
        """
        Wrapper for training a Scikit-Learn model with k-fold cross-validation.

        Args:
            model: A Scikit-Learn model instance.
            task_type (str): "regression", "binary", or "multiclass".
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
            tuple: Mean and variance of validation scores.
        """
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        all_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train).squeeze()
            X_val = scaler.transform(X_val).squeeze()

            if self.task_type == "neural_multihead":
                scaler_y = StandardScaler().fit(y_train)
                y_train = y_train.squeeze()
                y_val = y_val.squeeze()
                y_train = scaler_y.transform(y_train)
                y_val = scaler_y.transform(y_val)
                input_dim = X_train.shape[1]
                output_dim = y_train.shape[1] if y_train.ndim > 1 else 1
                model = get_multihead_regressor(input_dim, output_dim)
                model.fit(X_train.astype(np.float32), y_train.astype(np.float32))
                preds = model.predict(X_val.astype(np.float32))
                score = np.mean((preds - y_val) ** 2)  # MSE

            elif self.task_type == "regression":
                self.model.fit(X_train, y_train)
                y_pred = self.model.predict(X_val)
                score = mean_squared_error(y_val, y_pred)

            elif self.task_type == "binary":
                self.model.fit(X_train, y_train)
                # Binary classification: use probabilities if available
                if hasattr(self.model, "predict_proba"):
                    y_pred = self.model.predict_proba(X_val)[:, 1]
                else:
                    y_pred = self.model.predict(X_val)
                score = roc_auc_score(y_val, y_pred)

            elif self.task_type == "multiclass":
                self.model.fit(X_train, y_train)
                if hasattr(self.model, "predict_proba"):
                    y_pred = self.model.predict_proba(X_val)
                    score = roc_auc_score(y_val, y_pred, multi_class='ovr', average='macro')
                else:
                    y_pred = self.model.predict(X_val)
                    score = accuracy_score(y_val, y_pred)

            else:
                raise ValueError(f"Unsupported task type: {self.task_type}")

            all_scores.append(score)

        return np.mean(all_scores), np.var(all_scores)
    
def plot_losses(losses, loss_names, save_path=None, log_path=None, stage_switches=None):
    """Plot loss curves in separate horizontal subplots and save loss values."""
    # Convert losses to numpy array if it's not already
    losses = np.array(losses)
    
    # Calculate total loss
    total_loss = np.sum(losses, axis=1)
    # all_losses = np.column_stack(losses)
    all_losses = np.column_stack([losses, total_loss])
    # all_names = loss_names #+ 
    all_names = loss_names + ['Total Loss']
    
    # Create figure with subplots
    num_losses = len(all_names)
    fig, axes = plt.subplots(1, num_losses, figsize=(5*num_losses, 5))
    
    # If there's only one loss, make axes iterable
    if num_losses == 1:
        axes = [axes]
    
    # Plot each loss in its own subplot
    for i, (ax, name) in enumerate(zip(axes, all_names)):
        ax.plot(all_losses[:, i], label=name, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss Value')
        ax.set_title(name)
        ax.legend()
        ax.grid(True)
        if stage_switches:
            for stage, epoch in stage_switches:
                ax.axvline(x=epoch, color='red', linestyle='--', label='Stage Switch')
    
    plt.tight_layout()
    
    if save_path:
        # Save plot
        plt.savefig(save_path)
        plt.close()
        
        # Save loss values to CSV
        csv_path = log_path
        import pandas as pd
        df = pd.DataFrame(all_losses, columns=all_names)
        df.to_csv(csv_path, index_label='epoch')
    else:
        plt.show()

def plot_weights(weights, weight_names, save_path):
    """Plot the evolution of loss weights over time in separate subplots"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    weights = np.array(weights)
    epochs = np.arange(len(weights))
    
    # Create a figure with subplots
    fig, axes = plt.subplots(len(weight_names), 1, figsize=(10, 4*len(weight_names)))
    if len(weight_names) == 1:
        axes = [axes]  # Make it iterable if only one weight
    
    for i, (ax, name) in enumerate(zip(axes, weight_names)):
        ax.plot(epochs, weights[:, i], label=name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Weight Value')
        ax.set_title(f'Evolution of {name}')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()