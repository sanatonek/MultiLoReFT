import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import numpy as np
import argparse
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# add parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils import *
from losses import *


class MultimodalDataset(Dataset):
    """Dataset class for multimodal data."""
    def __init__(self, h1, h2, x1, x2, labels):
        self.h1 = torch.tensor(h1, dtype=torch.float32)
        self.h2 = torch.tensor(h2, dtype=torch.float32)
        self.x1 = torch.tensor(x1, dtype=torch.float32)
        self.x2 = torch.tensor(x2, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.h1[idx], self.h2[idx], self.x1[idx], self.x2[idx], self.labels[idx]


class ProjectionModule(nn.Module):
    """Neural network module for multimodal projection learning."""
    
    def __init__(self, input_dims, shared_rank, specific_rank):
        super(ProjectionModule, self).__init__()
        self.shared_rank = shared_rank
        self.specific_rank = specific_rank
        self.threshold = 0.05
        self.pruned = False
        
        # Initialize projection matrices
        self.R_s = nn.Parameter(torch.randn(shared_rank, input_dims[0], dtype=torch.float32))
        self.R_m1 = nn.Parameter(torch.randn(specific_rank, input_dims[0], dtype=torch.float32))
        self.R_m2 = nn.Parameter(torch.randn(specific_rank, input_dims[1], dtype=torch.float32))
        
        # Initialize weights
        self._orthogonal_init()
        
        # Create weight networks for each modality
        self._create_weight_networks(input_dims)
    
    def _create_weight_networks(self, input_dims):
        """Create weight networks for each modality."""
        # Modality 1 weights
        self.W_s0 = nn.Sequential(
            nn.Linear(input_dims[0], input_dims[0] * 2, dtype=torch.float32),
            nn.ReLU(),
            #nn.Linear(input_dims[0] * 2, self.specific_rank, dtype=torch.float32) # old lead to shape mismatches when ranks were not the same
            nn.Linear(input_dims[0] * 2, self.shared_rank, dtype=torch.float32)
        )
        
        self.W_m0 = nn.Sequential(
            nn.Linear(input_dims[0], input_dims[0] * 2, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(input_dims[0] * 2, self.specific_rank, dtype=torch.float32)
        )
        
        # Modality 2 weights
        self.W_s1 = nn.Sequential(
            nn.Linear(input_dims[1], input_dims[1] * 2, dtype=torch.float32),
            nn.ReLU(),
            #nn.Linear(input_dims[1] * 2, self.specific_rank, dtype=torch.float32) # old lead to shape mismatches when ranks were not the same
            nn.Linear(input_dims[1] * 2, self.shared_rank, dtype=torch.float32)
        )
        
        self.W_m1 = nn.Sequential(
            nn.Linear(input_dims[1], input_dims[1] * 2, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(input_dims[1] * 2, self.specific_rank, dtype=torch.float32)
        )
    
    def _orthogonal_init(self):
        """Initialize projection matrices with uniform distribution."""
        torch.nn.init.uniform_(self.R_s, -0.9, 0.9)
        torch.nn.init.uniform_(self.R_m1, -0.9, 0.9)
        torch.nn.init.uniform_(self.R_m2, -0.9, 0.9)
    
    def get_trainable_parameters(self):
        """Get parameters to train based on current stage."""
        if self.trainable_stage == "shared":
            return [self.R_s] + list(self.W_s0.parameters()) + list(self.W_s1.parameters())
        elif self.trainable_stage == "private":
            return [self.R_m1, self.R_m2] + list(self.W_m0.parameters()) + list(self.W_m1.parameters())
        else:  # joint
            return list(self.parameters())
    
    def prune_singular_values(self):
        """Prune singular values below threshold and update network weights."""
        def prune_matrix(name, R, weights_to_prune):
            U, S, V = torch.svd(R)
            if len(S) < 2:
                return R, len(S)
            
            min_sv_idx = torch.argmin(S)
            min_sv = S[min_sv_idx]
            if min_sv > self.threshold:
                return R, len(S)
            
            # Create mask for keeping dimensions
            keep_indices = torch.ones(R.shape[0], dtype=torch.bool)
            keep_indices[:len(S)][min_sv_idx] = False
            reduced_R = R[keep_indices, :]
            
            # Update weight networks
            for weight_seq in weights_to_prune:
                last_layer = weight_seq[-1]
                in_features = last_layer.in_features
                new_layer = nn.Linear(in_features, keep_indices.sum().item(), dtype=torch.float32)
                new_layer.weight.data = last_layer.weight.data[keep_indices, :]
                new_layer.bias.data = last_layer.bias.data[keep_indices]
                weight_seq[-1] = new_layer
            
            # Update parameter
            del self._parameters[name]
            self.register_parameter(name, nn.Parameter(reduced_R))
            return getattr(self, name), keep_indices.sum().item()
            print(f"Pruned dimensions kept {kept_s}")
        
        # Prune each matrix
        kept_s, kept_m1, kept_m2 = 0, 0, 0
        if len(self.R_s) > 2:
            self.R_s, kept_s = prune_matrix("R_s", self.R_s, [self.W_s0, self.W_s1])
            self.shared_rank = kept_s
        if len(self.R_m1) > 2:
            self.R_m1, kept_m1 = prune_matrix("R_m1", self.R_m1, [self.W_m0])
            self.specific_rank = kept_m1
        if len(self.R_m2) > 2:
            self.R_m2, kept_m2 = prune_matrix("R_m2", self.R_m2, [self.W_m1])
            self.specific_rank = kept_m2
        
        # print(f"Pruned dimensions: Shared kept {kept_s}, Modality1 kept {kept_m1}, Modality2 kept {kept_m2}")
    
    def update_optimizer(self, optimizer):
        """Update optimizer after pruning."""
        return torch.optim.Adam(self.parameters(), lr=optimizer.param_groups[0]["lr"])
    
    def forward(self, embeddings):
        """Forward pass through the network."""
        h1, h2 = embeddings[0], embeddings[1]
        
        # old lead to shape mismatches when ranks were not the same
        h1_out = torch.matmul((self.W_s0(h1) - torch.matmul(h1, self.R_s.T)), self.R_s) + \
                 torch.matmul((self.W_m0(h1) - torch.matmul(h1, self.R_m1.T)), self.R_m1)
        h2_out = torch.matmul((self.W_s1(h2) - torch.matmul(h2, self.R_s.T)), self.R_s) + \
                 torch.matmul((self.W_m1(h2) - torch.matmul(h2, self.R_m2.T)), self.R_m2)
        
        # Compute shared projections - both weight output and projection match shared_rank
        #shared_h1 = torch.matmul((self.W_s0(h1) - torch.matmul(h1, self.R_s.T)), self.R_s)
        #shared_h2 = torch.matmul((self.W_s1(h2) - torch.matmul(h2, self.R_s.T)), self.R_s)
        # Compute modality-specific projections - both weight output and projection match specific_rank
        #specific_h1 = torch.matmul((self.W_m0(h1) - torch.matmul(h1, self.R_m1.T)), self.R_m1)
        #specific_h2 = torch.matmul((self.W_m1(h2) - torch.matmul(h2, self.R_m2.T)), self.R_m2)
        # Combine projections
        #h1_out = shared_h1 + specific_h1
        #h2_out = shared_h2 + specific_h2
        
        return h1_out, h2_out
    
    def decouple(self, phis, full=True, th=0.1):
        """Separate shared and modality-specific representations."""
        rep_components = []
        
        # Get singular values
        s_values = torch.svd(self.R_s)[1]
        shared_sv = torch.where(s_values > th)[0]
        m1_values = torch.svd(self.R_m1)[1]
        m1_sv = torch.where(m1_values > th)[0]
        m2_values = torch.svd(self.R_m2)[1]
        m2_sv = torch.where(m2_values > th)[0]
        
        # Decouple representations
        for i, z in enumerate(phis):
            if full:
                zs = torch.matmul(z, self.R_s.T)
                zm = torch.matmul(z, (self.R_m1.T if i==0 else self.R_m2.T))
            else:
                zs = torch.matmul(z, self.R_s[shared_sv].T)
                zm = torch.matmul(z, (self.R_m1[m1_sv].T if i==0 else self.R_m2[m2_sv].T))
            rep_components.append((zm, zs))
        
        return rep_components


def evaluate_validation_loss(model, val_dataloader, device):
    """Evaluate model on validation set."""
    model.eval()
    val_total_loss = 0
    
    with torch.no_grad():
        for val_batch in val_dataloader:
            h1, h2, x1, x2, label = val_batch
            h1 = F.normalize(h1.float(), dim=1).to(device)
            h2 = F.normalize(h2.float(), dim=1).to(device)
            x1 = x1.float().to(device)
            x2 = x2.float().to(device)
            
            phis = model([h1, h2])
            z_components = model.decouple(phis, full=True)
            losses_list, _, _, _ = compute_stage_losses(model, h1, h2, z_components, model.trainable_stage)
            val_loss = torch.stack(losses_list).mean()
            val_total_loss += val_loss.item()
    
    model.train()
    return val_total_loss / len(val_dataloader)

def evaluate_regression(z_n, h1, h2):
    train_size = int(0.8 * len(h1))
    regression_df = pd.DataFrame(columns=["name", "r2_mean", "r2_std"])
    #print(f'Regression predictability on {train_size} samples (validation on {len(h) - train_size} samples)')
    for i in range(2):
        for j in range(3):
            if i == 0:
                h = h1
            else:
                h = h2
            if j == 2:
                z = torch.cat((z_n[i][0], z_n[i][1]), dim=1)
            else:
                z = z_n[i][j]

            # perform "parallel" regression with linear NN
            train_loader = DataLoader(torch.cat((h[:train_size], z[:train_size]), dim=1), batch_size=64, shuffle=True)
            val_loader = DataLoader(torch.cat((h[train_size:], z[train_size:]), dim=1), batch_size=64, shuffle=False)
            linear = torch.nn.Linear(z.shape[1], h.shape[1]).to(z.device)
            optimizer = torch.optim.Adam(linear.parameters(), lr=0.001, weight_decay=0)
            loss_fn = torch.nn.MSELoss()
            early_stopping = 10
            val_losses = []
            #reg_pbar = tqdm(range(1000), desc="Training linear regression", leave=True)
            #for epoch in reg_pbar:
            for epoch in range(1000):
                train_loss = 0
                linear.train()
                for batch in train_loader:
                    optimizer.zero_grad()
                    pred = linear(batch[:, h.shape[1]:])
                    #print(batch.shape, pred.shape)
                    loss = loss_fn(pred, batch[:, :h.shape[1]])
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    train_loss += loss.item()
                train_loss /= len(train_loader)
                linear.eval()
                val_losses.append(0)
                for batch in val_loader:
                    #optimizer.zero_grad()
                    with torch.no_grad():
                        pred = linear(batch[:, h.shape[1]:])
                        loss = loss_fn(pred, batch[:, :h.shape[1]])
                        val_losses[-1] += loss.item()
                val_losses[-1] /= len(val_loader)
                if epoch > early_stopping and min(val_losses[-early_stopping:]) > min(val_losses):
                    break
                #reg_pbar.set_postfix({"loss": round(train_loss, 4), "val_loss": round(val_losses[-1], 4)})
            
            h_pred = linear(z)
            h_pred = h_pred.detach().cpu().numpy()
            h_mean = h.mean(0).cpu().numpy()
            r_squares = 1 - (((h.cpu().numpy() - h_pred)**2).sum(0) / ((h.cpu().numpy() - h_mean)**2).sum(0))
            if j == 0:
                name = "Zm"
            elif j == 2:
                name = "(Zm+Zs)"
            else:
                name = "Zs"
            #print(f'{name}{i+1} -----Goodness of fit (R2 score): {r_squares.mean():.3f} (var: {r_squares.std():.3f})')
            regression_df = pd.concat([regression_df, pd.DataFrame({
                "name": [f"{name}{i+1}"],
                "r2_mean": [r_squares.mean()],
                "r2_std": [r_squares.std()]
            })], ignore_index=True)

    return regression_df

def evaluate_classification(z_n, labels):
    """Evaluate how well each component (shared and modality-specific) can predict the target label.
    
    Args:
        z_n: Tuple of (modality_specific, shared) representations for each modality
        labels: Target labels for binary classification
    """
    #print(f'Classification predictability')
    components = [
        ("Zs1", z_n[0][1]),  # Shared representation from modality 1
        ("Zs2", z_n[1][1]),  # Shared representation from modality 2
        ("Zm1", z_n[0][0]),  # Modality-specific representation from modality 1
        ("Zm2", z_n[1][0])   # Modality-specific representation from modality 2
    ]

    class_df = pd.DataFrame(columns=["name", "accuracy", "precision", "recall", "f1", "roc_auc"])
    
    # train a classifier on the shared and modality-specific representations
    #print('Task: Binary classification')
    for name, z in components:
        try:
            # Prepare data
            z_np = z.detach().cpu().numpy()
            y_np = labels.astype(int)
            
            # Check if binary classification
            unique_classes = np.unique(y_np)
            if len(unique_classes) != 2:
                print(f"Warning: Expected binary classification but found {len(unique_classes)} classes. Skipping {name}.")
                continue
                
            # Initialize classifier with balanced class weights for robustness
            model = LogisticRegression(
                max_iter=1000, 
                class_weight='balanced',
                solver='liblinear',  # Works well for small datasets
                random_state=args.seed
            )
            
            scoring = {
                'accuracy': make_scorer(accuracy_score),
                'precision': make_scorer(precision_score),
                'recall': make_scorer(recall_score),
                'f1': make_scorer(f1_score),
                'roc_auc': make_scorer(roc_auc_score)
            }
            
            cv_results = cross_validate(
                model, z_np, y_np, 
                cv=5, 
                scoring=scoring,
                return_train_score=False
            )
            
            # Print results
            #print(f"{name} -----Accuracy:  {cv_results['test_accuracy'].mean():.3f} ± {cv_results['test_accuracy'].std():.3f}",
            #      f"  Precision: {cv_results['test_precision'].mean():.3f} ± {cv_results['test_precision'].std():.3f}",
            #      f"  Recall:    {cv_results['test_recall'].mean():.3f} ± {cv_results['test_recall'].std():.3f}",
            #      f"  F1 Score:  {cv_results['test_f1'].mean():.3f} ± {cv_results['test_f1'].std():.3f}",
            #      f"  ROC AUC:   {cv_results['test_roc_auc'].mean():.3f} ± {cv_results['test_roc_auc'].std():.3f}")
            class_df = pd.concat([class_df, pd.DataFrame({
                "name": [name],
                "accuracy": [cv_results['test_accuracy'].mean()],
                "precision": [cv_results['test_precision'].mean()],
                "recall": [cv_results['test_recall'].mean()],
                "f1": [cv_results['test_f1'].mean()],
                "roc_auc": [cv_results['test_roc_auc'].mean()]
            })], ignore_index=True)
            
        except Exception as e:
            print(f"Error evaluating {name}: {str(e)}")
            continue
    
    return class_df

def calc_correlation_matrix(model, threshold=0.05):
    # Get matrices above threshold
    shared_sv = torch.where(torch.linalg.svdvals(model.R_s) > threshold)
    Rs = model.R_s[shared_sv].detach().cpu().numpy()
    m1_sv = torch.where(torch.linalg.svdvals(model.R_m1) > threshold)
    Rm1 = model.R_m1[m1_sv].detach().cpu().numpy()
    m2_sv = torch.where(torch.linalg.svdvals(model.R_m2) > threshold)
    Rm2 = model.R_m2[m2_sv].detach().cpu().numpy()
    matrices = [Rs, Rm1, Rm2]

    names = ['R_s', 'R_m1', 'R_m2']
    corr_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            flat_i = matrices[i].flatten()
            flat_j = matrices[j].flatten()
            if len(flat_i) > len(flat_j):
                flat_j = np.pad(flat_j, (0, len(flat_i) - len(flat_j)))
            elif len(flat_j) > len(flat_i):
                flat_i = np.pad(flat_i, (0, len(flat_j) - len(flat_i)))
            corr_matrix[i,j] = np.corrcoef(flat_i, flat_j)[0,1]
    
    corr_df = pd.DataFrame({
        "name": ['R_s-R_m1', 'R_s-R_m2', 'R_m1-R_m2'],
        "metric": ["correlation"]*3,
        "value": [corr_matrix[0,1], corr_matrix[0,2], corr_matrix[1,2]]
    })
    
    return corr_df

def eval_model(model, test_h1, test_h2, test_labels, device, threshold=0.05):
    # regression and classification
    h1 = F.normalize(torch.Tensor(test_h1).float(), dim=1).to(device)
    h2 = F.normalize(torch.Tensor(test_h2).float(), dim=1).to(device)
    phis = model([h1,h2])
    z_n = model.decouple(phis, full=True, th=0.05)
    regression_df = evaluate_regression(z_n, h1, h2)
    classification_df = evaluate_classification(z_n, test_labels[:, 0])

    # rank and corr matrix
    corr_df = calc_correlation_matrix(model, threshold)
    shared_sv = len(torch.where(torch.linalg.svdvals(model.R_s) > threshold)[0].tolist())
    m1_sv = len(torch.where(torch.linalg.svdvals(model.R_m1) > threshold)[0].tolist())
    m2_sv = len(torch.where(torch.linalg.svdvals(model.R_m2) > threshold)[0].tolist())
    rank_df = pd.DataFrame({
        "name": ["R_s", "R_m1", "R_m2"],
        "metric": ["rank"]*3,
        "value": [shared_sv, m1_sv, m2_sv]
    })
    rank_df = pd.concat([corr_df, rank_df], axis=0, ignore_index=True)

    return regression_df, classification_df, rank_df

def train(model, dataloader, val_dataloader, device, epochs=1000, lr=1e-4, weight_decay=1e-4, loss_balance='gradient', scheduling='cosine', model_name="model", patience1=15, patience2=20):
    """Train the projection model with early stopping."""
    # Initialize loss tracking
    if loss_balance == 'gradient':
        loss_balancer = GradientNormalizedLoss(num_losses=3)
    else:
        raise ValueError(f"Unknown loss balance method: {loss_balance}. TBD")
    all_epoch_losses = []
    lrs = []
    all_epoch_stages = []
    
    # Early stopping configuration
    early_stopping_config = {
        "shared": {
            "patience": patience1,
            "min_improvement_ratio": 0.01,
            "min_epochs": 50,
            "max_epochs": 500
        },
        "private": {
            "patience": patience1,
            "min_improvement_ratio": 0.01,
            "min_epochs": 50,
            "max_epochs": 500
        },
        "joint": {
            "patience": patience2,
            "min_improvement_ratio": 0.005,
            "min_epochs": 100,
            "max_epochs": 2000
        }
    }
    
    # Initialize stage tracking
    stage_tracking = {
        "best_val_loss": 500,#float('inf'),
        "plateau_counter": 0,
        "min_epochs_counter": 0,
        "last_val_loss": 500,#float('inf'),
        "initial_loss": None
    }
    
    # Training loop
    ranks_per_epoch = {
        "Rs": [],
        "Rm1": [],
        "Rm2": []
    }
    pruning_per_epoch = []
    for epoch in range(epochs):
        total_loss = 0
        epoch_losses = np.zeros(3)
        
        # Initialize stage if first epoch
        if epoch == 0:
            model.trainable_stage = "shared"
            trainable_params = model.get_trainable_parameters()
            # Initialize optimizer and scheduler
            optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
            if scheduling == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
            elif scheduling == 'exponential':
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
            elif scheduling == 'constant':
                scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
            elif scheduling == 'reduceonplateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
            print(f"[Epoch {epoch}] → Starting with SHARED stage.")
            stage_tracking = {
                "best_val_loss": 500,#float('inf'),
                "plateau_counter": 0,
                "min_epochs_counter": 0,
                "last_val_loss": 500,#float('inf'),
                "initial_loss": None
            }
        
        # Evaluate validation loss
        val_loss = evaluate_validation_loss(model, val_dataloader, device)
        
        # Update stage tracking
        if stage_tracking["initial_loss"] is None:
            stage_tracking["initial_loss"] = val_loss
            print(f"Initial {model.trainable_stage} stage loss: {val_loss:.4f}")
        
        current_stage = model.trainable_stage
        stage_config = early_stopping_config[current_stage]
        stage_tracking["min_epochs_counter"] += 1
        
        # Calculate improvement
        relative_improvement = (stage_tracking["best_val_loss"] - val_loss) / stage_tracking["best_val_loss"]
        required_improvement = stage_config["min_improvement_ratio"]
        
        # Update tracking metrics
        if relative_improvement > required_improvement:
            stage_tracking["best_val_loss"] = val_loss
            stage_tracking["plateau_counter"] = 0
            # print(f"Significant improvement in {current_stage} stage: {relative_improvement*100:.2f}%")
        else:
            stage_tracking["plateau_counter"] += 1
        
        # Check for stage transition
        should_switch = (
            (stage_tracking["min_epochs_counter"] >= stage_config["min_epochs"] and
             stage_tracking["plateau_counter"] >= stage_config["patience"]) or
            stage_tracking["min_epochs_counter"] >= stage_config["max_epochs"]
        )
        
        if should_switch:
            if current_stage == "shared":
                model.trainable_stage = "private"
                print(f"\t\t[Epoch {epoch}] → Switched to PRIVATE stage after {stage_tracking['min_epochs_counter']} epochs")
            elif current_stage == "private":
                model.trainable_stage = "joint"
                print(f"\t\t[Epoch {epoch}] → Switched to JOINT stage after {stage_tracking['min_epochs_counter']} epochs")
            elif current_stage == "joint":
                print(f"\t\t[Epoch {epoch}] → JOINT stage reached after {stage_tracking['min_epochs_counter']} epochs")
                print(f"Final {current_stage} stage loss: {val_loss:.4f} (initial: {stage_tracking['initial_loss']:.4f})")
                break
            
            # Reset tracking
            stage_tracking = {
                "best_val_loss": 500,#float('inf'),
                "plateau_counter": 0,
                "min_epochs_counter": 0,
                "last_val_loss": 500,#float('inf'),
                "initial_loss": None
            }
            
            # Update optimizer
            trainable_params = model.get_trainable_parameters()
            optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
            if scheduling == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
            elif scheduling == 'exponential':
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
            elif scheduling == 'constant':
                scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
            elif scheduling == 'reduceonplateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        # Training step
        for batch in dataloader:
            h1, h2, x1, x2, label = batch
            h1 = F.normalize(h1.float(), dim=1).to(device)
            h2 = F.normalize(h2.float(), dim=1).to(device)
            x1 = x1.float().to(device)
            x2 = x2.float().to(device)
            
            phis = model([h1, h2])
            z_components = model.decouple(phis, full=True)
            losses_list, loss_names, all_losses, all_loss_names = compute_stage_losses(
                model, h1, h2, z_components, model.trainable_stage)
            
            losses = torch.stack(losses_list)
            loss, weights = loss_balancer(losses, model, trainable_params)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            epoch_losses += all_losses
        
        # Update metrics
        epoch_losses = epoch_losses / len(dataloader)
        all_epoch_losses.append(epoch_losses)
        lrs.append(optimizer.param_groups[0]['lr'])
        all_epoch_stages.append(model.trainable_stage)
        scheduler.step()
        
        # Prune if in joint stage
        if model.trainable_stage == "joint" and epoch_losses[2] < stage_tracking['best_val_loss']:  # Index 2 is MI loss based on all_loss_names
            model.prune_singular_values()
            optimizer = model.update_optimizer(optimizer)
            pruning_per_epoch.append(1)
        else:
            pruning_per_epoch.append(0)
        
        threshold = 0.05
        shared_sv = len(torch.where(torch.linalg.svdvals(model.R_s) > threshold)[0].tolist())
        m1_sv = len(torch.where(torch.linalg.svdvals(model.R_m1) > threshold)[0].tolist())
        m2_sv = len(torch.where(torch.linalg.svdvals(model.R_m2) > threshold)[0].tolist())
        ranks_per_epoch["Rs"].append(shared_sv)
        ranks_per_epoch["Rm1"].append(m1_sv)
        ranks_per_epoch["Rm2"].append(m2_sv)
    
    plot_losses(all_epoch_losses, loss_names=all_loss_names, save_path=f"./plots/{model_name}_loss.pdf")
    save_checkpoint(model, optimizer, epoch, loss, filepath=f"./ckpts/{model_name}.pth")

    # return all training losses
    train_df = pd.DataFrame(all_epoch_losses, columns=all_loss_names)
    train_df['lr'] = lrs
    train_df['stage'] = all_epoch_stages
    train_df['rank Rs'] = ranks_per_epoch["Rs"]
    train_df['rank Rm1'] = ranks_per_epoch["Rm1"]
    train_df['rank Rm2'] = ranks_per_epoch["Rm2"]
    train_df['pruning'] = pruning_per_epoch
    train_df['epoch'] = np.arange(len(train_df))
    return train_df

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, filepath)

def main(dev_id=0, seed=0, out_dir="./results/", file_name="sweep", sim_data='sim_100000_in5-5_data5-5_shared2_c2_shared', subset=10000):
    """Main function to run the training pipeline."""
    # add the seed to the file name
    file_name = f"{file_name}_seed{seed}_{sim_data}_subset{subset}"
    device = torch.device(f"cuda:{dev_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # create all the necessary directories if not existing
    if not os.path.exists("./ckpts"):
        os.makedirs("./ckpts")
    if not os.path.exists("./plots"):
        os.makedirs("./plots")
    if not os.path.exists("./data"):
        os.makedirs("./data")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    # create a folder for the sweep plots
    if not os.path.exists(f"./plots/{file_name}"):
        os.makedirs(f"./plots/{file_name}")
    if not os.path.exists(f"./ckpts/{file_name}"):
        os.makedirs(f"./ckpts/{file_name}")
    
    # load data
    loaded_data = np.load("./data/"+sim_data+".npz")
    input_dims = [int(sim_data.split("_in")[-1].split("-")[0]), int(sim_data.split("_in")[-1].split("_data")[0].split("-")[1])]
    h1, h2, x1, x2, labels = loaded_data["h1"], loaded_data["h2"], loaded_data["x1"], loaded_data["x2"], loaded_data["labels"]
    h1 = h1[:subset]
    h2 = h2[:subset]
    x1 = x1[:subset]
    x2 = x2[:subset]
    labels = labels[:subset]
    # Create datasets
    n_train = int(0.8 * h1.shape[0])
    n_val = int(0.1 * h1.shape[0])
    dataset = MultimodalDataset(h1[:n_train], h2[:n_train], x1[:n_train], x2[:n_train], labels[:n_train])
    val_dataset = MultimodalDataset(h1[n_train:n_train+n_val], h2[n_train:n_train+n_val], x1[n_train:n_train+n_val], x2[n_train:n_train+n_val], labels[n_train:n_train+n_val])
    #test_dataset = MultimodalDataset(h1[n_train+n_val:], h2[n_train+n_val:], x1[n_train+n_val:], x2[n_train+n_val:], labels[n_train+n_val:])
    
    # set up the sweep hyperparameters
    batch_sizes = [128]
    learning_rates = [1e-5]
    lr_annealing = ['cosine']
    weight_decays = [0]
    n_specific_rank = [2, 4, 6, 8, 10]
    n_shared_rank = [2, 4, 6, 8, 10]
    epochs = 3000
    joint_patience = 50
    patience = 20
    n_combinations = len(batch_sizes) * len(learning_rates) * len(lr_annealing) * len(weight_decays) * len(n_specific_rank) * len(n_shared_rank)
    print(f"Running {n_combinations} combinations of hyperparameters...")

    # sweep
    for bs in batch_sizes:
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=bs, shuffle=True)
        for lr in learning_rates:
            for la in lr_annealing:
                for wd in weight_decays:
                    for sr in n_specific_rank:
                        for sh in n_shared_rank:
                            print(f"Training with batch_size={bs}, learning_rate={lr}, weight_decay={wd}, specific_rank={sr}, shared_rank={sh}")
                            # Initialize model
                            projection_model = ProjectionModule(
                                input_dims=input_dims,
                                shared_rank=sh, 
                                specific_rank=sr
                            ).to(device)
                            
                            # Train model
                            train_df = train(projection_model, dataloader, val_dataloader, device, epochs=epochs, lr=lr, weight_decay=wd, loss_balance='gradient', scheduling=la, model_name=f"{file_name}/bs{bs}_lr{lr}_la-{la}_wd{wd}_sr{sr}_sh{sh}", patience1=patience, patience2=joint_patience)
                            regression_df, classification_df, rank_df = eval_model(projection_model, h1[n_train+n_val:], h2[n_train+n_val:], labels[n_train+n_val:], device)
                            param_df = pd.DataFrame({
                                "batch_size": [bs],
                                "learning_rate": [lr],
                                "lr_annealing": [la],
                                "weight_decay": [wd],
                                "specific_rank": [sr],
                                "shared_rank": [sh],
                                "patience1": [patience],
                                "patience2": [joint_patience],
                            })
                            # add to the result dfs (repeat the len to match each df)
                            train_df = pd.concat([train_df, pd.concat([param_df]*len(train_df), ignore_index=True)], axis=1)
                            regression_df = pd.concat([regression_df, pd.concat([param_df]*len(regression_df), ignore_index=True)], axis=1)
                            classification_df = pd.concat([classification_df, pd.concat([param_df]*len(classification_df), ignore_index=True)], axis=1)
                            rank_df = pd.concat([rank_df, pd.concat([param_df]*len(rank_df), ignore_index=True)], axis=1)

                            # Save results (extend file if already exists)
                            if os.path.exists(f"{out_dir}/{file_name}_train.csv"):
                                train_df.to_csv(f"{out_dir}/{file_name}_train.csv", mode='a', header=False, index=False)
                            else:
                                train_df.to_csv(f"{out_dir}/{file_name}_train.csv", index=False)
                            if os.path.exists(f"{out_dir}/{file_name}_regression.csv"):
                                regression_df.to_csv(f"{out_dir}/{file_name}_regression.csv", mode='a', header=False, index=False)
                            else:
                                regression_df.to_csv(f"{out_dir}/{file_name}_regression.csv", index=False)
                            if os.path.exists(f"{out_dir}/{file_name}_classification.csv"):
                                classification_df.to_csv(f"{out_dir}/{file_name}_classification.csv", mode='a', header=False, index=False)
                            else:
                                classification_df.to_csv(f"{out_dir}/{file_name}_classification.csv", index=False)
                            if os.path.exists(f"{out_dir}/{file_name}_rank.csv"):
                                rank_df.to_csv(f"{out_dir}/{file_name}_rank.csv", mode='a', header=False, index=False)
                            else:
                                rank_df.to_csv(f"{out_dir}/{file_name}_rank.csv", index=False)


if __name__ == "__main__":
    # Add command-line argument parsing
    parser = argparse.ArgumentParser(description="Train multimodal projection model")
    parser.add_argument("--gpu", type=int, default=0, 
                      help="GPU device ID to use (default: 0)")
    parser.add_argument("--seed", type=int, default=0,
                      help="Random seed for reproducibility (default: 0)")
    parser.add_argument("--out_dir", type=str, default="./results/",
                      help="Output directory for results (default: './results/')")
    parser.add_argument("--file_name", type=str, default="sweep",
                      help="Base name for output files (default: 'sweep')")
    parser.add_argument("--data", type=str, default="sim_100000_in5-5_data5-5_shared2_c2_shared")
    parser.add_argument("--subset", type=int, default=10000,
                        help="Subset size for data (default: 10000)")
    args = parser.parse_args()
    
    # Call main with the specified GPU ID
    main(dev_id=args.gpu, seed=args.seed, out_dir=args.out_dir, file_name=args.file_name, sim_data=args.data, subset=args.subset)