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
from utils import *
from losses import *
#from transformers import BertTokenizer, AutoTokenizer
from src.visualization import *
from src.utils import setup_wandb, log_wandb, custom_weight_init
from src.eval_metrics import calc_corrs_and_ranks, evaluate_validation_loss, eval_model, reeval_model

# ignore warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

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


class MultiLoReFT(nn.Module):
    """LoReFT module for multimodal projection learning."""
    def __init__(
            self, 
            input_dims, 
            shared_rank, 
            specific_rank, 
            staging=True, 
            pruning_threshold=1e-3, 
            pruning=True,
            r_init="uniform",
            device=None,
            zm_orhto_loss=False,
            scaling_factor=1,
            mi_loss=True,
            switch_criterion="improvement"
        ):
        super(MultiLoReFT, self).__init__()
        self.shared_rank = shared_rank
        self.specific_rank = specific_rank
        self.pruning_threshold = pruning_threshold
        self.pruned = False
        self.staging = staging
        self.pruning = False
        self.device = device
        self.scaling_factor = scaling_factor
        self.zm_ortho_loss = zm_orhto_loss
        self.mi_loss = mi_loss
        self.switch_criterion = switch_criterion
        if staging:
            self.trainable_stage = "shared"
            self.stage_tracking = {
                    "best_val_loss": 5000,
                    "plateau_counter": 0,
                    "min_epochs_counter": 0,
                }
        else:
            self.trainable_stage = "joint"
        
        self.encoder1 = nn.Sequential(
            nn.Linear(input_dims[0], input_dims[0] * 2, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(input_dims[0] * 2, input_dims[0], dtype=torch.float32) # just a nonlinear transformation should be fine
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(input_dims[1], input_dims[1] * 2, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(input_dims[1] * 2, input_dims[1], dtype=torch.float32)
        )
        self.sparse = nn.Sequential(
            nn.Linear(input_dims[0] + input_dims[1], (input_dims[0] + input_dims[1]) * self.scaling_factor, dtype=torch.float32), # actually dont need it sparse so smaller should be fine to be able to prune
            nn.Softplus()
        )

        # Initialize projection matrices
        self.R_s = nn.Parameter(torch.empty(self.shared_rank, (input_dims[0] + input_dims[1]) * self.scaling_factor, dtype=torch.float32))
        self.R_m1 = nn.Parameter(torch.empty(self.specific_rank, (input_dims[0] + input_dims[1]) * self.scaling_factor, dtype=torch.float32))
        self.R_m2 = nn.Parameter(torch.empty(self.specific_rank, (input_dims[0] + input_dims[1]) * self.scaling_factor, dtype=torch.float32))

        self.decoder1 = nn.Sequential(
            nn.Linear(self.shared_rank + self.specific_rank, input_dims[0] * 2, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(input_dims[0] * 2, input_dims[0], dtype=torch.float32)
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(self.shared_rank + self.specific_rank, input_dims[1] * 2, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(input_dims[1] * 2, input_dims[1], dtype=torch.float32)
        )
        self._orthogonal_init(r_init)
        
    
    def _orthogonal_init(self, r_init):
        """Initialize projection matrices with uniform distribution."""
        custom_weight_init(self.R_s, init_option=r_init)
        custom_weight_init(self.R_m1, init_option=r_init)
        custom_weight_init(self.R_m2, init_option=r_init)
    
    def get_trainable_parameters(self):
        """Get parameters to train based on current stage."""
        if self.trainable_stage == "shared":
            return [self.R_s] + list(self.encoder1.parameters()) + list(self.encoder2.parameters()) + list(self.sparse.parameters()) + list(self.decoder1.parameters()) + list(self.decoder2.parameters())
        elif self.trainable_stage == "private":
            return [self.R_m1, self.R_m2] + list(self.encoder1.parameters()) + list(self.encoder2.parameters()) + list(self.sparse.parameters()) + list(self.decoder1.parameters()) + list(self.decoder2.parameters())
        else:  # joint
            return list(self.parameters())
    
    def prune_singular_values(self):
        raise NotImplementedError("Pruning not implemented yet.")
    
    def update_optimizer(self, optimizer):
        """Update optimizer after pruning."""
        return torch.optim.Adam(self.parameters(), lr=optimizer.param_groups[0]["lr"])
    
    def forward(self, embeddings):
        """Forward pass through the network."""
        h1, h2 = embeddings[0], embeddings[1]

        h1_h = self.encoder1(h1)
        h2_h = self.encoder2(h2)
        h_h = torch.cat((h1_h, h2_h), dim=1)
        h_sparse = self.sparse(h_h)

        z_m1 = torch.matmul(h_sparse, self.R_m1.T)
        z_m2 = torch.matmul(h_sparse, self.R_m2.T)
        z_s = torch.matmul(h_sparse, self.R_s.T)

        h1_out = self.decoder1(torch.cat((z_m1, z_s), dim=1))
        h2_out = self.decoder2(torch.cat((z_m2, z_s), dim=1))
        return h1_out, h2_out


    def decouple(self, h, full=True, th=0.1):
        h1, h2 = h[0], h[1]

        h1_h = self.encoder1(h1)
        h2_h = self.encoder2(h2)
        h_h = torch.cat((h1_h, h2_h), dim=1)
        h_sparse = self.sparse(h_h)

        z_m1 = torch.matmul(h_sparse, self.R_m1.T)
        z_m2 = torch.matmul(h_sparse, self.R_m2.T)
        z_s = torch.matmul(h_sparse, self.R_s.T)

        rep_components = [
            [z_m1, z_s],  # Modality-specific and shared representations for modality 1
            [z_m2, z_s]   # Modality-specific and shared representations for modality 2
        ]
        
        return rep_components
    
    def get_sparsity(self, embeddings, which='h'):
        h1, h2 = embeddings[0], embeddings[1]
        h1_h = self.encoder1(h1)
        h2_h = self.encoder2(h2)
        h_h = torch.cat((h1_h, h2_h), dim=1)
        h_sparse = self.sparse(h_h)

        if which == 's':
            z_s = torch.matmul(h_sparse, self.R_s.T)
            # compute l1 loss on z_s as the avg number of active neurons
            l1_loss = torch.mean(torch.abs(z_s))
            #active_dims = torch.sum(torch.mean(torch.abs(z_s),dim=0) > self.pruning_threshold)
            #l1_loss = active_dims / z_s.shape[1]
            return l1_loss
        elif which == 'm':
            # compute l1 loss on z_m1 and z_m2
            z_m1 = torch.matmul(h_sparse, self.R_m1.T)
            z_m2 = torch.matmul(h_sparse, self.R_m2.T)
            l1_loss = torch.mean(torch.abs(z_m1)) + torch.mean(torch.abs(z_m2))
            #active_dims_m1 = torch.sum(torch.mean(torch.abs(z_m1),dim=0) > self.pruning_threshold)
            #active_dims_m2 = torch.sum(torch.mean(torch.abs(z_m2),dim=0) > self.pruning_threshold)
            #l1_loss = (active_dims_m1 + active_dims_m2) / (z_m1.shape[1] + z_m2.shape[1])
            return l1_loss
        else:
            # sparsity loss on the R matrices
            return 0 * (torch.mean(torch.abs(self.R_s)) + torch.mean(torch.abs(self.R_m1)) + torch.mean(torch.abs(self.R_m2)))

    def compute_stage_losses(self, h1, h2, z_components):
        # Compute all losses
        #l_shared = loss_shared_consistency(z_components[0][1], z_components[1][1])
        if self.trainable_stage == "joint":
            #l_sparsity = model.get_sparsity([h1, h2])
            #l_sparsity = model.get_sparsity([h1, h2], which='m') + model.get_sparsity([h1, h2], which='s') * 0
            l_sparsity = self.get_sparsity([h1, h2]) #* 0
        elif self.trainable_stage == "shared":
            l_sparsity = self.get_sparsity([h1, h2], which='s')
        else:
            l_sparsity = self.get_sparsity([h1, h2], which='m')
        l_orthogonal = loss_orthogonality(self.R_s, self.R_m1, self.R_m2)
        if self.zm_ortho_loss:
            l_orthogonal = l_orthogonal + (loss_orthogonal_embedding(z_components[0][0], z_components[0][1]) + loss_orthogonal_embedding(z_components[1][0], z_components[1][1]))
        #l_mi = loss_mutual_info(h1, h2, z_components)
        # reconstruction loss
        if self.mi_loss:
            l_mi = loss_mutual_info(h1, h2, z_components)
        else:
            # mse recon
            recon = self.forward([h1, h2])
            l_mi = torch.nn.functional.mse_loss(recon[0], h1) + torch.nn.functional.mse_loss(recon[1], h2)
        
        all_losses = [l_sparsity.item(), l_orthogonal.item(), l_mi.item()]
        all_loss_names = ["Sparsity Loss", "Orthogonal Loss", "Mutual Info Loss"]
        
        # Return appropriate losses based on stage
        if self.trainable_stage == "shared":
            return [l_sparsity, l_mi], ["Sparsity Loss", "Mutual Info Loss"], all_losses, all_loss_names
        elif self.trainable_stage == "private":
            return [l_orthogonal, l_mi], ["Orthogonal Loss", "Mutual Info Loss"], all_losses, all_loss_names
        elif self.trainable_stage == "joint":
            return [l_orthogonal, l_sparsity, l_mi], ["Orthogonal Loss", "Sparsity Loss", "Mutual Info Loss"], all_losses, all_loss_names

    def evaluate_validation_loss(self, val_dataloader):
        """Evaluate model on validation set."""
        val_total_loss = 0
        self.eval()
        with torch.no_grad():
            for val_batch in val_dataloader:
                h1, h2, x1, x2, label = val_batch
                
                h1 = F.normalize(h1.float(), dim=1).to(self.device)
                h2 = F.normalize(h2.float(), dim=1).to(self.device)
                phis = self.forward([h1, h2])
                
                z_components = self.decouple([h1,h2], full=True)
                losses_list, _, _, _ = self.compute_stage_losses(h1, h2, z_components)
                val_loss = torch.stack(losses_list).mean()
                val_total_loss += val_loss.item()
        self.train()
        return val_total_loss / len(val_dataloader)


    def train_projection(self, dataloader, val_dataloader, early_stopping_config, model_hyperparameters, epochs=100):
        """Train the projection model with early stopping."""
        #print(f"Training on device: {self.device}")
        #print(f"Model is on device: {next(self.parameters()).device}")
        # Initialize loss tracking
        loss_balancer = GradientNormalizedLoss(num_losses=3)
        lr = model_hyperparameters['learning_rate']
        wd = model_hyperparameters['weight_decay']
        
        trainable_params = self.get_trainable_parameters()
        optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=wd)
        if model_hyperparameters.get('lr_annealing') == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        elif model_hyperparameters.get('lr_annealing') == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        elif model_hyperparameters.get('lr_annealing') == 'linear':
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=early_stopping_config[self.trainable_stage]["max_epochs"])
        elif model_hyperparameters.get('lr_annealing') == 'constant':
            scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=epochs)
        
        # Training loop
        all_epoch_losses = []
        lrs = []
        all_epoch_stages = []
        for epoch in range(epochs):
            total_loss = 0
            epoch_losses = np.zeros(3)

            # Evaluate validation loss
            val_loss, val_logs, val_log_names = evaluate_validation_loss(self, val_dataloader, self.device)
            corr_rank_dict = calc_corrs_and_ranks(self)
            stage_int = [0 if self.trainable_stage == "shared" else 1 if self.trainable_stage == "private" else 2][0]
            log_dict = {
                "epoch": epoch,
                "stage": stage_int,
                "lr": optimizer.param_groups[0]['lr'],
                "relative_improvement": (self.stage_tracking["best_val_loss"] - val_loss) / self.stage_tracking["best_val_loss"]
            }
            #for i, loss_name in enumerate(loss_names):
            #    log_dict[loss_name] = None
            for i, val_log_name in enumerate(val_log_names):
                log_dict[val_log_name] = val_logs[i]
            for key, value in corr_rank_dict.items():
                log_dict[key] = value
            if epoch > 0:
                for i, loss_name in enumerate(all_loss_names):
                    log_dict[loss_name] = all_epoch_losses[-1][i]
            log_wandb(log_dict)

            # Training step
            for batch in (dataloader):
                h1, h2, x1, x2, label = batch
                
                h1 = F.normalize(h1.float(), dim=1).to(self.device)
                h2 = F.normalize(h2.float(), dim=1).to(self.device)
                phis = self.forward([h1, h2])
                
                z_components = self.decouple([h1,h2], full=True)
                losses_list, loss_names, all_losses, all_loss_names = self.compute_stage_losses(h1, h2, z_components)
                
                losses = torch.stack(losses_list)
                
                optimizer.zero_grad()
                loss, weights = loss_balancer(losses, trainable_params)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                epoch_losses += all_losses
            
            # Update metrics
            epoch_losses = epoch_losses / len(dataloader)
            all_epoch_losses.append(epoch_losses)
            all_epoch_stages.append(self.trainable_stage)
            scheduler.step()
            
            val_loss = self.evaluate_validation_loss(val_dataloader)
            if self.pruning:
                # Prune if in joint stage
                if self.trainable_stage == "joint" and val_loss < self.stage_tracking['best_val_loss']:  # Index 2 is MI loss based on all_loss_names
                    self.prune_singular_values()
                    optimizer = self.update_optimizer(optimizer)
                    if model_hyperparameters.get('lr_annealing') == 'cosine':
                        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
                    elif model_hyperparameters.get('lr_annealing') == 'exponential':
                        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
                    elif model_hyperparameters.get('lr_annealing') == 'linear':
                        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=early_stopping_config[self.trainable_stage]["max_epochs"])
                    elif model_hyperparameters.get('lr_annealing') == 'constant':
                        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=epochs)
            
            if self.staging:
                # Evaluate validation loss
                
                # Update stage tracking
                stage_config = early_stopping_config[self.trainable_stage]
                self.stage_tracking["min_epochs_counter"] += 1
                
                # Calculate improvement
                relative_improvement = (self.stage_tracking["best_val_loss"] - val_loss) / self.stage_tracking["best_val_loss"]

                if val_loss < self.stage_tracking["best_val_loss"]:
                    self.stage_tracking["best_val_loss"] = val_loss
                
                # Update tracking metrics
                if self.switch_criterion == "improvement":
                    if relative_improvement > stage_config["min_improvement_ratio"]:
                        #self.stage_tracking["best_val_loss"] = val_loss
                        self.stage_tracking["plateau_counter"] = 0
                    else:
                        self.stage_tracking["plateau_counter"] += 1
                elif self.switch_criterion == "val_loss":
                    if val_loss < self.stage_tracking["best_val_loss"]:
                        self.stage_tracking["best_val_loss"] = val_loss
                        self.stage_tracking["plateau_counter"] = 0
                    else:
                        self.stage_tracking["plateau_counter"] += 1
                
                # Check for stage transition
                should_switch = (
                    self.stage_tracking["plateau_counter"] >= stage_config["patience"] or
                    self.stage_tracking["min_epochs_counter"] >= stage_config["max_epochs"]
                )
                
                if should_switch:
                    if self.trainable_stage == "shared":
                        self.trainable_stage = "private"
                        #print(f"***** [Epoch {epoch}] → Switched to PRIVATE stage after {self.stage_tracking['min_epochs_counter']} epochs ***** ")
                    elif self.trainable_stage == "private":
                        self.trainable_stage = "joint"
                        #print(f"***** [Epoch {epoch}] → Switched to JOINT stage after {self.stage_tracking['min_epochs_counter']} epochs ***** ")
                    else:
                        break
                    #print(f"Final {self.trainable_stage} stage loss: {val_loss:.4f}")
                    self.stage_tracking["best_val_loss"] = 5000
                    self.stage_tracking["plateau_counter"] = 0
                    self.stage_tracking["min_epochs_counter"] = 0

                    trainable_params = self.get_trainable_parameters()
                    optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=wd)

                    if model_hyperparameters.get('lr_annealing') == 'cosine':
                        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
                    elif model_hyperparameters.get('lr_annealing') == 'exponential':
                        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
                    elif model_hyperparameters.get('lr_annealing') == 'linear':
                        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=early_stopping_config[self.trainable_stage]["max_epochs"])
                    elif model_hyperparameters.get('lr_annealing') == 'constant':
                        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=epochs)
        
        # add the representation plots, correlation plots, and matrix plots to wandb
        h1 = F.normalize(torch.Tensor(val_dataloader.dataset.h1).float(), dim=1).to(self.device)
        h2 = F.normalize(torch.Tensor(val_dataloader.dataset.h2).float(), dim=1).to(self.device)
        labels = val_dataloader.dataset.labels
        phis = self.forward([h1,h2])
        z_n = self.decouple([h1,h2], full=True, th=self.pruning_threshold)
        plot_representations_wandb(z_n, labels)
        plot_projection_matrices_wandb(self)
        # return all training losses
        all_epoch_losses = np.array(all_epoch_losses)
        train_df = pd.DataFrame(all_epoch_losses, columns=all_loss_names)
        #train_df['lr'] = lrs
        train_df['stage'] = all_epoch_stages
        train_df['epoch'] = np.arange(len(train_df))
        return train_df
    
def plot_representations_wandb(z_n, labels, l_ind=0):
    fig, axs = plt.subplots(2, 2, figsize=(16, 16))    
    titles = [
        ('Modality-specific A', 0, 0),
        ('Shared A', 0, 1),
        ('Modality-specific B', 1, 0),
        ('Shared B', 1, 1)
    ]
    
    for title, i, j in titles:
        ax = axs[i, j]
        ax.set_title(title)     
        data = z_n[i][j].detach().cpu().numpy()
        
        if data.shape[1] > 2:
            # Normal PCA case (2+ dimensions)
            pca = PCA(n_components=2)
            x = pca.fit_transform(data)
            df_plot = pd.DataFrame(x, columns=['x1', 'x2'])
            df_plot['label'] = labels[:, l_ind]
            sns.scatterplot(data=df_plot, x='x1', y='x2', hue='label', ax=ax)
        elif data.shape[1] == 2:
            # If already 2D, just plot directly
            df_plot = pd.DataFrame(data, columns=['x1', 'x2'])
            df_plot['label'] = labels[:, l_ind]
            sns.scatterplot(data=df_plot, x='x1', y='x2', hue='label', ax=ax)
        elif data.shape[1] == 1:
            # Handle 1D case by adding a zero column for visualization
            x = np.hstack([data, np.zeros_like(data)])
            ax.scatter(x[:, 0], x[:, 1], c=labels[:, l_ind])
        else:
            # If no valid features, just write "No Data"
            ax.text(0.5, 0.5, "No Data", horizontalalignment='center', verticalalignment='center')
    # return fig
    wandb.log({"Representation Plots": wandb.Image(fig)})
    plt.close(fig)

def plot_projection_matrices_wandb(model):

    # Get matrices
    Rs = model.R_s.detach().cpu().numpy()
    Rm1 = model.R_m1.detach().cpu().numpy()
    Rm2 = model.R_m2.detach().cpu().numpy()

    # Plot matrices
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    matrices = [Rs, Rm1, Rm2]
    titles = ["Shared Projection (R_s)", "Modality-Specific (R_m1)", "Modality-Specific (R_m2)"]
    
    # Find the maximum absolute value for symmetric color range
    max_abs = max(abs(Rs).max(), abs(Rm1).max(), abs(Rm2).max())
    vmin, vmax = -max_abs, max_abs

    # Plot matrices
    for ax, matrix, title in zip(axs, matrices, titles):
        sns.heatmap(matrix, ax=ax, cmap="RdBu_r", cbar=True, vmin=vmin, vmax=vmax, center=0)
        # Get the correct matrix based on title
        if "Shared" in title:
            sv = torch.linalg.svdvals(model.R_s).detach().cpu().numpy()
        elif "R_m1" in title:
            sv = torch.linalg.svdvals(model.R_m1).detach().cpu().numpy()
        else:
            sv = torch.linalg.svdvals(model.R_m2).detach().cpu().numpy()
        ax.set_title(f"{title}\nSVs: {sv}")
    
    wandb.log({"Projection Matrices": wandb.Image(fig)})
    plt.close(fig)

    # Plot correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
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
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                xticklabels=names, yticklabels=names, ax=ax)
    ax.set_title('Correlation between Projection Matrices')
    
    wandb.log({"Projection Matrix Correlations": wandb.Image(fig)})
    plt.close(fig)

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
        #"batch_size": [64, 128, 256, 512],
        "batch_size": [128],
        "learning_rate": [1e-4],
        #"lr_annealing": ['constant', 'cosine', 'exponential', 'linear'],
        "lr_annealing": ['linear'],
        "weight_decay": [1e-4],
        "n_specific_rank": [4],
        "n_shared_rank": [4],
        "weight_init": ['uniform', 'xavier_uniform', 'kaiming_uniform'],
        #"weight_init": ['uniform'],
        "patience1": [20],
        "patience2": [50],
        "min_improvement_ratio1": [0.005],
        "min_improvement_ratio2": [0.001],
        "max_epochs": [1000],
        #"staging": [True, False],
        #"pruning": [True, False],
        #"z_ortho_loss": [False, True]
        "staging": [True],
        "mi_loss": [False, True],
        "scaling_factor": [1, 10],
        "z_ortho_loss": [False],
        "stopping_criterion": ['improvement', 'val_loss'] #note on sweep1: stopping criterion always improvement, ignore hyperparameter
    }
    n_combinations = np.prod([len(v) for v in hyperparameters.values()])
    print(f"Total combinations: {n_combinations}")
    epochs = 5000

    run_iter = 0
    for bs in hyperparameters["batch_size"]:
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=bs, shuffle=True)
        for lr in hyperparameters["learning_rate"]:
            for wd in hyperparameters["weight_decay"]:
                for weight_init in hyperparameters["weight_init"]:
                    for lr_anneal in hyperparameters["lr_annealing"]:
                        for patience1 in hyperparameters["patience1"]:
                            for patience2 in hyperparameters["patience2"]:
                                for min_improvement_ratio1 in hyperparameters["min_improvement_ratio1"]:
                                    for min_improvement_ratio2 in hyperparameters["min_improvement_ratio2"]:
                                        for max_epochs in hyperparameters["max_epochs"]:
                                            for staging in hyperparameters["staging"]:
                                                for mi_loss in hyperparameters["mi_loss"]:
                                                    for z_ortho_loss in hyperparameters["z_ortho_loss"]:
                                                        for scaling_factor in hyperparameters["scaling_factor"]:
                                                            for switch_criterion in hyperparameters["stopping_criterion"]:

                                                                # Print current hyperparameters
                                                                print(f"Running iteration {run_iter + 1}/{n_combinations}")

                                                                # Create early stopping config
                                                                early_stopping_config = {
                                                                    "shared": {
                                                                        "patience": patience1,
                                                                        "max_epochs": max_epochs,
                                                                        "min_improvement_ratio": min_improvement_ratio1
                                                                    },
                                                                    "private": {
                                                                        "patience": patience1,
                                                                        "max_epochs": max_epochs,
                                                                        "min_improvement_ratio": min_improvement_ratio1
                                                                    },
                                                                    "joint": {
                                                                        "patience": patience2,
                                                                        "max_epochs": max_epochs,
                                                                        "min_improvement_ratio": min_improvement_ratio2
                                                                    }
                                                                }

                                                                model_hyperparameters = {
                                                                    "batch_size": bs,
                                                                    "learning_rate": lr,
                                                                    "weight_decay": wd,
                                                                    "weight_init": weight_init,
                                                                    "lr_annealing": lr_anneal,
                                                                    "n_specific_rank": 4,
                                                                    "n_shared_rank": 4,
                                                                    "patience1": patience1,
                                                                    "patience2": patience2,
                                                                    "min_improvement_ratio1": min_improvement_ratio1,
                                                                    "min_improvement_ratio2": min_improvement_ratio2,
                                                                    "max_epochs": max_epochs,
                                                                    "staging": staging,
                                                                    "z_ortho_loss": z_ortho_loss,
                                                                    "model_number": run_iter,
                                                                    "seed": seed,
                                                                    "scaling_factor": scaling_factor,
                                                                    "mi_loss": mi_loss,
                                                                    "switch_criterion": switch_criterion
                                                                }
                                                                
                                                                setup_wandb(
                                                                    run_name=f"{file_name}_{run_iter}",
                                                                    hyperparams=model_hyperparameters,
                                                                    project_name="multimodal_adapter",
                                                                    entity="vschuster-broad-institute"
                                                                )

                                                                # Initialize model
                                                                projection_model = MultiLoReFT(
                                                                    input_dims=[5,5], 
                                                                    shared_rank=4,
                                                                    specific_rank=4, 
                                                                    staging=staging,
                                                                    r_init=weight_init,
                                                                    zm_orhto_loss=z_ortho_loss,
                                                                    scaling_factor=scaling_factor,
                                                                    mi_loss=mi_loss,
                                                                    switch_criterion=switch_criterion,
                                                                    device=device).to(device)
                                                                
                                                                # Train model
                                                                train_df = projection_model.train_projection(dataloader, val_dataloader, early_stopping_config, epochs=epochs, model_hyperparameters=model_hyperparameters)
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
    parser.add_argument("--file_name", type=str, default="sim_sweep_sparse",
                      help="Base name for output files (default: 'sweep')")
    args = parser.parse_args()
    main(dev_id=args.gpu, seed=args.seed, out_dir=args.out_dir, file_name=args.file_name)