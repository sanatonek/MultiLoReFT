import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sim_data import generate_multimodal_data
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
    
    def __init__(self, input_dims, shared_rank, specific_rank, data_dim):
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
            nn.Linear(input_dims[0] * 2, self.specific_rank, dtype=torch.float32)
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
            nn.Linear(input_dims[1] * 2, self.specific_rank, dtype=torch.float32)
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
        
        # Compute projections
        h1_out = torch.matmul((self.W_s0(h1) - torch.matmul(h1, self.R_s.T)), self.R_s) + \
                 torch.matmul((self.W_m0(h1) - torch.matmul(h1, self.R_m1.T)), self.R_m1)
        h2_out = torch.matmul((self.W_s1(h2) - torch.matmul(h2, self.R_s.T)), self.R_s) + \
                 torch.matmul((self.W_m1(h2) - torch.matmul(h2, self.R_m2.T)), self.R_m2)
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


def train(model, dataloader, val_dataloader, optimizer, device, scheduler, epochs=100):
    """Train the projection model with early stopping."""
    # Initialize loss tracking
    loss_balancer = GradientNormalizedLoss(num_losses=3)
    all_epoch_losses = []
    
    # Early stopping configuration
    early_stopping_config = {
        "shared": {
            "patience": 15,
            "min_improvement_ratio": 0.01,
            "min_epochs": 50,
            "max_epochs": 200
        },
        "private": {
            "patience": 15,
            "min_improvement_ratio": 0.01,
            "min_epochs": 50,
            "max_epochs": 200
        },
        "joint": {
            "patience": 20,
            "min_improvement_ratio": 0.005,
            "min_epochs": 100,
            "max_epochs": 300
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
    for epoch in range(epochs):
        total_loss = 0
        epoch_losses = np.zeros(3)
        
        # Initialize stage if first epoch
        if epoch == 0:
            model.trainable_stage = "shared"
            trainable_params = model.get_trainable_parameters()
            optimizer = torch.optim.Adam(trainable_params, lr=1e-4, weight_decay=1e-4)
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
            
            print(f"Final {current_stage} stage loss: {val_loss:.4f} (initial: {stage_tracking['initial_loss']:.4f})")
            
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
            optimizer = torch.optim.Adam(trainable_params, lr=1e-4, weight_decay=1e-4)
        
        # Print progress
        if epoch % 10 == 0:
            print(f"[Epoch {epoch}] {current_stage.upper()} stage: "
                  f"val_loss={val_loss:.4f}, "
                  f"best_val_loss={stage_tracking['best_val_loss']:.4f}, "
                  f"relative_improvement={relative_improvement*100:.2f}%, "
                  f"plateau={stage_tracking['plateau_counter']}/{stage_config['patience']}, "
                  f"epochs={stage_tracking['min_epochs_counter']}/{stage_config['max_epochs']}")
        
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
        scheduler.step()
        
        # Prune if in joint stage
        if model.trainable_stage == "joint" and epoch_losses[2] < stage_tracking['best_val_loss']:  # Index 2 is MI loss based on all_loss_names
            model.prune_singular_values()
            optimizer = model.update_optimizer(optimizer)
        
        # Save checkpoint and print progress
        if epoch % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")
            loss_report = ", ".join(f"{name}={val:.4f}" for val, name in zip(epoch_losses, all_loss_names))
            print(f"Loss values: {loss_report}")
            save_checkpoint(model, optimizer, epoch, loss, filepath="./ckpts/projection_module.pth")
    
    # Plot final losses
    all_epoch_losses = np.array(all_epoch_losses)
    plot_losses(all_epoch_losses, loss_names=all_loss_names, save_path="./plots/loss_curves.pdf")
    save_checkpoint(model, optimizer, epoch, loss, filepath="./ckpts/projection_module.pth")


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, filepath)


def main():
    """Main function to run the training pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate and load data
    # h1, h2, x1, x2, labels = generate_multimodal_data(
    #     n_samples=6000, mod_dim=5, save_path="./data/simulated_data.npz")
    loaded_data = np.load("./data/simulated_data.npz")
    h1, h2, x1, x2, labels = loaded_data["h1"], loaded_data["h2"], loaded_data["x1"], loaded_data["x2"], loaded_data["labels"]
    # Create datasets
    dataset = MultimodalDataset(h1[:4000], h2[:4000], x1[:4000], x2[:4000], labels[:4000])
    val_dataset = MultimodalDataset(h1[4000:5000], h2[4000:5000], x1[4000:5000], x2[4000:5000], labels[4000:5000])
    
    # Create dataloaders
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=True)
    
    # Initialize model
    projection_model = ProjectionModule(
        input_dims=[5,5], 
        shared_rank=4, 
        specific_rank=4, 
        data_dim={'A':5, 'B':6}
    ).to(device)
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(projection_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
    
    # Train model
    train(projection_model, dataloader, val_dataloader, optimizer, device, scheduler, epochs=800)


if __name__ == "__main__":
    main()

