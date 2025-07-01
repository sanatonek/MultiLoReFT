import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils import *
from losses import *
import random
import os


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
    def __init__(self, input_dims, shared_rank, specific_rank, staging=True, encoders=None, intervene_layer=-1, 
                pruning_threshold=0.05, pruning=True, device=None, dataset_name="simulated"):
        super(MultiLoReFT, self).__init__()
        self.shared_rank = shared_rank
        self.specific_rank = specific_rank
        self.pruning_threshold = pruning_threshold
        self.pruned = False
        self.encoders = encoders
        self.dataset_name = dataset_name
        self.intervene_layer = intervene_layer
        if encoders is not None:
            for i in range(len(encoders)):
                self.encoders[i] = encoders[i].to(device)
                self.encoders[i].eval()
        self.staging = staging
        if not staging:
            self.stage_switches = [(0, 0)]
        else:
            self.stage_switches = []
        self.pruning = pruning
        self.device = device
        if staging:
            self.trainable_stage = "shared"
        else:
            self.trainable_stage = "joint"
        self.stage_tracking = {
                "best_val_loss": 5000,
                "best_val_MI_loss": 5000,
                "plateau_counter": 0,
                "min_epochs_counter": 0,
            }
        
        # Initialize projection matrices
        self.R_s = nn.Parameter(torch.randn(shared_rank, max(input_dims[0], input_dims[1]), dtype=torch.float32))
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
            nn.Linear(input_dims[0] * 2, self.shared_rank, dtype=torch.float32)
        ) 
        self.W_s0.apply(self._init_weights)  
        self.W_m0 = nn.Sequential(
            nn.Linear(input_dims[0], input_dims[0] * 2, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(input_dims[0] * 2, self.specific_rank, dtype=torch.float32)
        )
        self.W_m0.apply(self._init_weights)
        # Modality 2 weights
        self.W_s1 = nn.Sequential(
            nn.Linear(input_dims[1], input_dims[1] * 2, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(input_dims[1] * 2, self.shared_rank, dtype=torch.float32)
        )  
        self.W_m0.apply(self._init_weights)     
        self.W_m1 = nn.Sequential(
            nn.Linear(input_dims[1], input_dims[1] * 2, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(input_dims[1] * 2, self.specific_rank, dtype=torch.float32)
        )
        self.W_m0.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def _orthogonal_init(self):
        nn.init.orthogonal_(self.R_s, gain=0.1)
        nn.init.orthogonal_(self.R_m1, gain=0.1)
        nn.init.orthogonal_(self.R_m2, gain=0.1)
    
    def get_trainable_parameters(self):
        """Get parameters to train based on current stage."""
        if self.trainable_stage == "shared":
            return [self.R_s] + list(self.W_s0.parameters()) + list(self.W_s1.parameters())
        elif self.trainable_stage == "private":
            return [self.R_m1, self.R_m2] + list(self.W_m0.parameters()) + list(self.W_m1.parameters())
        else:  # joint 
            return list(self.parameters())
    
    def prune_singular_values(self, single=False):
        """Prune singular values below threshold and update network weights."""
        def prune_matrix(name, R, weights_to_prune):
            U, S, V = torch.svd(R)
            if len(S) < 2:
                return R, len(S)
            
            # Original code that removes one at a time
            if single:
                min_sv_idx = torch.argmin(S)
                min_sv = S[min_sv_idx]
                if min_sv > self.pruning_threshold:
                    return R, len(S)
                keep_indices = torch.ones(R.shape[0], dtype=torch.bool)
                keep_indices[:len(S)][min_sv_idx] = False
                reduced_R = R[keep_indices, :]

            else:
                keep_indices = torch.ones(R.shape[0], dtype=torch.bool)
                below_threshold = S < self.pruning_threshold
                num_below = below_threshold.sum().item()
                if num_below == 0:
                    return R, len(S)
                # Calculate number to remove (between 1-10% of matrix size)
                n_remove = max(1, min(num_below, int(0.1 * len(S))))
                # Get indices of n smallest singular values
                smallest_n_idx = torch.argsort(S)[:n_remove]
                keep_indices[:len(S)][smallest_n_idx] = False
                reduced_R = R[keep_indices, :]
            
            # # Update weight networks
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
            print(f">>>>>>>>>>>>>>>>>> Pruned %s to %d dimensions "%(name, len(reduced_R)))
            return getattr(self, name), keep_indices.sum().item()
        
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
        
         
    def update_optimizer(self, optimizer, lr):
        # optimizer.param_groups[0].update({"params": self.get_trainable_parameters()})
        # return optimizer  
        del optimizer
        torch.cuda.empty_cache()
        return torch.optim.Adam(self.get_trainable_parameters(), lr=lr) 

    def forward(self, embeddings):
        h1 = F.normalize(embeddings[0], p=2, dim=-1)
        h2 = F.normalize(embeddings[1], p=2, dim=-1)
        # Shared projections
        proj_s0 = self.W_s0(h1) - F.linear(h1, self.R_s)             # (B, shared_rank)
        shared_h1 = F.linear(proj_s0, self.R_s.T)                    # (B, D)

        proj_s1 = self.W_s1(h2) - F.linear(h2, self.R_s)
        shared_h2 = F.linear(proj_s1, self.R_s.T)

        # Modality-specific projections
        proj_m0 = self.W_m0(h1) - F.linear(h1, self.R_m1)
        spec_h1 = F.linear(proj_m0, self.R_m1.T)

        proj_m1 = self.W_m1(h2) - F.linear(h2, self.R_m2)
        spec_h2 = F.linear(proj_m1, self.R_m2.T)

        # Final representations
        h1_out = h1 + shared_h1 + spec_h1
        h2_out = h2 + shared_h2 + spec_h2
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
        for i, phi in enumerate(phis):
            if full:
                zs = F.linear(phi, self.R_s)
                zm = F.linear(phi, (self.R_m1 if i==0 else self.R_m2))
            else:
                zs = torch.matmul(z, self.R_s[shared_sv].T)
                zm = torch.matmul(z, (self.R_m1[m1_sv].T if i==0 else self.R_m2[m2_sv].T))
            rep_components.append((zm, zs))
        return rep_components

    def fuse_representations(self, phis):
        """Fuse representations."""
        zs1 = F.linear(phis[0], self.R_s)
        zm1 = F.linear(phis[0], self.R_m1)
        zs2 = F.linear(phis[1], self.R_s)
        zm2 = F.linear(phis[1], self.R_m2)
        # Choose zs1 or zs2 based on a binary random sample
        random_zs = zs1 if torch.randint(0, 2, (1,)).item() == 0 else zs2
        return torch.cat((zm1, zm2, random_zs), dim=-1)

    def compute_stage_losses(self, h1, h2, z_components):
        # Compute all losses
        # l_shared = loss_shared_consistency(z_components[0][1], z_components[1][1])
        # l_orthogonal = loss_orthogonality(self.R_s, self.R_m1, self.R_m2)
        l_orthogonal = loss_independence(z_components[0][1], z_components[1][1], z_components[0][0], z_components[1][0])
        l_mi = loss_mutual_info(h1, h2, z_components, all=False if self.trainable_stage == "shared" else True)
        
        all_losses = [l_orthogonal.item(), l_mi.item()]
        all_loss_names = ["Orthogonal Loss", "Mutual Info Loss"]
        
        # Return appropriate losses based on stage
        if self.trainable_stage == "shared":
            return [l_mi], ["Mutual Info Loss"], all_losses, all_loss_names
        elif self.trainable_stage == "private":
            return [l_orthogonal, l_mi], ["Orthogonal Loss",  "Mutual Info Loss"], all_losses, all_loss_names
        elif self.trainable_stage == "joint":
            return [l_orthogonal, l_mi], ["Orthogonal Loss", "Mutual Info Loss"], all_losses, all_loss_names
        else:
            raise ValueError(f"Unknown training stage: {stage}")

    def evaluate_validation_loss(self, val_dataloader, **kwargs):
        """Evaluate model on validation set."""
        val_total_loss = 0
        val_loss_list = np.zeros(2)  # Initialize list to store losses
        self.eval()
        loss_balancer = GradientNormalizedLoss(num_losses=3)
        with torch.no_grad():
            for val_batch in val_dataloader:
                if self.encoders is not None:
                    x1, x2, label = val_batch
                    label = label.to(self.device)
                    x1 = (x1).to(self.device)
                    x2 = (x2).to(self.device)
                    # This needs to be customized
                    tokens_en = en_tokenizer(x2, return_tensors="pt", padding=True, truncation=True).to(device)
                    tokens_fr = fr_tokenizer(x2, padding=True, truncation=True, return_tensors="pt").to(device)
                    model_output = self.encoders[2](**tokens_fr)
                    embeddings_fr = model_output.last_hidden_state[:, 0, :].to(self.device)
                    model_output = self.encoders[1](**tokens_en)
                    embeddings_en = model_output.last_hidden_state[:, 0, :].to(self.device)
                    h1 = self.encoders[0].forward_features(x1)[:, 0, :].to(self.device)
                    h2 = torch.where(label.unsqueeze(1).expand(-1, embeddings_en.size(1)) == 0, embeddings_en, embeddings_fr)
                else:
                    if self.dataset_name == "flickr":
                        # Generate random binary labels for each item in batch
                        h1 = val_batch[0]
                        h2 = val_batch[1]
                        lang_idx = torch.randint(0, 2, (len(h1),), device=h1[0].device)
                        # Use the labels to select language for each item
                        h2 = torch.stack([h2[0], h2[1]], dim=1).gather(1, lang_idx.unsqueeze(1).unsqueeze(2).expand(-1, -1, h2[0].shape[-1])).squeeze(1)
                        # x2 = [x2[0][i] if idx == 0 else x2[1][i] for i, idx in enumerate(lang_idx)]
                        # label = lang_idx
                    else:
                        h1 = val_batch[0]
                        h2 = val_batch[1]
                
                h1 = F.normalize(h1.float(), dim=1).to(self.device)
                h2 = F.normalize(h2.float(), dim=1).to(self.device)
                # phis = self.forward([h1, h2])
                
                # z_components = self.decouple(phis, full=True)
                z_components = self.decouple([h1, h2], full=True)
                losses_list, _, all_losses_list, _ = self.compute_stage_losses(h1, h2, z_components)
                val_loss = torch.stack(losses_list).mean()
                # val_loss, weights = loss_balancer(torch.stack(losses_list), self.get_trainable_parameters())

                val_total_loss += val_loss.item()
                # Return both total loss and average loss list
                val_loss_list += all_losses_list
                torch.cuda.empty_cache()
        if self.encoders is not None:
            del model_output, embeddings_en, embeddings_fr
        self.train()
        
        # Average the losses over batches
        val_loss_list = [loss / len(val_dataloader) for loss in val_loss_list]
        return val_total_loss / len(val_dataloader), val_loss_list


    def train_projection(self, dataloader, val_dataloader, early_stopping_config, lr=1e-3, epochs=100, exp_name="projection_module", **kwargs):
        """Train the projection model with early stopping."""
        # Create checkpoints directory if it doesn't exist
        os.makedirs("./ckpts", exist_ok=True)
        os.makedirs("./plots", exist_ok=True)
        os.makedirs("./logs", exist_ok=True)
        save_path='./ckpts/%s.pth'%(exp_name)
        print(f"Training on device: {self.device}")
        print(f"Model is on device: {next(self.parameters()).device}")
        # Initialize loss tracking
        loss_balancer = GradientNormalizedLoss(num_losses=3)
        all_epoch_losses = []
        trainable_params = self.get_trainable_parameters()
        optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            epoch_losses = np.zeros(2)
            # Training step
            for batch in (dataloader):
                if self.encoders is not None:
                    x1, x2, label = batch
                    label = label.to(self.device)
                    x1 = (x1).to(self.device)
                    x2 = (x2).to(self.device)
                    # This needs to be customized
                    with torch.no_grad():
                        tokens_en = en_tokenizer(x2, return_tensors="pt", padding=True, truncation=True).to(self.device)
                        tokens_fr = fr_tokenizer(x2, padding=True, truncation=True, return_tensors="pt").to(self.device)
                        model_output = self.encoders[2](**tokens_fr)
                        embeddings_fr = model_output.last_hidden_state[:, 0, :].to(self.device)
                        model_output = self.encoders[1](**tokens_en)
                        embeddings_en = model_output.last_hidden_state[:, 0, :].to(self.device)
                        h1 = self.encoders[0].forward_features(x1)[:, 0, :].to(self.device)
                        h2 = torch.where(label.unsqueeze(1).expand(-1, embeddings_en.size(1)) == 0, embeddings_en, embeddings_fr)
                else:
                    if self.dataset_name == "flickr":
                        # Generate random binary labels for each item in batch
                        h1 = batch[0]
                        h2 = batch[1]
                        lang_idx = torch.randint(0, 2, (len(h1),)).to(h1[0].device)
                        # Use the labels to select language for each item
                        h2 = torch.stack([h2[0], h2[1]], dim=1).gather(1, lang_idx.unsqueeze(1).unsqueeze(2).expand(-1, -1, h2[0].shape[-1])).squeeze(1)
                        # x2 = [x2[0][i] if idx == 0 else x2[1][i] for i, idx in enumerate(lang_idx)]
                        # label = lang_idx
                    else:
                        h1 = batch[0]
                        h2 = batch[1]
                
                h1 = F.normalize(h1.float(), dim=1).to(self.device)
                h2 = F.normalize(h2.float(), dim=1).to(self.device)
                phis = self.forward([h1, h2])
                
                z_components = self.decouple(phis, full=True)
                losses_list, loss_names, all_losses, all_loss_names = self.compute_stage_losses(h1, h2, z_components)
                
                losses = torch.stack(losses_list)
                
                optimizer.zero_grad()
                loss, weights = loss_balancer(losses, trainable_params)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                epoch_losses += all_losses
                del  h1, h2, phis, z_components
                torch.cuda.empty_cache()
            
            # Update metrics
            epoch_losses = epoch_losses / len(dataloader)
            all_epoch_losses.append(epoch_losses)
            scheduler.step()
            
            val_loss, val_loss_list = self.evaluate_validation_loss(val_dataloader, **kwargs)
            if self.pruning:
                # Prune if in joint stage
                if self.trainable_stage == "joint" and (val_loss_list[-1] <= self.stage_tracking['best_val_MI_loss'] * 1.05) and epoch>self.stage_switches[-1][-1]+10 and epoch>100:  # Index 2 is MI loss based on all_loss_names, allow 5% margin
                    self.prune_singular_values()
                    optimizer = self.update_optimizer(optimizer, lr=lr)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
            
            
            if self.staging:
                stage_config = early_stopping_config[self.trainable_stage]
                self.stage_tracking["min_epochs_counter"] += 1
                
                # Calculate improvement
                relative_improvement = (self.stage_tracking["best_val_loss"] - val_loss) / self.stage_tracking["best_val_loss"]
                
                if val_loss<self.stage_tracking["best_val_loss"]:
                    self.stage_tracking["best_val_loss"] = val_loss
                if val_loss_list[-1]<self.stage_tracking["best_val_MI_loss"]:
                    self.stage_tracking["best_val_MI_loss"] = val_loss_list[-1]
                
                # Update tracking metrics
                if relative_improvement > stage_config["min_improvement_ratio"]:
                    self.stage_tracking["plateau_counter"] = 0
                else:
                    self.stage_tracking["plateau_counter"] += 1
                
                # Check for stage transition
                should_switch = (
                    self.stage_tracking["plateau_counter"] >= stage_config["patience"] or
                    self.stage_tracking["min_epochs_counter"] >= stage_config["max_epochs"]
                )
                
                if self.trainable_stage != "joint" and self.staging and should_switch:
                    if self.trainable_stage == "shared":
                        self.trainable_stage = "private"
                        self.stage_switches = getattr(self, 'stage_switches', [])
                        self.stage_switches.append(('private', epoch))
                        print(f"***** [Epoch {epoch}] → Switched to PRIVATE stage after {self.stage_tracking['min_epochs_counter']} epochs ***** ")
                    elif self.trainable_stage == "private":
                        self.trainable_stage = "joint"
                        self.stage_tracking["best_val_MI_loss"] = 5000
                        self.stage_switches = getattr(self, 'stage_switches', [])
                        self.stage_switches.append(('joint', epoch))
                        print(f"***** [Epoch {epoch}] → Switched to JOINT stage after {self.stage_tracking['min_epochs_counter']} epochs ***** ")
                    print(f"Final {self.trainable_stage} stage loss: {val_loss:.4f}")
                    # trainable_params = self.get_trainable_parameters()
                    # optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=1e-4)
                    optimizer = self.update_optimizer(optimizer, lr=lr)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
                    self.stage_tracking["best_val_loss"] = 5000
                    self.stage_tracking["plateau_counter"] = 0
                    self.stage_tracking["min_epochs_counter"] = 0
                
                # # Update optimizer
                # trainable_params = model.get_trainable_parameters()
                # optimizer = torch.optim.Adam(trainable_params, lr=1e-4, weight_decay=1e-4)
            
            # Print progress
            if epoch % 1 == 0:
                print(f"[Epoch {epoch}] {self.trainable_stage.upper()} stage: "
                    f"val_loss={val_loss:.4f}, "
                    f"best_val_loss={self.stage_tracking['best_val_loss']:.4f}, ")
                loss_report = ", ".join(f"{name}={val:.4f}" for val, name in zip(epoch_losses, all_loss_names))
                print(f"Loss values: {loss_report}")
                if self.staging:
                    print(f"relative_improvement={relative_improvement*100:.2f}%, "
                    f"plateau={self.stage_tracking['plateau_counter']}/{stage_config['patience']}, "
                    f"epochs={self.stage_tracking['min_epochs_counter']}/{stage_config['max_epochs']}")
            self.save_checkpoint(optimizer, epoch, loss, filepath=save_path)
        
            # Plot final losses
            plot_losses(np.array(all_epoch_losses), loss_names=all_loss_names, save_path="./plots/%s_loss_curves.pdf"%(exp_name), log_path="./logs/%s_loss_curves.csv"%(exp_name), stage_switches=self.stage_switches)
        self.save_checkpoint(optimizer, epoch, loss, filepath=save_path)


    def save_checkpoint(self, optimizer, epoch, loss, filepath):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict()
        }, filepath)


def main():
    """Main function to run the training pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from sim_data import generate_multimodal_data
    
    # Generate and load data
    try:
        loaded_data = np.load("./data/simplest_sim_nongaussian.npz")
    except FileNotFoundError:
        # Create data directory if it doesn't exist
        os.makedirs("./data", exist_ok=True)
        h1, h2, x1, x2, labels = generate_multimodal_data(
            n_samples=6000, mod_dim=10, save_path="./data/simulated_data.npz")
        loaded_data = np.load("./data/simulated_data.npz")
    h1, h2, x1, x2, labels = loaded_data["h1"], loaded_data["h2"], loaded_data["x1"], loaded_data["x2"], loaded_data["labels"]
    # Create datasets
    dataset = MultimodalDataset(h1[:4000], h2[:4000], x1[:4000], x2[:4000], labels[:4000])
    val_dataset = MultimodalDataset(h1[4000:5000], h2[4000:5000], x1[4000:5000], x2[4000:5000], labels[4000:5000])
    
    # Create dataloaders
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=512, shuffle=True)
    
        # Early stopping configuration
    early_stopping_config = {
        "shared": {
            "patience": 200,
            "min_improvement_ratio": 0.001,
            # "min_epochs": 50,
            "max_epochs": 400
        },
        "private": {
            "patience": 100,
            "min_improvement_ratio": 0.001,
            # "min_epochs": 50,
            "max_epochs": 400
        },
        "joint": {
            "patience": 100,
            "min_improvement_ratio": 0.001,
            # "min_epochs": 100,
            "max_epochs": 300
        }
    }

    # Initialize model
    projection_model = MultiLoReFT(
        input_dims=[10,10], 
        shared_rank=20, 
        specific_rank=10, 
        pruning_threshold=0.2,
        staging=True,
        pruning=True,
        device=device
    ).to(device)
    

    # Train model
    projection_model.train_projection(dataloader, val_dataloader, early_stopping_config, lr=1e-3, epochs=1000)


if __name__ == "__main__":
    main()

