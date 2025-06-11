import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizer, AutoTokenizer
from src.losses import loss_shared_consistency, loss_orthogonality, loss_mutual_info, GradientNormalizedLoss
from src.visualization import plot_losses
from src.utils import custom_weight_init, log_wandb
from src.eval_metrics import calc_corrs_and_ranks, evaluate_validation_loss, eval_model, reeval_model

class MultiLoReFT(nn.Module):
    """LoReFT module for multimodal projection learning."""
    def __init__(
            self, 
            input_dims, 
            shared_rank, 
            specific_rank, 
            staging=True, 
            encoders=None, 
            pruning_threshold=0.05, 
            pruning=True, 
            r_init="uniform",
            device=None, 
            dataset_name="simulated",
            verbose=True,
            wandb_log=False
        ):
        super(MultiLoReFT, self).__init__()
        self.shared_rank = shared_rank
        self.specific_rank = specific_rank
        self.pruning_threshold = pruning_threshold
        self.pruned = False
        self.dataset_name = dataset_name
        self.encoders = encoders
        if encoders is not None:
            for i in range(len(encoders)):
                self.encoders[i] = encoders[i].to(device)
                self.encoders[i].eval()
        self.staging = staging
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
        self.verbose = verbose
        self.wandb_log = wandb_log
        
        # Initialize projection matrices
        self.R_s = nn.Parameter(torch.empty(shared_rank, max(input_dims[0], input_dims[1]), dtype=torch.float32))
        self.R_m1 = nn.Parameter(torch.empty(specific_rank, input_dims[0], dtype=torch.float32))
        self.R_m2 = nn.Parameter(torch.empty(specific_rank, input_dims[1], dtype=torch.float32))
        # Initialize weights
        self._orthogonal_init(r_init)
        # Create weight networks for each modality
        self._create_weight_networks(input_dims, r_init)
    
    def _create_weight_networks(self, input_dims, r_init):
        """Create weight networks for each modality."""
        # Modality 1 weights
        self.W_s0 = nn.Sequential(
            nn.Linear(input_dims[0], input_dims[0] * 2, dtype=torch.float32),
            nn.ReLU(),
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
            nn.Linear(input_dims[1] * 2, self.shared_rank, dtype=torch.float32)
        )       
        self.W_m1 = nn.Sequential(
            nn.Linear(input_dims[1], input_dims[1] * 2, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(input_dims[1] * 2, self.specific_rank, dtype=torch.float32)
        )

        # custom weight initialization
        custom_weight_init(self.W_s0, init_option=r_init)
        custom_weight_init(self.W_m0, init_option=r_init)
        custom_weight_init(self.W_s1, init_option=r_init)
        custom_weight_init(self.W_m1, init_option=r_init)
    
    #def _orthogonal_init(self, r_init):
    #    """Initialize projection matrices with uniform distribution."""
    #    custom_weight_init(self.R_s, init_option=r_init)
    #    custom_weight_init(self.R_m1, init_option=r_init)
    #    custom_weight_init(self.R_m2, init_option=r_init)
    
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
    
    def prune_singular_values(self):
        """Prune singular values below threshold and update network weights."""
        def prune_matrix(name, R, weights_to_prune):
            U, S, V = torch.svd(R)
            if len(S) < 2:
                return R, len(S)
            
            # Original code that removes one at a time
            min_sv_idx = torch.argmin(S)
            min_sv = S[min_sv_idx]
            if min_sv > self.pruning_threshold:
                return R, len(S)
            
            # Create mask for keeping dimensions
            keep_indices = torch.ones(R.shape[0], dtype=torch.bool)
            keep_indices[:len(S)][min_sv_idx] = False
            reduced_R = R[keep_indices, :]

            # New code that removes all dimensions below threshold at once
            # keep_indices = torch.ones(R.shape[0], dtype=torch.bool)
            # keep_indices[:len(S)] = S >= self.pruning_threshold
            
            # if torch.all(keep_indices[:len(S)]):
            #     return R, len(S)
                
            # reduced_R = R[keep_indices, :]
            
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
        
        # print(f"Pruned dimensions: Shared kept {kept_s}, Modality1 kept {kept_m1}, Modality2 kept {kept_m2}")
         
    def update_optimizer(self, optimizer):
        optimizer.param_groups[0].update({"params": self.get_trainable_parameters()})
        return optimizer   
    # def update_optimizer(self, optimizer):
    #     """Update optimizer after pruning."""
    #     return torch.optim.Adam(self.parameters(), lr=optimizer.param_groups[0]["lr"])
    
    # def forward(self, embeddings):
    #     """Performing the representation fine-tuning."""
    #     h1, h2 = embeddings[0], embeddings[1]
    #     # Compute projections
    #     h1_out = h1 + (self.W_s0(h1) - F.linear(h1, self.R_s.T)) @ self.R_s + (self.W_m0(h1) - F.linear(h1, self.R_m1.T)) @ self.R_m1
    #     h2_out = h2 + (self.W_s1(h2) - F.linear(h2, self.R_s.T)) @ self.R_s + (self.W_m1(h2) - F.linear(h2, self.R_m2.T)) @ self.R_m2
    #     # h1_out = h1 + torch.matmul((self.W_s0(h1) - torch.matmul(h1, self.R_s.T)), self.R_s) + \
    #     #          torch.matmul((self.W_m0(h1) - torch.matmul(h1, self.R_m1.T)), self.R_m1)
    #     # h2_out = h2 + torch.matmul((self.W_s1(h2) - torch.matmul(h2, self.R_s.T)), self.R_s) + \
    #     #          torch.matmul((self.W_m1(h2) - torch.matmul(h2, self.R_m2.T)), self.R_m2)
    #     return h1_out, h2_out
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
        
        # Decouple representations
        for i, phi in enumerate(phis):
            if full:
                # zs = torch.matmul(phi, self.R_s.T)
                # zm = torch.matmul(phi, (self.R_m1.T if i==0 else self.R_m2.T))
                zs = F.linear(phi, self.R_s)
                zm = F.linear(phi, (self.R_m1 if i==0 else self.R_m2))
            else:
                zs = torch.matmul(phi, self.R_s[shared_sv].T)
                zm = torch.matmul(phi, (self.R_m1[m1_sv].T if i==0 else self.R_m2[m2_sv].T))
            rep_components.append((zm, zs))
        
        return rep_components

    def compute_stage_losses(self, h1, h2, z_components):
        # Compute all losses
        l_shared = loss_shared_consistency(z_components[0][1], z_components[1][1])
        l_orthogonal = loss_orthogonality(self.R_s, self.R_m1, self.R_m2)
        l_mi = loss_mutual_info(h1, h2, z_components)
        
        all_losses = [l_shared.item(), l_orthogonal.item(), l_mi.item()]
        all_loss_names = ["Shared Loss", "Orthogonal Loss", "Mutual Info Loss"]
        
        # Return appropriate losses based on stage
        if self.trainable_stage == "shared":
            return [l_shared, l_mi], ["Shared Loss", "Mutual Info Loss"], all_losses, all_loss_names
        elif self.trainable_stage == "private":
            return [l_orthogonal, l_mi, l_shared], ["Orthogonal Loss", "Mutual Info Loss", "Shared Loss"], all_losses, all_loss_names
        elif self.trainable_stage == "joint":
            return [l_orthogonal, l_shared, l_mi], ["Orthogonal Loss", "Shared Loss", "Mutual Info Loss"], all_losses, all_loss_names
        else:
            raise ValueError(f"Unknown training stage: {self.trainable_stage}")

    def evaluate_validation_loss(self, val_dataloader):
        """Evaluate model on validation set."""
        val_total_loss = 0
        val_loss_list = [0]*3
        loss_balancer = GradientNormalizedLoss(num_losses=3)
        self.eval()
        with torch.no_grad():
            for val_batch in val_dataloader:
                if self.encoders is not None:
                    x1, x2, label = val_batch
                    label = label.to(self.device)
                    x1 = (x1).to(self.device)
                    x2 = (x2).to(self.device)
                    # This needs to be customized
                    tokens_en = BertTokenizer.from_pretrained('bert-base-uncased')(x2, return_tensors="pt", padding=True, truncation=True).to(self.device)
                    tokens_fr = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")(x2, padding=True, truncation=True, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        model_output = self.encoders[2](**tokens_fr)
                        embeddings_fr = model_output.last_hidden_state[:, 0, :].to(self.device)
                        model_output = self.encoders[1](**tokens_en)
                        embeddings_en = model_output.last_hidden_state[:, 0, :].to(self.device)
                        h1 = self.encoders[0].forward_features(x1)[:, 0, :].to(self.device)
                        h2 = torch.where(label.unsqueeze(1).expand(-1, embeddings_en.size(1)) == 0, embeddings_en, embeddings_fr)
                else:
                    h1, h2, x1, x2, label = val_batch
                    if self.dataset_name == "flickr":
                        # Generate random binary labels for each item in batch
                        lang_idx = torch.randint(0, 2, (len(h1),), device=self.device)
                        # Use the labels to select language for each item
                        h2 = torch.stack([h2[0], h2[1]], dim=1).gather(1, lang_idx.unsqueeze(1).unsqueeze(2).expand(-1, -1, h2[0].shape[-1])).squeeze(1)
                        x2 = [x2[0][i] if idx == 0 else x2[1][i] for i, idx in enumerate(lang_idx)]
                        label = lang_idx
                
                h1 = F.normalize(h1.float(), dim=1).to(self.device)
                h2 = F.normalize(h2.float(), dim=1).to(self.device)
                phis = self.forward([h1, h2])
                
                z_components = self.decouple(phis, full=True)
                losses_list, _, all_losses_list, _ = self.compute_stage_losses(h1, h2, z_components)
                val_loss = torch.stack(losses_list).mean()
                val_total_loss += val_loss.item()
                for i, loss in enumerate(all_losses_list):
                    val_loss_list[i] += loss
        self.train() 
        val_loss_list = [loss / len(val_dataloader) for loss in val_loss_list]
        return val_total_loss / len(val_dataloader), val_loss_list
    
    def init_lr_scheduler(self, optimizer, hyperparameters, early_stopping_config, epochs):
        if hyperparameters is None:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
        if hyperparameters.get('lr_schedule') == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        elif hyperparameters.get('lr_schedule') == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        elif hyperparameters.get('lr_schedule') == 'linear':
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=early_stopping_config[self.trainable_stage]["max_epochs"])
        elif hyperparameters.get('lr_schedule') == 'constant':
            scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=epochs)
        else:
            # Default to cosine annealing if no valid schedule is provided
            print("No valid learning rate schedule provided, using CosineAnnealingLR.")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
        return scheduler

    def train_projection(self, dataloader, val_dataloader, early_stopping_config, hyperparameters=None, lr=1e-3, epochs=100, exp_name="projection_module"):#save_path="./ckpts/projection_module.pth"):
        """Train the projection model with early stopping."""
        save_path='./ckpts/%s.pth'%(exp_name)
        if self.verbose:
            print(f"Training on device: {self.device}")
            print(f"Model is on device: {next(self.parameters()).device}")
        # Initialize loss tracking
        loss_balancer = GradientNormalizedLoss(num_losses=3)
        all_epoch_losses = []
        all_epoch_stages = []
        trainable_params = self.get_trainable_parameters()
        optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=1e-4)
        scheduler = self.init_lr_scheduler(optimizer, hyperparameters, early_stopping_config, epochs)
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            epoch_losses = np.zeros(3)

            if self.wandb_log:
                val_loss, val_logs, val_log_names = evaluate_validation_loss(self, val_dataloader, self.device)
                corr_rank_dict = calc_corrs_and_ranks(self)
                log_dict = {
                    "epoch": epoch,
                    "stage": [0 if self.trainable_stage == "shared" else 1 if self.trainable_stage == "private" else 2][0],
                    "lr": optimizer.param_groups[0]['lr'],
                    "relative_improvement": (self.stage_tracking["best_val_loss"] - val_loss) / self.stage_tracking["best_val_loss"]
                }
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
                if self.encoders is not None:
                    x1, x2, label = batch
                    label = label.to(self.device)
                    x1 = (x1).to(self.device)
                    x2 = (x2).to(self.device)
                    # This needs to be customized
                    tokens_en = BertTokenizer.from_pretrained('bert-base-uncased')(x2, return_tensors="pt", padding=True, truncation=True).to(device)
                    tokens_fr = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")(x2, padding=True, truncation=True, return_tensors="pt").to(device)
                    with torch.no_grad():
                        model_output = self.encoders[2](**tokens_fr)
                        embeddings_fr = model_output.last_hidden_state[:, 0, :].to(self.device)
                        model_output = self.encoders[1](**tokens_en)
                        embeddings_en = model_output.last_hidden_state[:, 0, :].to(self.device)
                        h1 = self.encoders[0].forward_features(x1)[:, 0, :].to(self.device)
                        h2 = torch.where(label.unsqueeze(1).expand(-1, embeddings_en.size(1)) == 0, embeddings_en, embeddings_fr)
                else:
                    h1, h2, x1, x2, label = batch
                    if self.dataset_name == "flickr":
                        # Generate random binary labels for each item in batch
                        lang_idx = torch.randint(0, 2, (len(h1),), device=self.device)
                        # Use the labels to select language for each item
                        h2 = torch.stack([h2[0], h2[1]], dim=1).gather(1, lang_idx.unsqueeze(1).unsqueeze(2).expand(-1, -1, h2[0].shape[-1])).squeeze(1)
                        x2 = [x2[0][i] if idx == 0 else x2[1][i] for i, idx in enumerate(lang_idx)]
                        label = lang_idx
                
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
            
            # Update metrics
            epoch_losses = epoch_losses / len(dataloader)
            all_epoch_losses.append(epoch_losses)
            all_epoch_stages.append(self.trainable_stage)
            scheduler.step()
            
            val_loss, val_loss_list = self.evaluate_validation_loss(val_dataloader)
            if self.pruning:
                # Prune if in joint stage
                if self.trainable_stage == "joint" and val_loss_list[2] <= self.stage_tracking['best_val_MI_loss']*1.05 and epoch>40:
                    self.prune_singular_values()
                    optimizer = self.update_optimizer(optimizer)
                    scheduler = self.init_lr_scheduler(optimizer, hyperparameters, early_stopping_config, epochs)
            
            if self.staging:
                stage_config = early_stopping_config[self.trainable_stage]
                self.stage_tracking["min_epochs_counter"] += 1
                
                # Calculate improvement
                relative_improvement = (self.stage_tracking["best_val_loss"] - val_loss) / self.stage_tracking["best_val_loss"]
                
                if val_loss<self.stage_tracking["best_val_loss"]:
                    self.stage_tracking["best_val_loss"] = val_loss
                if val_loss_list[2] < self.stage_tracking["best_val_MI_loss"]:
                    self.stage_tracking["best_val_MI_loss"] = val_loss_list[2]
                
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
                    print(f"Final {self.trainable_stage} stage loss: {val_loss:.4f}")
                    if self.trainable_stage == "shared":
                        self.trainable_stage = "private"
                        self.stage_switches = getattr(self, "stage_switches", [])
                        self.stage_switches.append(('private', epoch))
                        print(f"***** [Epoch {epoch}] → Switched to PRIVATE stage after {self.stage_tracking['min_epochs_counter']} epochs ***** ")
                    elif self.trainable_stage == "private":
                        self.trainable_stage = "joint"
                        self.stage_switches.append(('joint', epoch))
                        print(f"***** [Epoch {epoch}] → Switched to JOINT stage after {self.stage_tracking['min_epochs_counter']} epochs ***** ")
                    # trainable_params = self.get_trainable_parameters()
                    # optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=1e-4)
                    optimizer = self.update_optimizer(optimizer)
                    scheduler = self.init_lr_scheduler(optimizer, hyperparameters, early_stopping_config, epochs)
                    self.stage_tracking["best_val_loss"] = 5000
                    self.stage_tracking["plateau_counter"] = 0
                    self.stage_tracking["min_epochs_counter"] = 0
                elif self.trainable_stage == "joint" and should_switch:
                    if self.verbose:
                        print(f"***** [Epoch {epoch}] → Early stopping after {self.stage_tracking['min_epochs_counter']} epochs ***** ")
                    break
                
                # # Update optimizer
                # trainable_params = model.get_trainable_parameters()
                # optimizer = torch.optim.Adam(trainable_params, lr=1e-4, weight_decay=1e-4)
            else: # ealy stopping!
                # Update stage tracking
                stage_config = early_stopping_config[self.trainable_stage]
                self.stage_tracking["min_epochs_counter"] += 1
                # Calculate improvement
                relative_improvement = (self.stage_tracking["best_val_loss"] - val_loss) / self.stage_tracking["best_val_loss"]
                if val_loss < self.stage_tracking["best_val_loss"]:
                    self.stage_tracking["best_val_loss"] = val_loss

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

                if should_switch:
                    if self.verbose:
                        print(f"***** [Epoch {epoch}] → Early stopping after {self.stage_tracking['min_epochs_counter']} epochs ***** ")
                    break
            
            if self.verbose:
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
        all_epoch_losses = np.array(all_epoch_losses)
        if self.verbose:
            plot_losses(all_epoch_losses, loss_names=all_loss_names, save_path="./plots/%s_loss_curves.pdf"%(exp_name), log_path="./logs/%s_loss_curves.csv"%(exp_name))
        self.save_checkpoint(optimizer, epoch, loss, filepath=save_path)

        if self.wandb_log:
            return all_epoch_losses, all_loss_names, all_epoch_stages


    def save_checkpoint(self, optimizer, epoch, loss, filepath):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict()
        }, filepath)