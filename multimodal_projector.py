import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils import *
from losses import *
import random
import os
from collections import defaultdict

class MultiLoReFT(nn.Module):
    """LoReFT module for multimodal projection learning."""
    def __init__(self, input_dims, shared_rank, specific_rank, staging=True, encoders=None, intervene_layer=-1, 
                shared_R_mode="double", pruning_threshold=0.05, pruning=True, device=None, dataset_name="simulated"):
        super(MultiLoReFT, self).__init__()
        self.shared_rank = shared_rank
        self.specific_rank = specific_rank
        self.pruning_threshold = pruning_threshold
        self.pruned = False
        self.encoders = encoders
        self.dataset_name = dataset_name
        self.intervene_layer = intervene_layer
        self.shared_R_mode = shared_R_mode
        self.input_dims = input_dims
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
        self.max_dim = max(input_dims)
        
        # Initialize projection matrices
        if shared_R_mode == "double":
            self.R_s1 = nn.Parameter(torch.randn(shared_rank, input_dims[0], dtype=torch.float32))
            self.R_s2 = nn.Parameter(torch.randn(shared_rank, input_dims[1], dtype=torch.float32))
            self.W_s0 = self._create_weight_networks(input_dims[0], self.shared_rank)
            self.W_s1 = self._create_weight_networks(input_dims[1], self.shared_rank)
            self.W_s0.apply(self._init_weights)
            self.W_s1.apply(self._init_weights)
            self.R_m1 = nn.Parameter(torch.randn(specific_rank, input_dims[0], dtype=torch.float32))
            self.R_m2 = nn.Parameter(torch.randn(specific_rank, input_dims[1], dtype=torch.float32))
            self.W_m0 = self._create_weight_networks(input_dims[0], self.specific_rank)
            self.W_m0.apply(self._init_weights)
            self.W_m1 = self._create_weight_networks(input_dims[1], self.specific_rank)
            self.W_m1.apply(self._init_weights)
        elif shared_R_mode == "pad":
            self.R_s1 = nn.Parameter(torch.randn(shared_rank, self.max_dim, dtype=torch.float32))
            self.W_s0 = self._create_weight_networks(self.max_dim, self.shared_rank)
            self.W_s1 = self._create_weight_networks(self.max_dim, self.shared_rank)
            self.W_s0.apply(self._init_weights)
            self.W_s1.apply(self._init_weights)
            self.R_m1 = nn.Parameter(torch.randn(specific_rank, self.max_dim, dtype=torch.float32))
            self.R_m2 = nn.Parameter(torch.randn(specific_rank, self.max_dim, dtype=torch.float32))
            self.W_m0 = self._create_weight_networks(self.max_dim, self.specific_rank)
            self.W_m0.apply(self._init_weights)
            self.W_m1 = self._create_weight_networks(self.max_dim, self.specific_rank)
            self.W_m1.apply(self._init_weights)
            # Initialize weights
        self._orthogonal_init()
            # Create weight networks for each modality
    
    def _create_weight_networks(self, input_dim, output_dim):
        """Create weight networks for each modality."""
        model = nn.Sequential(
                    nn.Linear(input_dim, 20, dtype=torch.float32),
                    nn.ReLU(),
                    nn.Linear(20, output_dim, dtype=torch.float32)
                )
        return model
        # # Modality 1 weights  
        # self.W_m0 = nn.Sequential(
        #     nn.Linear(input_dims[0], 20, dtype=torch.float32),
        #     nn.ReLU(),
        #     nn.Linear(20, self.specific_rank, dtype=torch.float32)
        # )
        # self.W_m0.apply(self._init_weights)
        # # Modality 2 weights
        # self.W_m1 = nn.Sequential(
        #     nn.Linear(input_dims[1], 20, dtype=torch.float32),
        #     nn.ReLU(),
        #     nn.Linear(20, self.specific_rank, dtype=torch.float32)
        # )
        # self.W_m1.apply(self._init_weights)

        # if self.shared_R_mode == "double":
        #     self.W_s1 = nn.Sequential(
        #         nn.Linear(input_dims[1], 20, dtype=torch.float32),
        #         nn.ReLU(),
        #         nn.Linear(20, self.shared_rank, dtype=torch.float32)
        #     )  
        #     self.W_s1.apply(self._init_weights) 
        #     self.W_s0 = nn.Sequential(
        #         nn.Linear(input_dims[0], 20, dtype=torch.float32),
        #         nn.ReLU(),
        #         nn.Linear(20, self.shared_rank, dtype=torch.float32)
        #     ) 
        #     self.W_s0.apply(self._init_weights)  
        # else:
        #     self.W_s0 = nn.Sequential(
        #         nn.Linear(self.max_dim, 20, dtype=torch.float32),
        #         nn.ReLU(),
        #         nn.Linear(20, self.shared_rank, dtype=torch.float32)
        #     ) 
        #     self.W_s0.apply(self._init_weights)  
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def _orthogonal_init(self):
        nn.init.orthogonal_(self.R_s1, gain=1)
        nn.init.orthogonal_(self.R_m1, gain=1)
        nn.init.orthogonal_(self.R_m2, gain=1)
        if self.shared_R_mode == "double":
            nn.init.orthogonal_(self.R_s2, gain=1)
        # Using Xavier uniform initialization instead
        # nn.init.xavier_uniform_(self.R_s1, gain=0.1)
        # nn.init.xavier_uniform_(self.R_s2, gain=0.1)
        # nn.init.xavier_uniform_(self.R_m1, gain=0.1)
        # nn.init.xavier_uniform_(self.R_m2, gain=0.1)
    
    def get_trainable_parameters(self):
        """Get parameters to train based on current stage."""
        if self.trainable_stage == "shared":
            return [self.R_s1, self.R_s2] + list(self.W_s0.parameters()) + list(self.W_s1.parameters()) if self.shared_R_mode == "double" else [self.R_s1] + list(self.W_s0.parameters()) + list(self.W_s1.parameters())
        elif self.trainable_stage == "private":
            return [self.R_m1, self.R_m2] + list(self.W_m0.parameters()) + list(self.W_m1.parameters())
        elif self.trainable_stage == "joint": 
            return [self.R_m1, self.R_m2, self.R_s1, self.R_s2] + list(self.W_m0.parameters()) + list(self.W_m1.parameters()) + list(self.W_s0.parameters()) + list(self.W_s1.parameters()) if self.shared_R_mode == "double" else [self.R_m1, self.R_m2, self.R_s1] + list(self.W_m0.parameters()) + list(self.W_m1.parameters()) + list(self.W_s0.parameters()) + list(self.W_s1.parameters())
    
    def prune_singular_values(self, single=False):
        """Prune singular values below threshold and update network weights."""
        def prune_matrix(name, R, weights_to_prune):
            if R.shape[0] < 3:
                return R, R.shape[0], False
            U, S, Vh = torch.linalg.svd(R, full_matrices=False)  # Vh = V^T
            if single:
                min_sv_idx = torch.argmin(S)
                min_sv = S[min_sv_idx]
                if min_sv > self.pruning_threshold:
                    return R, len(S), False
                n_remove = 1

            else:
                keep_indices = torch.ones(R.shape[0], dtype=torch.bool)
                below_threshold = S < self.pruning_threshold
                num_below = below_threshold.sum().item()
                print(f"Number of singular values below threshold: {num_below}, {S}")
                if num_below == 0:
                    return R, len(S), False
                # Calculate number to remove (between 1-10% of matrix size)
                n_remove = max(1, min(num_below, int(0.1 * len(S))))
            k = R.shape[0] - n_remove  # or choose by threshold
            reduced_R = (S[:k].unsqueeze(1) * Vh[:k, :])        # diag(S_n) @ Vh_n
            reduced_R = reduced_R.to(device=self.device, dtype=torch.float32)
            # reduced_R = (U[:, :k] * S[:k]) @ Vh[:k, :] 
            print(f"Reduced {name} to {k} dimensions", R.shape, reduced_R.shape)

            UkT = U[:, :k].T
            # # Update weight networks
            for i, weight_seq in enumerate(weights_to_prune):
                last_layer = weight_seq[-1]
                assert isinstance(last_layer, nn.Linear), "Expected last layer to be nn.Linear"
                in_features  = last_layer.in_features    # keep same
                out_old      = last_layer.out_features   # D_old

                device = last_layer.weight.device
                dtype  = last_layer.weight.dtype

                # Build new last layer with out_features = k
                new_last = nn.Linear(in_features, k, bias=True, device=device, dtype=dtype)

                with torch.no_grad():
                    # Rotate rows by Uk^T so that z_new = Uk^T z_old
                    # old W maps h -> z_old in R^{D_old}; we want h -> z_new in R^{k}
                    new_last.weight.copy_(UkT @ last_layer.weight.data)  # (in_features, k)
                    if last_layer.bias is not None:
                        new_last.bias.copy_(UkT @ last_layer.bias.data)  # (k,)
                    else:
                        nn.init.zeros_(new_last.bias)
                weight_seq[-1] = new_last
            UkT = U[:, :k].T            # (9, 10)
            # Update parameter
            # del self._parameters[name]
            # self.register_parameter(name, nn.Parameter(reduced_R))
            print(f">>>>>>>>>>>>>>>>>> Pruned %s to %d dimensions "%(name, len(reduced_R)), reduced_R.shape)
            self.stage_tracking["plateau_counter"] = 0
            return reduced_R, k, True
        
        # Prune each matrix
        kept_s1, kept_s2, kept_m1, kept_m2 = 0, 0, 0, 0
        if self.shared_R_mode == "double":
            pruned_R, kept_s1, is_pruned = prune_matrix("R_s1", self.R_s1, [self.W_s0])
            if is_pruned:
                self.shared_rank = kept_s1
                self.R_s1 = torch.nn.Parameter(pruned_R)
                self.optimizer.param_groups[0]['params'] =  [self.R_s1, self.R_s2] + list(self.W_s0.parameters()) + list(self.W_s1.parameters())
            pruned_R, kept_s2, is_pruned = prune_matrix("R_s2", self.R_s2, [self.W_s1])
            if is_pruned:
                self.shared_rank = kept_s2
                self.R_s2 = torch.nn.Parameter(pruned_R)
                self.optimizer.param_groups[0]['params'] =  [self.R_s1, self.R_s2] + list(self.W_s0.parameters()) + list(self.W_s1.parameters())
                
        else:
            pruned_R, kept_s1, is_pruned = prune_matrix("R_s1", self.R_s1, [self.W_s0, self.W_s1])
            if is_pruned:
                self.shared_rank = kept_s1
                self.R_s1 = torch.nn.Parameter(pruned_R)
                self.optimizer.param_groups[0]['params'] = [self.R_s1] + list(self.W_s0.parameters()) + list(self.W_s1.parameters())
        
        pruned_R, kept_m1, is_pruned = prune_matrix("R_m1", self.R_m1, [self.W_m0])
        if is_pruned:
            self.specific_rank = kept_m1
            self.R_m1 = torch.nn.Parameter(pruned_R)
            self.optimizer.param_groups[1]['params'] = [self.R_m1, self.R_m2] + list(self.W_m0.parameters()) + list(self.W_m1.parameters())
        pruned_R, kept_m2, is_pruned = prune_matrix("R_m2", self.R_m2, [self.W_m1])
        if is_pruned:
            self.specific_rank = kept_m2
            self.R_m2 = torch.nn.Parameter(pruned_R)
            self.optimizer.param_groups[1]['params'] = [self.R_m1, self.R_m2] + list(self.W_m0.parameters()) + list(self.W_m1.parameters())
        self.optimizer.state = defaultdict(dict, self.optimizer.state)
         

    def forward(self, embeddings):
        h1 = F.normalize(embeddings[0], p=2, dim=-1)
        h2 = F.normalize(embeddings[1], p=2, dim=-1)

        if self.input_dims[0] != self.input_dims[1] and self.shared_R_mode == "pad":
            h1 = F.pad(h1, (0, self.max_dim - h1.shape[1]))
            h2 = F.pad(h2, (0, self.max_dim - h2.shape[1]))
        # Shared projections
        proj_s0 = self.W_s0(h1) - F.linear(h1, self.R_s1)             # (B, shared_rank)
        shared_h1 = F.linear(proj_s0, self.R_s1.T)              # (B, D)

        proj_s1 = self.W_s1(h2) - F.linear(h2, (self.R_s1 if self.shared_R_mode == "pad" else self.R_s2))
        shared_h2 = F.linear(proj_s1, (self.R_s1.T if self.shared_R_mode == "pad" else self.R_s2.T))

        # Modality-specific projections
        proj_m0 = self.W_m0(h1) - F.linear(h1, self.R_m1)
        spec_h1 = F.linear(proj_m0, self.R_m1.T)

        proj_m1 = self.W_m1(h2) - F.linear(h2, self.R_m2)
        spec_h2 = F.linear(proj_m1, self.R_m2.T)

        # Final representations
        phi1 = h1 + shared_h1 + spec_h1
        phi2 = h2 + shared_h2 + spec_h2
        return phi1, phi2

    def decouple(self, phis, full=True, th=0.1):
        """Separate shared and modality-specific representations."""
        rep_components = []
        # Get singular values
        # s1_values = torch.svd(self.R_s1)[1]
        # s2_values = torch.svd(self.R_s2)[1]
        # s_values = torch.cat((s1_values, s2_values), dim=0)
        # shared_sv = torch.where(s_values > th)[0]
        # m1_values = torch.svd(self.R_m1)[1]
        # m1_sv = torch.where(m1_values > th)[0]
        # m2_values = torch.svd(self.R_m2)[1]
        # m2_sv = torch.where(m2_values > th)[0]
        for i, phi in enumerate(phis):
            # if full:
            zs = F.linear(phi, self.R_s1 if i==0 else (self.R_s1 if self.shared_R_mode == "pad" else self.R_s2))
            zm = F.linear(phi, (self.R_m1 if i==0 else self.R_m2))
            # else:
            #     zs = torch.matmul(z, (self.R_s1[shared_sv].T if i==0 else self.R_s2[shared_sv].T))
            #     zm = torch.matmul(z, (self.R_m1[m1_sv].T if i==0 else self.R_m2[m2_sv].T))
            rep_components.append((zm, zs))
        return rep_components

    def fuse_representations(self, phis):
        """Fuse representations."""
        zs1 = F.linear(phis[0], self.R_s1)
        zm1 = F.linear(phis[0], self.R_m1)
        zs2 = F.linear(phis[1], self.R_s1 if self.shared_R_mode == "pad" else self.R_s2)
        zm2 = F.linear(phis[1], self.R_m2)
        # Choose zs1 or zs2 based on a binary random sample
        random_zs = zs1 if torch.randint(0, 2, (1,)).item() == 0 else zs2
        return torch.cat((zm1, zm2, random_zs), dim=-1)

    def compute_stage_losses(self, h1, h2, z_components):
        # Compute all losses
        # l_shared = loss_shared_consistency(z_components[0][1], z_components[1][1])
        l_orthogonality = loss_orthogonality(self.R_s1, self.R_m1, self.R_m2)
        l_independence = loss_independence(z_s1=z_components[0][1], z_s2=z_components[1][1], z_m1=z_components[0][0], z_m2=z_components[1][0], mod=1)
        # l_independence = -loss_mutual_info(h1, h2, z_components, mode="private")
        l_mi = loss_mutual_info(h1, h2, z_components, mode="shared" if self.trainable_stage == "shared" else "all")
        
        all_losses = [l_orthogonality.item(), l_independence.item(), l_mi.item()]
        all_loss_names = ["Orthogonality Loss", "Independence Loss", "Mutual Info Loss"]
        
        # Return appropriate losses based on stage
        if self.trainable_stage == "shared":
            return [l_mi], ["Mutual Info Loss"], all_losses, all_loss_names
        elif self.trainable_stage == "private":
            return [l_orthogonality, l_independence, l_mi], ["Orthogonality Loss", "Independence Loss", "Mutual Info Loss"], all_losses, all_loss_names
        elif self.trainable_stage == "joint":
            return [l_orthogonality, l_independence, l_mi], ["Orthogonality Loss", "Independence Loss", "Mutual Info Loss"], all_losses, all_loss_names
        else:
            raise ValueError(f"Unknown training stage: {stage}")

    def evaluate_validation_loss(self, val_dataloader, **kwargs):
        """Evaluate model on validation set."""
        val_total_loss = 0
        val_loss_list = [0,0,0]  # Initialize list to store losses
        self.eval()
        loss_balancer = GradientNormalizedLoss(num_losses=2)
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
                
                h1 = h1.to(self.device)
                h2 = h2.to(self.device)
                phis = self.forward([h1, h2])
                
                z_components = self.decouple(phis, full=True)
                # z_components = self.decouple([h1, h2], full=True)
                losses_list, _, all_losses_list, _ = self.compute_stage_losses(h1, h2, z_components)
                val_loss = torch.stack(losses_list).mean()
                # val_loss, _ = loss_balancer(torch.stack(losses_list), weights=loss_weights)

                # if self.trainable_stage == "shared":
                #     val_loss = losses_list[0]
                # else:
                #     val_loss = losses_list[0] + losses_list[1]

                val_total_loss += val_loss.item()
                # Return both total loss and average loss list
                val_loss_list[0]+=all_losses_list[0]
                val_loss_list[1]+=all_losses_list[1]
                val_loss_list[2]+=all_losses_list[2]
                torch.cuda.empty_cache()
        if self.encoders is not None:
            del model_output, embeddings_en, embeddings_fr
        self.train()
        
        # Average the losses over batches
        # val_loss_list = [loss / len(val_dataloader) for loss in val_loss_list]
        return val_total_loss / len(val_dataloader), [l / len(val_dataloader) for l in val_loss_list]


    def train_projection(self, dataloader, val_dataloader, early_stopping_config, lr=1e-3, epochs=100, exp_name="projection_module", **kwargs):
        """Train the projection model with early stopping."""
        # Create checkpoints directory if it doesn't exist
        os.makedirs("./ckpts", exist_ok=True)
        os.makedirs("./plots/%s"%(self.dataset_name), exist_ok=True)
        os.makedirs("./logs", exist_ok=True)
        save_path='./ckpts/%s_%s.pth'%(self.dataset_name, exp_name)
        print(f"Training on device: {self.device}")
        print(f"Model is on device: {next(self.parameters()).device}")
        # Initialize loss tracking
        self.lr = lr
        loss_balancer = GradientNormalizedLoss(num_losses=3)
        all_epoch_losses = []
        all_val_losses = []
        trainable_params = self.get_trainable_parameters()
        if self.shared_R_mode == "double":
            self.optimizer = torch.optim.Adam([
                    {"params": [self.R_s1, self.R_s2] + list(self.W_s0.parameters()) + list(self.W_s1.parameters()), "lr": lr},
                    {"params": [self.R_m1, self.R_m2] + list(self.W_m0.parameters()) + list(self.W_m1.parameters()), "lr": lr}
                ], weight_decay=1e-3)
        else:
            self.optimizer = torch.optim.Adam([
                {"params": [self.R_s1] + list(self.W_s0.parameters()) + list(self.W_s1.parameters()), "lr": lr},
                {"params": [self.R_m1, self.R_m2] + list(self.W_m0.parameters()) + list(self.W_m1.parameters()), "lr": lr}
            ], weight_decay=1e-3)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=500)
        
        epoch_loss_list = [[],[],[]]
        val_loss_list = [[],[],[]]
        total_loss_list = []
        total_val_loss_list = []
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_loss_components = [0, 0, 0]
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
                
                # h1 = F.normalize(h1.float(), dim=1).to(self.device)
                # h2 = F.normalize(h2.float(), dim=1).to(self.device)
                h1 = h1.to(self.device)
                h2 = h2.to(self.device)
                phis = self.forward([h1, h2])
                
                z_components = self.decouple(phis, full=True)
                losses_list, loss_names, all_losses, all_loss_names = self.compute_stage_losses(h1, h2, z_components)
                
                losses = torch.stack(losses_list)
                
                loss, weights = loss_balancer(losses, trainable_params)
                # weights = [0.1] if self.trainable_stage == "shared" else [1, 0.1]
                # loss = sum(l * w for l, w in zip(losses, weights))

                if self.trainable_stage == "shared":
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                elif self.trainable_stage == "private":
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                else:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                epoch_loss += loss.item()
                epoch_loss_components[0] += all_losses[0]
                epoch_loss_components[1] += all_losses[1]
                epoch_loss_components[2] += all_losses[2]
                del  h1, h2, phis, z_components
                torch.cuda.empty_cache()
            
            total_loss_list.append(epoch_loss/len(dataloader))
            epoch_loss_list[0].append(epoch_loss_components[0]/len(dataloader))
            epoch_loss_list[1].append(epoch_loss_components[1]/len(dataloader))
            epoch_loss_list[2].append(epoch_loss_components[2]/len(dataloader))
            # Update metrics
            # epoch_losses = [np.mean(l) for l in epoch_losses]
            
            val_loss, val_loss_components = self.evaluate_validation_loss(val_dataloader, **kwargs)
            total_val_loss_list.append(val_loss)
            val_loss_list[0].append(val_loss_components[0]/len(val_dataloader))
            val_loss_list[1].append(val_loss_components[1]/len(val_dataloader))
            val_loss_list[2].append(val_loss_components[2]/len(val_dataloader))
            if self.pruning:
                if self.trainable_stage == "joint" and (abs(val_loss_list[-1][-1]) <= abs(1.05*self.stage_tracking['best_val_MI_loss'])) and epoch>self.stage_switches[-1][-1]+50:  # Index 2 is MI loss based on all_loss_names, allow 5% margin
                # if self.trainable_stage == "joint" and (abs(val_loss) <= abs(1.1*self.stage_tracking['best_val_loss'])) and epoch>self.stage_switches[-1][-1]+50:  # Index 2 is MI loss based on all_loss_names, allow 5% margin
                    self.prune_singular_values()
                    # optimizer = self.update_optimizer(optimizer, lr=lr)
                    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
            
            if self.staging:
                stage_config = early_stopping_config[self.trainable_stage]
                self.stage_tracking["min_epochs_counter"] += 1
                
                if len(total_val_loss_list) >= 5:
                    recent_avg_val_loss = np.mean(total_val_loss_list[-5:])
                else:
                    recent_avg_val_loss = np.mean(total_val_loss_list)
                relative_improvement = (self.stage_tracking["best_val_loss"] - recent_avg_val_loss) / self.stage_tracking["best_val_loss"]
                # if np.mean(val_loss)<self.stage_tracking["best_val_loss"]:
                if recent_avg_val_loss<self.stage_tracking["best_val_loss"]:
                    self.stage_tracking["best_val_loss"] = recent_avg_val_loss#np.mean(val_loss)
                if self.trainable_stage == "joint" and val_loss_list[-1][-1] <self.stage_tracking["best_val_MI_loss"]:
                    self.stage_tracking["best_val_MI_loss"] = val_loss_list[-1][-1]
                
                # Update tracking metrics
                if relative_improvement > stage_config["min_improvement_ratio"]:
                    self.stage_tracking["plateau_counter"] = 0
                else:
                    self.stage_tracking["plateau_counter"] += 1
                
                # # Update optimizer
                # trainable_params = model.get_trainable_parameters()
                # optimizer = torch.optim.Adam(trainable_params, lr=1e-4, weight_decay=1e-4)
            
            # Print progress
            if epoch % 1 == 0:
                print(f"[Epoch {epoch}] {self.trainable_stage.upper()} stage: "
                    f"val_loss={np.mean(val_loss):.4f}, "
                    f"best_val_loss={self.stage_tracking['best_val_loss']:.4f}, ")
                loss_report = ", ".join(f"{name}={val:.4f}" for val, name in zip([l[-1] for l in epoch_loss_list], all_loss_names))
                print(f"Loss values: {loss_report}")
                if self.staging:
                    print(f"relative_improvement={relative_improvement*100:.2f}%, "
                    f"plateau={self.stage_tracking['plateau_counter']}/{stage_config['patience']}, "
                    f"epochs={self.stage_tracking['min_epochs_counter']}/{stage_config['max_epochs']}")
            self.save_checkpoint(epoch, loss, filepath=save_path)
            should_switch = (
                                self.stage_tracking["plateau_counter"] >= stage_config["patience"] or
                                self.stage_tracking["min_epochs_counter"] >= stage_config["max_epochs"]
                            )

            if self.staging and should_switch:
                # self.optimizer_specific = torch.optim.Adam(
                #     [self.R_m1, self.R_m2] + list(self.W_m0.parameters()) + list(self.W_m1.parameters()), 
                #     lr=self.lr, weight_decay=1e-3
                # )   
                # self.optimizer_shared = torch.optim.Adam(
                #     [self.R_s1, self.R_s2] + list(self.W_s0.parameters()) + list(self.W_s1.parameters()), 
                #     lr=self.lr, weight_decay=1e-3
                # )
                # self.scheduler_specific = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_specific, T_max=500)
                # self.scheduler_shared = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_shared, T_max=500)
                if self.trainable_stage == "shared":
                    self.trainable_stage = "private"
                    self.stage_switches = getattr(self, 'stage_switches', [])
                    self.stage_switches.append(('private', epoch))
                    print(f"***** [Epoch {epoch}] → Switched to PRIVATE stage after {self.stage_tracking['min_epochs_counter']} epochs ***** ")
                elif self.trainable_stage == "private":
                    self.lr = self.lr/10
                    self.trainable_stage = "joint"
                    # self.stage_tracking["best_val_MI_loss"] = 5000
                    self.stage_switches = getattr(self, 'stage_switches', [])
                    self.stage_switches.append(('joint', epoch))
                    print(f"***** [Epoch {epoch}] → Switched to JOINT stage after {self.stage_tracking['min_epochs_counter']} epochs ***** ")
                elif self.trainable_stage == "joint":
                    print(f"Final {self.trainable_stage} stage loss: {val_loss:.4f}")
                    print("Training complete.")
                    return
                # optimizer = self.update_optimizer(optimizer, lr=lr)
                # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
                self.stage_tracking["best_val_loss"] = 5000
                self.stage_tracking["plateau_counter"] = 0
                self.stage_tracking["min_epochs_counter"] = 0
        
            # Plot final losses
            epoch_loss_list.append(total_loss_list)
            plot_losses(np.array(epoch_loss_list), loss_names=all_loss_names+["Total weighted Loss"], save_path="./plots/%s/%s_loss_curves.pdf"%(self.dataset_name, exp_name), log_path="./logs/%s_loss_curves.csv"%(exp_name), stage_switches=self.stage_switches)
            
            val_loss_list.append(total_val_loss_list)
            plot_losses(np.array(val_loss_list), loss_names=all_loss_names+["Total weighted Loss"], save_path="./plots/%s/%s_loss_curves_validation.pdf"%(self.dataset_name, exp_name), log_path="./logs/%s_loss_curves_validation.csv"%(exp_name), stage_switches=self.stage_switches)

        self.save_checkpoint(epoch, loss, filepath=save_path)


    def save_checkpoint(self, epoch, loss, filepath):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict()
        }, filepath)

