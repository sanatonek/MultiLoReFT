"""
Multi-LoReFT: Low-Rank Factorization for Multimodal Representation Learning.

Implements the MultiLoReFT model for decomposing multimodal embeddings into
shared and modality-specific subspaces with optional staged training and
singular value pruning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from collections import defaultdict

from .utils import load_checkpoint
from .losses import (
    loss_orthogonality,
    loss_independence,
    loss_mutual_info,
    GradientNormalizedLoss,
)
import wandb

__all__ = ["MultiLoReFT"]


class MultiLoReFT(nn.Module):
    """LoReFT module for multimodal projection learning."""

    def __init__(
        self,
        input_dims,
        shared_rank,
        specific_rank,
        staging=True,
        encoders=None,
        intervene_layer=-1,
        pruning_threshold=0.05,
        pruning=True,
        device=None,
        dataset_name="simulated",
    ):
        super(MultiLoReFT, self).__init__()
        self.shared_rank = shared_rank
        self.specific_rank = specific_rank
        self.pruning_threshold = pruning_threshold
        self.pruned = False
        self.encoders = encoders
        self.dataset_name = dataset_name
        self.intervene_layer = intervene_layer
        self.input_dims = input_dims
        self.modality_count = len(input_dims)
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
        self.n_loss_components = 3
        self.stage_tracking = {
            "best_val_loss": 5000,
            "best_val_MI_loss": 5000,
            "plateau_counter": 0,
            "min_epochs_counter": 0,
        }
        self.max_dim = max(input_dims)

        # Initialize projection matrices (use ModuleList/ParameterList so .to(device) moves all submodules)
        self.R_s1 = nn.Parameter(torch.randn(shared_rank, self.max_dim, dtype=torch.float32))
        self.R_ms = nn.ParameterList(
            [
                nn.Parameter(torch.randn(specific_rank, self.max_dim, dtype=torch.float32))
                for _ in range(self.modality_count)
            ]
        )
        self.W_ms = nn.ModuleList(
            [self._create_weight_networks(self.max_dim, self.specific_rank) for _ in range(self.modality_count)]
        )
        self.W_ss = nn.ModuleList(
            [self._create_weight_networks(self.max_dim, self.shared_rank) for _ in range(self.modality_count)]
        )
        for i in range(self.modality_count):
            self.W_ss[i].apply(self._init_weights)
            self.W_ms[i].apply(self._init_weights)
        self._orthogonal_init()

    def _params_shared_head(self):
        """Flat list: R_s1 + all W_ss parameters (for optimizer / pruning)."""
        params = [self.R_s1]
        for i in range(self.modality_count):
            params.extend(list(self.W_ss[i].parameters()))
        return params

    def _params_private_head(self):
        """Flat list: all R_ms + all W_ms parameters."""
        params = [self.R_ms[i] for i in range(self.modality_count)]
        for i in range(self.modality_count):
            params.extend(list(self.W_ms[i].parameters()))
        return params

    def _create_weight_networks(self, input_dim, output_dim):
        """Create weight networks for each modality."""
        model = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(input_dim * 2, output_dim, dtype=torch.float32),
        )
        return model

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def _orthogonal_init(self):
        nn.init.orthogonal_(self.R_s1, gain=1)
        for i in range(self.modality_count):
            nn.init.orthogonal_(self.R_ms[i], gain=1)

    def get_trainable_parameters(self):
        """Get parameters to train based on current stage."""
        if self.trainable_stage == "shared":
            return self._params_shared_head()
        elif self.trainable_stage == "private":
            return self._params_private_head()
        elif self.trainable_stage == "joint":
            return self._params_shared_head() + self._params_private_head()
        else:
            raise ValueError(f"Unknown training stage: {self.trainable_stage}")

    def prune_singular_values(self, single=False):
        """Prune singular values below threshold and update network weights."""
        def prune_matrix(name, R, weights_to_prune):
            if R.shape[0] < 3:
                return R, R.shape[0], False
            U, S, Vh = torch.linalg.svd(R, full_matrices=False)
            if single:
                min_sv_idx = torch.argmin(S)
                min_sv = S[min_sv_idx]
                if min_sv > self.pruning_threshold:
                    return R, len(S), False
                n_remove = 1
            else:
                below_threshold = S < self.pruning_threshold
                num_below = below_threshold.sum().item()
                print(f"Number of singular values below threshold (total size: {len(S)}): {num_below}")
                if num_below == 0:
                    return R, len(S), False
                n_remove = max(1, min(num_below, int(0.1 * len(S) )))
            k = R.shape[0] - n_remove
            reduced_R = S[:k].unsqueeze(1) * Vh[:k, :]
            reduced_R = reduced_R.to(device=self.device, dtype=torch.float32)

            UkT = U[:, :k].T
            for i, weight_seq in enumerate(weights_to_prune):
                last_layer = weight_seq[-1]
                assert isinstance(last_layer, nn.Linear), "Expected last layer to be nn.Linear"
                in_features = last_layer.in_features
                device = last_layer.weight.device
                dtype = last_layer.weight.dtype

                new_last = nn.Linear(in_features, k, bias=True, device=device, dtype=dtype)
                with torch.no_grad():
                    new_last.weight.copy_(UkT @ last_layer.weight.data)
                    if last_layer.bias is not None:
                        new_last.bias.copy_(UkT @ last_layer.bias.data)
                    else:
                        nn.init.zeros_(new_last.bias)
                weight_seq[-1] = new_last

            print(f"Pruned {name} to {len(reduced_R)} dimensions: {reduced_R.shape}")
            self.stage_tracking["plateau_counter"] = 0
            return reduced_R, k, True

        kept_s1, kept_s2, kept_m1, kept_m2 = 0, 0, 0, 0
        pruned_R, kept_s1, is_pruned = prune_matrix("R_s1", self.R_s1, [self.W_ss[i] for i in range(self.modality_count)])
        if is_pruned:
            self.shared_rank = kept_s1
            self.R_s1 = nn.Parameter(pruned_R)
            self.optimizer.param_groups[0]["params"] = self._params_shared_head()

        for i in range(self.modality_count):
            pruned_R, kept_m, is_pruned = prune_matrix(f"R_m{i}", self.R_ms[i], [self.W_ms[i]])
            if is_pruned:
                self.specific_rank = kept_m
                self.R_ms[i] = nn.Parameter(pruned_R)
                self.optimizer.param_groups[1]["params"] = self._params_private_head()
        self.optimizer.state = defaultdict(dict, self.optimizer.state)

    def forward(self, embeddings):
        hs = [F.normalize(embedding, p=2, dim=-1) for embedding in embeddings]

        if len(np.unique(self.input_dims)) > 1:
            hs = [F.pad(h, (0, self.max_dim - h.shape[1])) for h in hs]

        phis = []
        for ind,h in enumerate(hs):
            proj_s = self.W_ss[ind](h) - F.linear(h, self.R_s1)
            shared_h = F.linear(proj_s, self.R_s1.T)
            proj_m = self.W_ms[ind](h) - F.linear(h, self.R_ms[ind])
            spec_h = F.linear(proj_m, self.R_ms[ind].T)
            phi = h + shared_h + spec_h
            phis.append(phi)
        return phis

    def decouple(self, phis, full=True, th=0.1):
        """Separate shared and modality-specific representations."""
        rep_components = []
        for i, phi in enumerate(phis):
            zs = F.linear(phi, self.R_s1)
            zm = F.linear(phi, self.R_ms[i])
            rep_components.append((zm, zs))
        return rep_components

    def fuse_representations(self, phis):
        """Fuse representations."""
        zs_list = []
        zm_list = []
        for ind, phi in enumerate(phis):
            zs_list.append(F.linear(phi, self.R_s1))
            zm_list.append(F.linear(phi, self.R_ms[ind]))
        mean_zs = sum(zs_list) / len(zs_list)
        return torch.cat((*zm_list, mean_zs), dim=-1)

    def compute_stage_losses(self, hs, phis, z_components, return_distortion_metrics=False):
        """Compute losses based on current training stage."""
        l_orthogonality = loss_orthogonality(self.R_s1, self.R_ms)
        l_independence = loss_independence(z_components)
        if return_distortion_metrics:
            l_mi, l_mi_components = loss_mutual_info(hs, z_components, mode=self.trainable_stage, return_all=True)
        else:
            l_mi = loss_mutual_info(hs, z_components, mode=self.trainable_stage)

        all_losses = [l_orthogonality.item(), l_independence.item(), l_mi.item()]
        all_loss_names = ["Orthogonality Loss", "Independence Loss", "Mutual Info Loss"]

        if self.trainable_stage == "shared":
            self.n_loss_components = 1
            return [l_mi], ["Mutual Info Loss"], all_losses, all_loss_names
        elif self.trainable_stage == "private":
            self.n_loss_components = 3
            return (
                [l_orthogonality, l_independence, l_mi],
                ["Orthogonality Loss", "Independence Loss", "Mutual Info Loss"],
                all_losses,
                all_loss_names,
            )
        elif self.trainable_stage == "joint":
            self.n_loss_components = 3
            if return_distortion_metrics:
                return (
                    [l_orthogonality, l_independence, l_mi],
                    ["Orthogonality Loss", "Independence Loss", "Mutual Info Loss"],
                    all_losses,
                    all_loss_names,
                    l_mi_components,
                )
            else:
                return (
                [l_orthogonality, l_independence, l_mi],
                ["Orthogonality Loss", "Independence Loss", "Mutual Info Loss"],
                all_losses,
                all_loss_names,
            )
        else:
            raise ValueError(f"Unknown training stage: {self.trainable_stage}")

    def evaluate_validation_loss(self, val_dataloader, **kwargs):
        """Evaluate model on validation set."""
        val_total_loss = 0
        val_loss_list = [0, 0, 0]
        self.eval()
        with torch.no_grad():
            for val_batch in val_dataloader:
                if self.encoders is not None:
                    x1, x2, label = val_batch
                    label = label.to(self.device)
                    x1 = x1.to(self.device)
                    x2 = x2.to(self.device)
                    tokens_en = en_tokenizer(
                        x2, return_tensors="pt", padding=True, truncation=True
                    ).to(self.device)
                    tokens_fr = fr_tokenizer(
                        x2, padding=True, truncation=True, return_tensors="pt"
                    ).to(self.device)
                    model_output = self.encoders[2](**tokens_fr)
                    embeddings_fr = model_output.last_hidden_state[:, 0, :].to(self.device)
                    model_output = self.encoders[1](**tokens_en)
                    embeddings_en = model_output.last_hidden_state[:, 0, :].to(self.device)
                    h1 = self.encoders[0].forward_features(x1)[:, 0, :].to(self.device)
                    h2 = torch.where(
                        label.unsqueeze(1).expand(-1, embeddings_en.size(1)) == 0,
                        embeddings_en,
                        embeddings_fr,
                    )
                else:
                    if self.dataset_name == "flickr":
                        h1 = val_batch[0]
                        h2 = val_batch[1]
                        lang_idx = torch.randint(0, 2, (len(h1),), device=h1[0].device)
                        h2 = torch.stack([h2[0], h2[1]], dim=1).gather(
                            1,
                            lang_idx.unsqueeze(1)
                            .unsqueeze(2)
                            .expand(-1, -1, h2[0].shape[-1]),
                        ).squeeze(1)
                    else:
                        h1 = val_batch[0]
                        h2 = val_batch[1]

                h1 = h1.to(self.device)
                h2 = h2.to(self.device)
                phis = self.forward([h1, h2])

                z_components = self.decouple(phis, full=True)
                losses_list, _, all_losses_list, _, val_mi_components = self.compute_stage_losses(
                    [h1, h2], phis, z_components, return_distortion_metrics=True
                )
                val_loss = torch.stack(losses_list).mean()
                val_total_loss += val_loss.item()
                val_loss_list[0] += all_losses_list[0]
                val_loss_list[1] += all_losses_list[1]
                val_loss_list[2] += all_losses_list[2]
                torch.cuda.empty_cache()
        if self.encoders is not None:
            del model_output, embeddings_en, embeddings_fr
        self.train()

        return val_total_loss / len(val_dataloader), [
            l / len(val_dataloader) for l in val_loss_list
        ]

    def train_projection(
        self,
        dataloader,
        val_dataloader,
        early_stopping_config,
        lr=1e-3,
        epochs=100,
        exp_name="projection_module",
        **kwargs,
    ):
        """Train the projection model with early stopping."""
        os.makedirs("./ckpts", exist_ok=True)
        os.makedirs("./plots/%s" % self.dataset_name, exist_ok=True)
        os.makedirs("./logs", exist_ok=True)
        save_path = "./ckpts/%s_%s.pth" % (self.dataset_name, exp_name)
        print(f"Training on device: {self.device}")
        print(f"Model is on device: {next(self.parameters()).device}")
        self.lr = lr
        trainable_params = self.get_trainable_parameters()
        self.optimizer = torch.optim.Adam(
            [
                {"params": self._params_shared_head(), "lr": lr},
                {"params": self._params_private_head(), "lr": lr},
            ],
            weight_decay=1e-3,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=500
        )

        epoch_loss_list = [[], [], []]
        val_loss_list = [[], [], []]
        total_loss_list = []
        total_val_loss_list = []

        loss_balancer = GradientNormalizedLoss(num_losses=self.n_loss_components)
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_loss_components = [0]*self.n_loss_components
            for batch in dataloader:
                if self.encoders is not None:
                    x1, x2, label = batch
                    label = label.to(self.device)
                    x1 = x1.to(self.device)
                    x2 = x2.to(self.device)
                    with torch.no_grad():
                        tokens_en = en_tokenizer(
                            x2, return_tensors="pt", padding=True, truncation=True
                        ).to(self.device)
                        tokens_fr = fr_tokenizer(
                            x2, padding=True, truncation=True, return_tensors="pt"
                        ).to(self.device)
                        model_output = self.encoders[2](**tokens_fr)
                        embeddings_fr = model_output.last_hidden_state[:, 0, :].to(self.device)
                        model_output = self.encoders[1](**tokens_en)
                        embeddings_en = model_output.last_hidden_state[:, 0, :].to(self.device)
                        h1 = self.encoders[0].forward_features(x1)[:, 0, :].to(
                            self.device
                        )
                        h2 = torch.where(
                            label.unsqueeze(1).expand(-1, embeddings_en.size(1)) == 0,
                            embeddings_en,
                            embeddings_fr,
                        )
                else:
                    if self.dataset_name == "flickr":
                        h1 = batch[0]
                        h2 = batch[1]
                        lang_idx = batch[-1]
                        h2 = torch.stack([h2[0], h2[1]], dim=1).gather(
                            1,
                            lang_idx.unsqueeze(1)
                            .unsqueeze(2)
                            .expand(-1, -1, h2[0].shape[-1]),
                        ).squeeze(1)
                        hs = [h1, h2]
                    else:
                        hs = [batch[i] for i in range(self.modality_count)]

                hs = [h.to(self.device) for h in hs]
                phis = self.forward(hs)

                z_components = self.decouple(phis, full=True)
                losses_list, loss_names, all_losses, all_loss_names = self.compute_stage_losses(
                    hs, phis, z_components
                )
                
                
                losses = torch.stack(losses_list)
                loss, weights = loss_balancer(losses, trainable_params)
                # weights = [1.0, 1.0, 1.0]
                # loss = sum(weights[i] * losses_i for i, losses_i in enumerate(losses))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                epoch_loss += loss.item()
                for i in range(self.n_loss_components):
                    epoch_loss_components[i] += all_losses[i]
                del hs, phis, z_components
                torch.cuda.empty_cache()

            total_loss_list.append(epoch_loss / len(dataloader))
            for i in range(self.n_loss_components):
                epoch_loss_list[i].append(epoch_loss_components[i] / len(dataloader))

            val_loss, val_loss_components = self.evaluate_validation_loss(
                val_dataloader, **kwargs
            )
            total_val_loss_list.append(val_loss)
            for i in range(self.n_loss_components):
                val_loss_list[i].append(val_loss_components[i] / len(val_dataloader))

            if len(total_val_loss_list) >= 5:
                recent_avg_val_loss = np.mean(total_val_loss_list[-5:])
                recent_avg_mi_loss = np.mean(val_loss_list[-1][-5:])
            else:
                recent_avg_val_loss = np.mean(total_val_loss_list)
                recent_avg_mi_loss = np.mean(val_loss_list[-1][-1])

            if self.pruning:
                if (
                    self.trainable_stage == "joint"
                    and abs(val_loss_list[-1][-1])  <= abs(1.01 * self.stage_tracking["best_val_MI_loss"])
                    # and abs(recent_avg_mi_loss)  <= abs(1.01 * self.stage_tracking["best_val_MI_loss"])
                    and epoch > self.stage_switches[-1][-1] + 100
                ):
                    self.prune_singular_values()

            if self.staging or self.trainable_stage == "joint":
                stage_config = early_stopping_config[self.trainable_stage]
                self.stage_tracking["min_epochs_counter"] += 1
                relative_improvement = (
                    self.stage_tracking["best_val_loss"] - recent_avg_val_loss
                ) / self.stage_tracking["best_val_loss"]
                if recent_avg_val_loss < self.stage_tracking["best_val_loss"]:
                    self.stage_tracking["best_val_loss"] = recent_avg_val_loss
                if (
                    self.trainable_stage == "joint"
                    and recent_avg_mi_loss < self.stage_tracking["best_val_MI_loss"]
                ):
                    self.stage_tracking["best_val_MI_loss"] = recent_avg_mi_loss

                if relative_improvement > stage_config["min_improvement_ratio"]:
                    self.stage_tracking["plateau_counter"] = 0
                else:
                    self.stage_tracking["plateau_counter"] += 1
            else:
                stage_config = early_stopping_config["joint"]

            wandb.log({"Total loss (Train)": loss})
            wandb.log({"Orthogonality Loss (Train)": all_losses[0]})
            wandb.log({"Independence Loss (Train)": all_losses[1]})
            wandb.log({"Mutual Info Loss (Train)": all_losses[2]})
            wandb.log({"Validation loss": val_loss, "epoch": epoch})
            wandb.log({"Validation Orthogonality Loss": val_loss_list[0][-1]})
            wandb.log({"Validation Independence Loss": val_loss_list[1][-1]})
            wandb.log({"Validation Mutual Info Loss": val_loss_list[2][-1]})
            wandb.log({"Shared rank": self.R_s1.shape[0]})
            for i in range(self.modality_count):
                wandb.log({f"Specific rank modality {i}": self.R_ms[i].shape[0]})

            if epoch % 5 == 0:
                print(
                    f"[Epoch {epoch}] {self.trainable_stage.upper()} stage: "
                    f"val_loss={val_loss:.4f}, "
                    f"best_val_loss={self.stage_tracking['best_val_loss']:.4f}, "
                    f"best_val_MI_loss={self.stage_tracking['best_val_MI_loss']:.4f}"
                )
                loss_report = ", ".join(
                    f"{name}={val:.4f}"
                    for val, name in zip(
                        [l[-1] for l in val_loss_list], all_loss_names
                    )
                )
                print(f"Loss values: {loss_report}")
                if self.staging:
                    print(
                        f"relative_improvement={relative_improvement*100:.2f}%, "
                        f"plateau={self.stage_tracking['plateau_counter']}/{stage_config['patience']}, "
                        f"epochs={self.stage_tracking['min_epochs_counter']}/{stage_config['max_epochs']}"
                    )
            self.save_checkpoint(epoch, loss, filepath=save_path)
            should_switch = (
                self.stage_tracking["plateau_counter"] >= stage_config["patience"]
                or self.stage_tracking["min_epochs_counter"]
                >= stage_config["max_epochs"]
            )

            if should_switch:
                if self.trainable_stage == "shared":
                    self.trainable_stage = "private"
                    self.stage_switches = getattr(self, "stage_switches", [])
                    self.stage_switches.append(("private", epoch))
                    print(
                        f"***** [Epoch {epoch}] → Switched to PRIVATE stage after {self.stage_tracking['min_epochs_counter']} epochs *****"
                    )
                elif self.trainable_stage == "private":
                    self.trainable_stage = "joint"
                    self.stage_switches = getattr(self, "stage_switches", [])
                    self.stage_switches.append(("joint", epoch))
                    print(
                        f"***** [Epoch {epoch}] → Switched to JOINT stage after {self.stage_tracking['min_epochs_counter']} epochs *****"
                    )
                elif self.trainable_stage == "joint":
                    print(f"Final {self.trainable_stage} stage loss: {val_loss:.4f}")
                    print("Training complete.")
                    return
                self.stage_tracking["best_val_loss"] = 5000
                self.stage_tracking["plateau_counter"] = 0
                self.stage_tracking["min_epochs_counter"] = 0

        self.save_checkpoint(epoch, loss, filepath=save_path)

    def save_checkpoint(self, epoch, loss, filepath):
        """Save model checkpoint."""
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.state_dict(),
            },
            filepath,
        )