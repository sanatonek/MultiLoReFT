"""Apollo baseline: multimodal VAE with shared and modality-specific latents."""
import sys
import os as _os
_script_dir = _os.path.dirname(_os.path.abspath(__file__))
_project_root = _os.path.dirname(_os.path.dirname(_script_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.baselines.baselines_utils import (
    parse_baseline_args,
    gather_samples,
    get_flickr_batch,
    run_evaluation,
    aggregate_seed_results,
    print_aggregated_results,
)

from src.data.base import MultimodalDataset
from scripts.evaluate_representations import evaluate_predictability


class Encoder(nn.Module):
    def __init__(self, input_size, shared_size, modality_specific_size, hidden_size=20):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2_shared = nn.Linear(hidden_size, shared_size)
        self.fc2_modality_specific = nn.Linear(hidden_size, modality_specific_size)

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        x = F.relu(self.fc1(x))
        z_shared = self.fc2_shared(x)
        z_modality_specific = self.fc2_modality_specific(x)
        return z_shared, z_modality_specific


class Decoder(nn.Module):
    def __init__(self, output_size, shared_size, modality_specific_size, hidden_size=20):
        super().__init__()
        self.fc1 = nn.Linear(shared_size + modality_specific_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, z_shared, z_modality_specific):
        z = torch.cat([z_shared, z_modality_specific], dim=-1)
        h = F.relu(self.fc1(z))
        return self.fc2(h)


class Apollo(nn.Module):
    """Apollo: shared and modality-specific latent VAE."""

    def __init__(self, encoders, decoders, n_train, z_sizes, shared_size,
                 modality_names, modality_shapes, ckpt_path):
        super().__init__()
        self.z_sizes = z_sizes
        self.shared_size = shared_size
        self.n_train = n_train
        self.modality_names = modality_names
        self.modality_shapes = modality_shapes
        self.ckpt_path = ckpt_path
        self.posterior_means = nn.ModuleDict({
            mod: nn.Embedding(n_train, z_sizes[mod]) for mod in modality_names
        })
        self.shared_post_mean = nn.Embedding(n_train, shared_size)
        self.encoders = nn.ModuleDict(encoders)
        self.decoders = nn.ModuleDict(decoders)

    def encode(self, x_batch):
        """Encode batch dict {mod: tensor} -> (z_s_all, z_m_all)."""
        z_s_all, z_m_all = {}, {}
        for mod, encoder in self.encoders.items():
            x_m = x_batch[mod]
            if isinstance(x_m, list):
                x_m = torch.stack([x_m[int(x_batch[-1][i])][i] for i in range(len(x_m))], dim=0)
            if not isinstance(x_m, torch.Tensor):
                x_m = torch.tensor(x_m, dtype=torch.float32)
            z_s, z_m = encoder(x_m)
            z_s_all[mod] = z_s
            z_m_all[mod] = z_m
        return z_s_all, z_m_all

    def get_components_for_eval(self, z_s_all, z_m_all):
        """Build components list for evaluate_predictability."""
        zs = (z_s_all["A"] + z_s_all["B"]) / 2
        return [
            ("Zs", zs.detach().cpu().numpy()),
            ("Zm1", z_m_all["A"].detach().cpu().numpy()),
            ("Zm2", z_m_all["B"].detach().cpu().numpy()),
            ("Z", np.concatenate([
                z_m_all["A"].detach().cpu().numpy(),
                z_m_all["B"].detach().cpu().numpy(),
                zs.detach().cpu().numpy(),
            ], axis=1)),
        ]

    def load_from_checkpoint(self, seed):
        for mod in self.modality_names:
            path = os.path.join(self.ckpt_path, f"{mod}_encoder_{seed}.pth")
            self.encoders[mod].load_state_dict(torch.load(path))
            path = os.path.join(self.ckpt_path, f"{mod}_decoder_{seed}.pth")
            self.decoders[mod].load_state_dict(torch.load(path))
            z_mod = np.load(os.path.join(self.ckpt_path, f"modality_z_{mod}_{seed}.npy"), allow_pickle=True)
            if isinstance(z_mod, np.ndarray) and z_mod.dtype == np.object_:
                z_mod = np.stack(z_mod).astype(np.float32)
            self.posterior_means[mod].weight.data.copy_(torch.from_numpy(z_mod))
        z_shared = np.load(os.path.join(self.ckpt_path, f"shared_z_{seed}.npy"), allow_pickle=True)
        if isinstance(z_shared, np.ndarray) and z_shared.dtype == np.object_:
            z_shared = np.stack(z_shared).astype(np.float32)
        self.shared_post_mean.weight.data.copy_(torch.from_numpy(z_shared))

    def save_to_checkpoint(self, seed):
        os.makedirs(self.ckpt_path, exist_ok=True)
        for mod in self.modality_names:
            torch.save(self.encoders[mod].state_dict(),
                       os.path.join(self.ckpt_path, f"{mod}_encoder_{seed}.pth"))
            torch.save(self.decoders[mod].state_dict(),
                       os.path.join(self.ckpt_path, f"{mod}_decoder_{seed}.pth"))
            np.save(os.path.join(self.ckpt_path, f"modality_z_{mod}_{seed}.npy"),
                    self.posterior_means[mod].weight.data.cpu().numpy())
        np.save(os.path.join(self.ckpt_path, f"shared_z_{seed}.npy"),
                self.shared_post_mean.weight.data.cpu().numpy())

    def train(self, trainloader, dataset_name, lr_enc=0.001, lr_dec=0.001, lr_optim=0.001,
              epochs_enc=20, epochs_dec=20, seed=0):
        decoder_params = [p for mod in self.modality_names for p in self.decoders[mod].parameters()]
        encoder_params = [p for mod in self.modality_names for p in self.encoders[mod].parameters()]
        latent_params = list(self.posterior_means[self.modality_names[0]].parameters())
        latent_params += list(self.posterior_means[self.modality_names[1]].parameters())
        latent_params += list(self.shared_post_mean.parameters())

        optimizer_dec = optim.Adam(decoder_params, lr=lr_dec)
        optimizer_latent = optim.Adam(latent_params, lr=lr_optim)
        optimizer_enc = optim.Adam(encoder_params, lr=lr_enc)

        dec_loss = self._optimize_latent(trainloader, epochs_dec, optimizer_dec, optimizer_latent)
        enc_loss = self._train_encoder(trainloader, epochs_enc, optimizer_enc)
        self.save_to_checkpoint(seed)

        plot_path = os.path.join("./plots/apollo", dataset_name)
        os.makedirs(plot_path, exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.plot(dec_loss, label="Decoder Loss", color="blue")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Decoder Loss Trend")
        ax1.legend()
        ax2.plot(enc_loss, label="Encoder Loss", color="orange")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.set_title("Encoder Loss Trend")
        ax2.legend()
        plt.savefig(os.path.join(plot_path, "apollo_loss_trend.png"))
        plt.close()
        return dec_loss, enc_loss

    def _optimize_latent(self, trainloader, n_epochs, optimizer_dec, optimizer_latent):
        mse_loss_fn = nn.MSELoss()
        loss_trend = []
        for epoch in range(n_epochs):
            batch_ind = 0
            epoch_loss = []
            for data in trainloader:
                for m_ind, (mod, z) in enumerate(self.posterior_means.items()):
                    x_m = data[m_ind]
                    if isinstance(x_m, list):
                        x_m = torch.stack([x_m[int(data[-1][i])][i] for i in range(len(x_m))], dim=0)
                    batch_size = len(x_m)
                    noise_m = torch.randn(batch_size, self.z_sizes[mod], dtype=torch.float32) * 0.5
                    noise_s = torch.randn(batch_size, self.shared_size, dtype=torch.float32) * 0.5
                    batch_ids = torch.arange(batch_ind, batch_ind + batch_size, device=z.weight.device)
                    z_m = self.posterior_means[mod](batch_ids).float() + noise_m
                    z_s = self.shared_post_mean(batch_ids).float() + noise_s
                    reconst = self.decoders[mod](z_s, z_m)
                    mse_loss = mse_loss_fn(x_m, reconst)
                    act_decay = 1e-3
                    kl_loss = (torch.mean(torch.linalg.norm(z_m, dim=1)) / z_m.size(1) +
                               torch.mean(torch.linalg.norm(z_s, dim=1)) / z_s.size(1)) * act_decay
                    total_loss = mse_loss + kl_loss
                    optimizer_dec.zero_grad()
                    optimizer_latent.zero_grad()
                    total_loss.backward()
                    optimizer_dec.step()
                    optimizer_latent.step()
                    epoch_loss.append(total_loss.item())
                batch_ind += data[0].size(0) if torch.is_tensor(data[0]) else len(data[0])
            loss_trend.append(np.mean(epoch_loss))
            if epoch % 10 == 0:
                print(f"Epoch {epoch} loss: {np.mean(epoch_loss):.4f}")
        return loss_trend

    def _train_encoder(self, trainloader, n_epochs, optimizer):
        loss_trend = []
        for epoch in range(n_epochs):
            batch_ind = 0
            epoch_loss = []
            for batch in trainloader:
                batch_loss = 0
                for m_ind, (mod, encoder) in enumerate(self.encoders.items()):
                    x_m = batch[m_ind]
                    if isinstance(x_m, list):
                        x_m = torch.stack([x_m[int(batch[-1][i])][i] for i in range(len(x_m))], dim=0)
                    batch_size = x_m.size(0)
                    z_s, z_m = encoder(x_m)
                    batch_ids = torch.arange(batch_ind, batch_ind + batch_size, device=z_s.device)
                    shared_mse = (self.shared_post_mean(batch_ids) - z_s).pow(2).mean()
                    mod_mse = (self.posterior_means[mod](batch_ids) - z_m).pow(2).mean()
                    loss = shared_mse + mod_mse
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    batch_loss += loss.item()
                batch_ind += batch[0].size(0) if torch.is_tensor(batch[0]) else len(batch[0])
                epoch_loss.append(batch_loss)
            loss_trend.append(np.mean(epoch_loss))
            if epoch % 10 == 0:
                print(f"Epoch {epoch} loss: {np.mean(epoch_loss):.4f}")
        return loss_trend


def _load_simulated_data(dataset_name):
    if dataset_name == "simulated_apollo":
        data = np.load("./data/simulated_data_apollo.npz")
    else:
        data = np.load("./data/simplest_sim_nongaussian.npz")
    h1, h2, x1, x2, labels = data["h1"], data["h2"], data["x1"], data["x2"], data["labels"]
    n_train = int(0.8 * len(h1))
    n_val = int(0.1 * len(h1))
    return h1, h2, x1, x2, labels, n_train, n_val


def main():
    args = parse_baseline_args("Apollo baseline training/evaluation")
    ckpt_path = args.ckpt_dir or f"./ckpts/apollo/apollo_{args.dataset_name}"
    os.makedirs(ckpt_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset_name in ("simulated", "simulated_apollo"):
        h1, h2, x1, x2, labels, n_train, n_val = _load_simulated_data(args.dataset_name)
        input_dims = [80, 40] if args.dataset_name == "simulated_apollo" else [10, 10]
        shared_rank = specific_rank_1 = specific_rank_2 = 40 if args.dataset_name == "simulated_apollo" else 8
        train_dataset = MultimodalDataset(h1[:n_train], h2[:n_train], x1[:n_train], x2[:n_train], labels[:n_train])
        test_dataset = MultimodalDataset(
            h1[n_train + n_val :], h2[n_train + n_val :],
            x1[n_train + n_val :], x2[n_train + n_val :], labels[n_train + n_val :],
        )
        train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False, drop_last=True)
        epochs_enc = epochs_dec = 500
        task_names = ["shared", "m1", "m2", "joint"] if args.dataset_name == "simulated" else ["shared", "m1", "m2"]
        train_batch = {"A": torch.tensor(h1[:n_train], dtype=torch.float32), "B": torch.tensor(h2[:n_train], dtype=torch.float32)}
        test_batch = {"A": torch.tensor(h1[n_train + n_val :], dtype=torch.float32), "B": torch.tensor(h2[n_train + n_val :], dtype=torch.float32)}
        n_labels = 3 if args.dataset_name == "simulated_apollo" else 4
        prediction_labels = [torch.tensor(labels[:n_train, i], dtype=torch.float32) for i in range(n_labels)]
        prediction_labels_test = [torch.tensor(labels[n_train + n_val :, i], dtype=torch.float32) for i in range(n_labels)]
    elif args.dataset_name == "cremad":
        from scripts.cremad import CremadDataset
        train_dataset = CremadDataset(split="train")
        test_dataset = CremadDataset(split="test")
        train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False, drop_last=True)
        input_dims = [train_dataset.video_dim, train_dataset.audio_dim]
        shared_rank = specific_rank_1 = specific_rank_2 = min(input_dims)
        epochs_enc = epochs_dec = 500
        task_names = ["subject_id", "sentence_id", "emotion", "age", "sex", "race", "ethnicity"]
        prediction_labels, _, h1, h2 = gather_samples(train_dataloader, args.dataset_name)
        prediction_labels_test, _, h1_t, h2_t = gather_samples(test_dataloader, args.dataset_name)
        train_batch = {"A": h1, "B": h2}
        test_batch = {"A": h1_t, "B": h2_t}
    elif args.dataset_name == "urfunny":
        from scripts.urfunny import UrFunnyDataset
        train_dataset = UrFunnyDataset(split="train")
        test_dataset = UrFunnyDataset(split="test")
        train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True, drop_last=False)
        test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=False)
        input_dims = [train_dataset.video_dim, train_dataset.text_dim]
        shared_rank = specific_rank_1 = specific_rank_2 = min(input_dims)
        epochs_enc = epochs_dec = 500
        task_names = ["humor"]
        prediction_labels, _, h1, h2 = gather_samples(train_dataloader, args.dataset_name)
        prediction_labels_test, _, h1_t, h2_t = gather_samples(test_dataloader, args.dataset_name)
        train_batch = {"A": h1, "B": h2}
        test_batch = {"A": h1_t, "B": h2_t}
    elif args.dataset_name == "flickr":
        from scripts.flickr import Multi30KMixedLangDataset
        train_dataset = torch.utils.data.Subset(
            Multi30KMixedLangDataset(split="train"), range(1000)
        )
        test_dataset = Multi30KMixedLangDataset(split="test")
        train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False, drop_last=True)
        input_dims = [768, 768]
        shared_rank = specific_rank_1 = specific_rank_2 = 700
        epochs_enc = epochs_dec = 200
        task_names = ["language"]
        train_batch, prediction_labels = get_flickr_batch(train_dataloader)
        test_batch, prediction_labels_test = get_flickr_batch(test_dataloader)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    hidden_size = 256 if args.dataset_name != "simulated" else 20
    z_sizes = {"A": specific_rank_1, "B": specific_rank_2}
    h_sizes = {"A": input_dims[0], "B": input_dims[1]}
    modality_names = ["A", "B"]
    n_train = len(train_batch["A"])
    encoders = {
        mod: Encoder(h_sizes[mod], shared_rank, z_sizes[mod], hidden_size)
        for mod in modality_names
    }
    decoders = {
        mod: Decoder(h_sizes[mod], shared_rank, z_sizes[mod], hidden_size)
        for mod in modality_names
    }
    model = Apollo(encoders, decoders, n_train=n_train, z_sizes=z_sizes, shared_size=shared_rank,
                  modality_names=modality_names, modality_shapes=h_sizes, ckpt_path=ckpt_path)

    if args.train:
        for i in range(3):
            model.train(train_dataloader, args.dataset_name, epochs_enc=epochs_enc, epochs_dec=epochs_dec, seed=i)

    results_across_seeds = {}
    for i in range(3):
        model = Apollo(encoders, decoders, n_train=n_train, z_sizes=z_sizes, shared_size=shared_rank,
                      modality_names=modality_names, modality_shapes=h_sizes, ckpt_path=ckpt_path)
        model.load_from_checkpoint(i)
        with torch.no_grad():
            z_s_all, z_m_all = model.encode(train_batch)
            z_s_test, z_m_test = model.encode(test_batch)
        components = model.get_components_for_eval(z_s_all, z_m_all)
        components_test = model.get_components_for_eval(z_s_test, z_m_test)
        results_dict = run_evaluation(
            components, components_test, prediction_labels, prediction_labels_test,
            task_names, args.dataset_name,
        )
        results_across_seeds = aggregate_seed_results(results_across_seeds, results_dict)

    print_aggregated_results(results_across_seeds)


if __name__ == "__main__":
    main()
