"""DRIM-U baseline: shared/unique factorization with adversarial alignment."""
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
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Function

from src.baselines.baselines_utils import (
    parse_baseline_args,
    gather_samples,
    get_flickr_batch,
    run_evaluation,
    aggregate_seed_results,
    print_aggregated_results,
)

from src.data.base import MultimodalDataset


class MLP_EncShared(nn.Module):
    def __init__(self, input_dim, d=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2 * input_dim),
            nn.ReLU(),
            nn.Linear(2 * input_dim, d),
        )

    def forward(self, x):
        return self.net(x)


class MLP_EncUnique(MLP_EncShared):
    pass


class MLP_DecUnique(nn.Module):
    def __init__(self, input_dim, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2 * input_dim),
            nn.ReLU(),
            nn.Linear(2 * input_dim, output_dim),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grl(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


class DRIM_U(nn.Module):
    """DRIM-U: Disentangling shared and modality-unique representations."""

    def __init__(self, input_dims, d_shared=128, d_unique_1=128, d_unique_2=128,
                 tau=0.1, gamma=0.8, rec_weight=1.0):
        super().__init__()
        self.encoder_s1 = MLP_EncShared(input_dims[0], d_shared)
        self.encoder_m1 = MLP_EncUnique(input_dims[0], d_unique_1)
        self.decoder_m1 = MLP_DecUnique(d_shared + d_unique_1, input_dims[0])
        self.disc1 = Discriminator(d_shared + d_unique_1)
        self.encoder_s2 = MLP_EncShared(input_dims[1], d_shared)
        self.encoder_m2 = MLP_EncUnique(input_dims[1], d_unique_2)
        self.decoder_m2 = MLP_DecUnique(d_shared + d_unique_2, input_dims[1])
        self.disc2 = Discriminator(d_shared + d_unique_2)
        self.tau = tau
        self.gamma = gamma
        self.rec_w = rec_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

    def shared_loss(self, zs1, zs2):
        zs1 = F.normalize(zs1, dim=1)
        zs2 = F.normalize(zs2, dim=1)
        batch_size = zs1.size(0)
        representations = torch.cat([zs1, zs2], dim=0)
        similarity_matrix = torch.matmul(representations, representations.T) / self.tau
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=zs1.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, float("-inf"))
        positives = torch.cat([
            torch.diag(similarity_matrix, batch_size),
            torch.diag(similarity_matrix, -batch_size),
        ])
        negatives = similarity_matrix
        logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=zs1.device)
        return nn.CrossEntropyLoss()(logits, labels)

    def unique_adversarial_loss(self, sm, um, disc, lambd_grl=1.0):
        B = sm.size(0)
        z_joint = torch.cat([sm, um], dim=1)
        perm = torch.randperm(B, device=sm.device)
        z_prod = torch.cat([sm[perm], um], dim=1)
        y_joint = torch.ones(B, device=sm.device)
        y_prod = torch.zeros(B, device=sm.device)
        logits_joint = disc(grl(z_joint, lambd=lambd_grl))
        logits_prod = disc(grl(z_prod, lambd=lambd_grl))
        return self.bce(logits_joint, y_joint) + self.bce(logits_prod, y_prod)

    def encode(self, batch):
        """Encode batch dict or (h1, h2) tuple -> (zs1, zm1, zs2, zm2)."""
        device = next(self.parameters()).device
        if isinstance(batch, dict):
            h1, h2 = batch["A"], batch["B"]
        else:
            h1, h2 = batch[0], batch[1]
        if isinstance(h2, list):
            h2 = torch.stack([h2[int(batch[-1][i])][i] for i in range(len(h2))], dim=0)
        if not isinstance(h1, torch.Tensor):
            h1 = torch.tensor(h1, dtype=torch.float32)
        if not isinstance(h2, torch.Tensor):
            h2 = torch.tensor(h2, dtype=torch.float32)
        h1, h2 = h1.float(), h2.float()
        h1, h2 = h1.to(device), h2.to(device)
        zs1 = self.encoder_s1(h1)
        zm1 = self.encoder_m1(h1)
        zs2 = self.encoder_s2(h2)
        zm2 = self.encoder_m2(h2)
        return zs1, zm1, zs2, zm2

    def get_components_for_eval(self, zs1, zm1, zs2, zm2):
        """Build components for evaluate_predictability."""
        zs = (zs1 + zs2) / 2
        return [
            ("Zs", zs.detach().cpu().numpy()),
            ("Zm1", zm1.detach().cpu().numpy()),
            ("Zm2", zm2.detach().cpu().numpy()),
            ("Z", np.concatenate([
                zm1.detach().cpu().numpy(),
                zm2.detach().cpu().numpy(),
                zs.detach().cpu().numpy(),
            ], axis=1)),
        ]

    def forward(self, h1, h2):
        zs1 = self.encoder_s1(h1)
        zm1 = self.encoder_m1(h1)
        xhat_m1 = self.decoder_m1(torch.cat([zs1, zm1], dim=1))
        zs2 = self.encoder_s2(h2)
        zm2 = self.encoder_m2(h2)
        xhat_m2 = self.decoder_m2(torch.cat([zs2, zm2], dim=1))
        Lr = self.mse(xhat_m1, h1) + self.mse(xhat_m2, h2)
        Lu_m1 = self.unique_adversarial_loss(zs1, zm1, self.disc1)
        Lu_m2 = self.unique_adversarial_loss(zs2, zm2, self.disc2)
        Lsh = self.shared_loss(zs1, zs2)
        Lu = Lu_m1 + Lu_m2
        return Lr, Lsh, Lu

    def training_step(self, h1, h2, weights=(1.0, 1.0, 0.8)):
        w_rec, w_sh, w_u = weights
        Lr, Lsh, Lu = self(h1, h2)
        loss = w_rec * Lr + w_sh * Lsh + w_u * Lu
        logs = {"L_rec": Lr.item(), "L_sh": Lsh.item(), "L_u": Lu.item(), "L_total": loss.item()}
        return loss, logs


def train_loop(model, train_loader, device, dataset_name, epochs, lr, ckpt_dir, seed):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{dataset_name}_drim_latest_{seed}.pth")
    for epoch in range(1, epochs + 1):
        meter = {"L_rec": 0.0, "L_sh": 0.0, "L_u": 0.0, "L_total": 0.0}
        steps = 0
        model.train()
        for batch in train_loader:
            h1, h2 = batch[0], batch[1]
            if isinstance(h2, list):
                h2 = torch.stack([h2[int(batch[-1][i])][i] for i in range(len(h1))], dim=0)
            h1, h2 = h1.to(device), h2.to(device)
            loss, logs = model.training_step(h1, h2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for k, v in logs.items():
                meter[k] += float(v)
            steps += 1
        if steps:
            for k in meter:
                meter[k] /= steps
        print(f"[Epoch {epoch:03d}] Lrec={meter['L_rec']:.4f} Lsh={meter['L_sh']:.4f} "
              f"Lu={meter['L_u']:.4f} Ltot={meter['L_total']:.4f}")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, ckpt_path)


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
    args = parse_baseline_args("DRIM-U baseline")
    ckpt_dir = args.ckpt_dir or "./drim_ckpts"
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
        train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False)
        n_epochs = 600 if args.dataset_name == "simulated_apollo" else 500
        task_names = ["shared", "m1", "m2", "joint"] if args.dataset_name == "simulated" else ["shared", "m1"]
        train_batch = {"A": torch.tensor(h1[:n_train], dtype=torch.float32), "B": torch.tensor(h2[:n_train], dtype=torch.float32)}
        test_batch = {"A": torch.tensor(h1[n_train + n_val :], dtype=torch.float32),
                      "B": torch.tensor(h2[n_train + n_val :], dtype=torch.float32)}
        n_labels = 4 if args.dataset_name == "simulated" else 2
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
        n_epochs = 500
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
        test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False, drop_last=False)
        input_dims = [train_dataset.video_dim, train_dataset.text_dim]
        shared_rank = specific_rank_1 = specific_rank_2 = min(input_dims)
        n_epochs = 500
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
        shared_rank = specific_rank_1 = specific_rank_2 = min(input_dims)
        n_epochs = 400
        task_names = ["language"]
        train_batch, prediction_labels = get_flickr_batch(train_dataloader)
        test_batch, prediction_labels_test = get_flickr_batch(test_dataloader)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    model = DRIM_U(
        input_dims=input_dims,
        d_shared=shared_rank,
        d_unique_1=specific_rank_1,
        d_unique_2=specific_rank_2,
        tau=0.1,
        gamma=0.8,
    ).to(device)

    if args.train:
        for i in range(3):
            train_loop(model, train_dataloader, device, args.dataset_name, n_epochs, 2e-4, ckpt_dir, i)

    results_across_seeds = {}
    for i in range(3):
        ckpt_path = os.path.join(ckpt_dir, f"{args.dataset_name}_drim_latest_{i}.pth")
        if not os.path.exists(ckpt_path):
            print(f"No checkpoint at {ckpt_path}, skipping evaluation.")
            continue
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict({k: v.float() for k, v in state["model"].items()})
        model.eval()
        with torch.no_grad():
            zs1, zm1, zs2, zm2 = model.encode(train_batch)
            zs1_t, zm1_t, zs2_t, zm2_t = model.encode(test_batch)
        components = model.get_components_for_eval(zs1, zm1, zs2, zm2)
        components_test = model.get_components_for_eval(zs1_t, zm1_t, zs2_t, zm2_t)
        results_dict = run_evaluation(
            components, components_test, prediction_labels, prediction_labels_test,
            task_names, args.dataset_name,
        )
        results_across_seeds = aggregate_seed_results(results_across_seeds, results_dict)

    if results_across_seeds:
        print_aggregated_results(results_across_seeds)


if __name__ == "__main__":
    main()
