"""Contrastive baseline: linear heads with NT-Xent loss."""
import sys
import os as _os
_script_dir = _os.path.dirname(_os.path.abspath(__file__))
_project_root = _os.path.dirname(_os.path.dirname(_script_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

from src.baselines.baselines_utils import (
    parse_baseline_args,
    gather_samples,
    get_flickr_batch,
    run_evaluation,
    aggregate_seed_results,
    print_aggregated_results,
)


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim1, dim2, out_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.proj1 = nn.Linear(dim1, out_dim)
        self.proj2 = nn.Linear(dim2, out_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=out_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.fc = nn.Linear(out_dim, out_dim)

    def forward(self, z1, z2):
        z1_proj = self.proj1(z1).unsqueeze(1)
        z2_proj = self.proj2(z2).unsqueeze(1)
        tokens = torch.cat([z1_proj, z2_proj], dim=1)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        return self.fc(attn_out.mean(dim=1))


class AttentionFusion(nn.Module):
    def __init__(self, dim1, dim2, out_dim):
        super().__init__()
        self.proj1 = nn.Linear(dim1, out_dim)
        self.proj2 = nn.Linear(dim2, out_dim)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, z1, z2):
        device = self.alpha.device
        z1 = z1.to(device)
        z2 = z2.to(device)
        w = torch.sigmoid(self.alpha)
        return w * self.proj1(z1) + (1 - w) * self.proj2(z2)


def contrastive_loss(z1, z2, temperature=0.07):
    """NT-Xent / InfoNCE loss."""
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    batch_size = z1.size(0)
    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = torch.matmul(representations, representations.T) / temperature
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z1.device)
    similarity_matrix = similarity_matrix.masked_fill(mask, float("-inf"))
    positives = torch.cat([
        torch.diag(similarity_matrix, batch_size),
        torch.diag(similarity_matrix, -batch_size),
    ])
    negatives = similarity_matrix
    logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
    labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z1.device)
    return nn.CrossEntropyLoss()(logits, labels)


class LinearHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


class ContrastiveModel(nn.Module):
    """Two linear heads for contrastive alignment. encode() returns (z1, z2)."""

    def __init__(self, input_dims, out_dim, device):
        super().__init__()
        self.head1 = LinearHead(input_dims[0], out_dim).to(device)
        self.head2 = LinearHead(input_dims[1], out_dim).to(device)
        self.device = device

    def encode(self, batch):
        """batch: dict with 'A' and 'B' keys, or (h1, h2) tuple."""
        if isinstance(batch, dict):
            h1, h2 = batch["A"], batch["B"]
        else:
            h1, h2 = batch[0], batch[1]
        h1 = h1.to(self.device) if not h1.is_cuda else h1
        h2 = h2.to(self.device) if not h2.is_cuda else h2
        with torch.no_grad():
            z1 = self.head1(h1)
            z2 = self.head2(h2)
        return z1, z2

    def get_components_for_eval(self, z1, z2):
        """Build components for evaluate_predictability (Z1, Z2, Z concat)."""
        z1_np = z1.cpu().numpy()
        z2_np = z2.cpu().numpy()
        return [
            ("Z1", z1_np),
            ("Z2", z2_np),
            ("Z", np.concatenate([z1_np, z2_np], axis=1)),
        ]


def _get_dataloaders(dataset_name):
    """Return (train_loader, test_loader) or (train_data, test_data) for simulated."""
    if dataset_name == "simulated_apollo":
        data = np.load("./data/simulated_data_apollo.npz")
        h1, h2 = data["h1"], data["h2"]
        n_train = int(0.8 * len(h1))
        n_val = int(0.1 * len(h1))
        return (h1[:n_train], h2[:n_train]), (h1[n_train + n_val :], h2[n_train + n_val :])
    elif dataset_name == "simulated":
        data = np.load("./data/simplest_sim_nongaussian.npz")
        h1, h2 = data["h1"], data["h2"]
        n_train = int(0.8 * len(h1))
        n_val = int(0.1 * len(h1))
        return (h1[:n_train], h2[:n_train]), (h1[n_train + n_val :], h2[n_train + n_val :])
    elif dataset_name == "cremad":
        from scripts.cremad import CremadDataset
        train_ds = CremadDataset(split="train")
        test_ds = CremadDataset(split="test")
        train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, drop_last=True)
        return train_loader, test_loader
    elif dataset_name == "flickr":
        from scripts.flickr import Multi30KMixedLangDataset
        train_ds = torch.utils.data.Subset(
            Multi30KMixedLangDataset(split="train"), range(5000)
        )
        test_ds = Multi30KMixedLangDataset(split="test")
        train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, drop_last=True)
        return train_loader, test_loader
    elif dataset_name == "urfunny":
        from scripts.urfunny import UrFunnyDataset
        train_ds = UrFunnyDataset(split="train")
        test_ds = UrFunnyDataset(split="test")
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=False)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, drop_last=False)
        return train_loader, test_loader
    raise ValueError(f"Unknown dataset: {dataset_name}")


def train_contrastive(dataset_name, device, input_dims, out_dim, epochs, lr, seed):
    train_data, test_data = _get_dataloaders(dataset_name)
    ckpt_dir = "./ckpts/contrastive"
    os.makedirs(ckpt_dir, exist_ok=True)

    if dataset_name in ("simulated", "simulated_apollo"):
        h1_train, h2_train = torch.tensor(train_data[0], dtype=torch.float32), torch.tensor(train_data[1], dtype=torch.float32)
        head1 = LinearHead(input_dims[0], out_dim).to(device)
        head2 = LinearHead(input_dims[1], out_dim).to(device)
        optimizer = optim.Adam(list(head1.parameters()) + list(head2.parameters()), lr=lr)
        for epoch in range(epochs):
            head1.train()
            head2.train()
            z1 = head1(h1_train.to(device))
            z2 = head2(h2_train.to(device))
            loss = contrastive_loss(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch {epoch} loss: {loss.item():.4f}")
    else:
        train_loader, _ = train_data, test_data
        for batch in train_loader:
            h1, h2 = batch[0], batch[1]
            if isinstance(h2, list):
                h2 = torch.stack([h2[int(batch[-1][i])][i] for i in range(len(h2))], dim=0)
            in_dim1, in_dim2 = h1.shape[1], h2.shape[1]
            break
        head1 = LinearHead(in_dim1, out_dim).to(device)
        head2 = LinearHead(in_dim2, out_dim).to(device)
        optimizer = optim.Adam(list(head1.parameters()) + list(head2.parameters()), lr=lr)
        for epoch in range(epochs):
            head1.train()
            head2.train()
            for batch in train_loader:
                h1, h2 = batch[0], batch[1]
                if isinstance(h2, list):
                    h2 = torch.stack([h2[int(batch[-1][i])][i] for i in range(len(h2))], dim=0)
                z1 = head1(h1.to(device))
                z2 = head2(h2.to(device))
                loss = contrastive_loss(z1, z2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch {epoch} done")

    torch.save(head1.state_dict(), os.path.join(ckpt_dir, f"contrastive_{dataset_name}_head1_{seed}.pth"))
    torch.save(head2.state_dict(), os.path.join(ckpt_dir, f"contrastive_{dataset_name}_head2_{seed}.pth"))
    print(f"Saved to {ckpt_dir}")


OUTPUT_DIM = {
    "simulated": 10,
    "simulated_apollo": 40,
    "cremad": 768,
    "flickr": 768,
    "urfunny": 700,
}


def main():
    from src.baselines.common import get_baseline_parser
    parser = get_baseline_parser("Contrastive baseline")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_dim", type=int, default=None)
    args = parser.parse_args()
    if args.out_dim is None:
        args.out_dim = OUTPUT_DIM.get(args.dataset_name, 128)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, test_data = _get_dataloaders(args.dataset_name)

    if args.dataset_name in ("simulated", "simulated_apollo"):
        input_dims = [10, 10] if args.dataset_name == "simulated" else [80, 40]
        n_train = int(0.8 * len(train_data[0]))
        n_val = int(0.1 * len(train_data[0]))
        task_names = ["shared", "m1", "m2", "joint"] if args.dataset_name == "simulated" else ["shared", "m1", "m2"]
        data = np.load("./data/simplest_sim_nongaussian.npz" if args.dataset_name == "simulated"
                       else "./data/simulated_data_apollo.npz")
        labels = data["labels"]
        train_batch = {"A": torch.tensor(train_data[0], dtype=torch.float32), "B": torch.tensor(train_data[1], dtype=torch.float32)}
        test_batch = {"A": torch.tensor(test_data[0], dtype=torch.float32), "B": torch.tensor(test_data[1], dtype=torch.float32)}
        n_labels = 4 if args.dataset_name == "simulated" else 3
        prediction_labels = [torch.tensor(labels[:n_train, i], dtype=torch.float32) for i in range(n_labels)]
        prediction_labels_test = [torch.tensor(labels[n_train + n_val :, i], dtype=torch.float32) for i in range(n_labels)]
    elif args.dataset_name == "cremad":
        _, test_loader = train_data, test_data
        task_names = ["emotion"]
        prediction_labels, _, h1, h2 = gather_samples(train_data, args.dataset_name, task_filter=task_names)
        prediction_labels_test, _, h1_t, h2_t = gather_samples(test_loader, args.dataset_name, task_filter=task_names)
        train_batch = {"A": h1, "B": h2}
        test_batch = {"A": h1_t, "B": h2_t}
        input_dims = [h1.shape[1], h2.shape[1]]
    elif args.dataset_name == "urfunny":
        _, test_loader = train_data, test_data
        task_names = ["humor"]
        prediction_labels, _, h1, h2 = gather_samples(train_data, args.dataset_name)
        prediction_labels_test, _, h1_t, h2_t = gather_samples(test_loader, args.dataset_name)
        train_batch = {"A": h1, "B": h2}
        test_batch = {"A": h1_t, "B": h2_t}
        input_dims = [h1.shape[1], h2.shape[1]]
    elif args.dataset_name == "flickr":
        train_loader, test_loader = train_data, test_data
        task_names = ["language"]
        train_batch, prediction_labels = get_flickr_batch(train_loader)
        test_batch, prediction_labels_test = get_flickr_batch(test_loader)
        input_dims = [train_batch["A"].shape[1], train_batch["B"].shape[1]]
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    if args.train:
        for i in range(3):
            train_contrastive(args.dataset_name, device, input_dims, args.out_dim,
                              args.epochs, args.lr, i)

    ckpt_dir = "./ckpts/contrastive"
    results_across_seeds = {}
    for i in range(3):
        model = ContrastiveModel(input_dims, args.out_dim, device)
        model.head1.load_state_dict(torch.load(os.path.join(ckpt_dir, f"contrastive_{args.dataset_name}_head1_{i}.pth")))
        model.head2.load_state_dict(torch.load(os.path.join(ckpt_dir, f"contrastive_{args.dataset_name}_head2_{i}.pth")))
        model.head1.eval()
        model.head2.eval()
        z1_train, z2_train = model.encode(train_batch)
        z1_test, z2_test = model.encode(test_batch)
        components = model.get_components_for_eval(z1_train, z2_train)
        components_test = model.get_components_for_eval(z1_test, z2_test)
        results_dict = run_evaluation(
            components, components_test, prediction_labels, prediction_labels_test,
            task_names, args.dataset_name,
        )
        results_across_seeds = aggregate_seed_results(results_across_seeds, results_dict)

    print_aggregated_results(results_across_seeds)


if __name__ == "__main__":
    main()
