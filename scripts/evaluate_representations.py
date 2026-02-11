import sys
import os as _os
_script_dir = _os.path.dirname(_os.path.abspath(__file__))
_project_root = _os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
import numpy as np
from sklearn.linear_model import LinearRegression
import clip
import math
import argparse
import random
from PIL import Image
import torch.nn as nn
import torch.optim as optim

from src.utils import load_checkpoint, SklearnTrainer, plot_losses
from src.multimodal_projector import MultiLoReFT
from src.data.base import MultimodalDataset
from configs import get_dataset_config, get_checkpoint_path
from scripts.flickr import Multi30KMixedLangDataset
from scripts.vqa import VQADataset
from scripts.cremad import CremadDataset
from scripts.urfunny import UrFunnyDataset


class MultiplicativeInteractions(nn.Module):
    def __init__(self, dim1, dim2, out_dim, mode="matrix"):
        """
        Multiplicative Interaction Fusion
        Args:
            dim1 (int): input dimension of z1
            dim2 (int): input dimension of z2
            out_dim (int): output dimension
            mode (str): one of ["matrix", "vector", "scalar"]
        """
        super().__init__()
        self.mode = mode.lower()

        if self.mode == "matrix":
            # full bilinear interaction
            self.W = nn.Parameter(torch.randn(dim1, dim2, out_dim) * 0.02)
        elif self.mode == "vector":
            # diagonal approximation (elementwise modulation)
            self.W = nn.Parameter(torch.randn(max(dim1, dim2), out_dim) * 0.02)
        elif self.mode == "scalar":
            # scalar gate
            self.W = nn.Parameter(torch.randn(out_dim) * 0.02)
        else:
            raise ValueError("mode must be one of ['matrix', 'vector', 'scalar']")

        self.U = nn.Linear(dim1, out_dim, bias=False)
        self.V = nn.Linear(dim2, out_dim, bias=False)
        self.b = nn.Parameter(torch.zeros(out_dim))

    def forward(self, z1, z2):
        """
        z1: [batch, dim1]
        z2: [batch, dim2]
        """
        device = self.W.device
        z1 = z1.to(device)
        z2 = z2.to(device)
        # self.W is already a parameter on the correct device
        # self.U, self.V, self.b are modules/parameters and will be on the same device as the model
        if self.mode == "matrix":
            # bilinear form: z1ᵀ W z2
            # einsum: (batch, dim1), (dim1, dim2, out) , (batch, dim2) -> (batch, out)
            bilinear = torch.einsum('bi,ijk,bj->bk', z1, self.W, z2)
        elif self.mode == "vector":
            # feature-wise product
            max_dim = max(z1.size(1), z2.size(1))
            z1_padded = torch.nn.functional.pad(z1, (0, max_dim - z1.size(1)))
            z2_padded = torch.nn.functional.pad(z2, (0, max_dim - z2.size(1)))
            bilinear = (z1_padded * z2_padded) @ self.W
        elif self.mode == "scalar":
            # dot product then scale
            min_dim = min(z1.size(1), z2.size(1))
            dot = torch.sum(z1[:, :min_dim] * z2[:, :min_dim], dim=1, keepdim=True)  # [batch, 1]
            bilinear = dot * self.W.unsqueeze(0)  # broadcast to [batch, out]
        else:
            raise ValueError

        return bilinear + self.U(z1) + self.V(z2) + self.b  


class AlignmentHead(nn.Module):
    """
    Lightweight head to align embeddings from one modality (h0) to another (h1).
    By default it's a single linear layer, but can be extended with nonlinearity.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            # simplest: linear map
            self.net = nn.Linear(input_dim, output_dim)
        else:
            # slightly richer: 1-hidden MLP
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )

    def forward(self, x):
        return self.net(x)

class AttentionFusion(nn.Module):
    def __init__(self, dim1, dim2, out_dim, num_heads=4, dropout=0.1):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads

        # smallest multiple of num_heads ≥ out_dim
        self.attn_dim = math.ceil(out_dim / num_heads) * num_heads

        self.proj1 = nn.Linear(dim1, self.attn_dim)
        self.proj2 = nn.Linear(dim2, self.attn_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=self.attn_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # project back to desired out_dim
        self.fc = nn.Linear(self.attn_dim, out_dim)

    def forward(self, z1, z2):
        z1 = self.proj1(z1).unsqueeze(1)  # [B,1,attn_dim]
        z2 = self.proj2(z2).unsqueeze(1)  # [B,1,attn_dim]

        tokens = torch.cat([z1, z2], dim=1)  # [B,2,attn_dim]

        attn_out, _ = self.attn(tokens, tokens, tokens)

        fused = attn_out.mean(dim=1)

        return self.fc(fused)

def train_alignment_head(h0, h1, device="cuda", epochs=50, lr=1e-3, hidden_dim=None):
    """
    h0: [N, D0] embeddings from modality 0 (torch.Tensor)
    h1: [N, D1] embeddings from modality 1 (torch.Tensor)
    """
    h0, h1 = h0.detach().to(device), h1.detach().to(device)
    model = AlignmentHead(h0.shape[1], h1.shape[1], hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(h0)
        loss = loss_fn(pred, h1)
        loss.backward()
        optimizer.step()

    # After training, evaluate correlation
    with torch.no_grad():
        pred = model(h0)
        cos_sim = torch.nn.functional.cosine_similarity(pred, h1, dim=1)
        mean_cos = cos_sim.mean().item()

    return model, mean_cos


class FixedProjector(torch.nn.Module):
    def __init__(self, d_in, k, ortho=True, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        W = torch.randn(d_in, k, generator=g) / (d_in**0.5)
        if ortho:
            Q, _ = torch.linalg.qr(W, mode='reduced')
            W = Q
        self.register_buffer('W', W, persistent=False)

    def forward(self, x):
        return x @ self.W  # [B,k]         

def find_closest_samples(z_space, z, space_name, k=5):
    similarities = torch.nn.functional.cosine_similarity(z.unsqueeze(0), z_space, dim=1)
    closest_indices = torch.topk(similarities, k=k).indices
    print(f"Closest samples in {space_name} space:", closest_indices.tolist())
    return closest_indices

def plot_closest_images(images_all, reference_image, closest_indices, filename):
    fig, axes = plt.subplots(1, 6, figsize=(20, 4))
    ref_img = np.transpose(reference_image, (1, 2, 0))
    ref_img = (ref_img * 0.5) + 0.5  # Denormalize
    axes[0].imshow(ref_img)
    axes[0].set_title('Reference Image')
    axes[0].axis('off')
    for i, idx in enumerate(closest_indices):
        img = np.transpose(images_all[idx], (1, 2, 0))
        img = (img * 0.5) + 0.5  # Denormalize
        axes[i+1].imshow(img)
        axes[i+1].set_title(f'Match {i+1}')
        axes[i+1].axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

class SimilarityMLP(torch.nn.Module):
    def __init__(self, dim1, dim2, hidden_dim=256):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(dim1, dim2),
        )
    def forward(self, x1):
        score = self.fc(x1)
        return score

def evaluate_cross_modal_retrieval(h0, h1, device, batch_size=512, similarity_model=None, k=10, components_train=None):
    """
    Batched version to evaluate cross-modal retrieval with learned similarity.
    similarity_model: a model taking (query, gallery) → score
    """
    def match_shapes(a, b):
        if a.shape[1] != b.shape[1]:
            proj_dim = max(a.shape[1], b.shape[1])
            if a.shape[1] < proj_dim:
                a = FixedProjector(a.shape[1], k=proj_dim, seed=123).to(a.device)(a)
            if b.shape[1] < proj_dim:
                b = FixedProjector(b.shape[1], k=proj_dim, seed=223).to(b.device)(b)
        return a, b

    h0, h1 = match_shapes(h0, h1)
    h0 = h0.to(device)
    h1 = h1.to(device)
    def recall_at_k_batched(query_set, gallery_set, k=10):
        query_set, gallery_set = match_shapes(query_set, gallery_set)
        correct_count = 0
        num_samples = query_set.shape[0]

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_query = query_set[start:end]
            gallery_query = gallery_set[start:end]
            batch_query = torch.nn.functional.normalize(batch_query, dim=1)
            gallery_set = torch.nn.functional.normalize(gallery_set, dim=1)
            sim_matrix = torch.nn.functional.cosine_similarity(batch_query.unsqueeze(1), gallery_query, dim=2)
            topk = sim_matrix.topk(k, dim=1).indices
            true_matches = torch.arange(0, len(gallery_query), device=device).unsqueeze(1).expand(-1, k)
            correct = (topk == true_matches).any(dim=1).float()
            correct_count += correct.sum().item()

        return correct_count / num_samples
    return recall_at_k_batched(h0, h1, k)


def evaluate_predictability(components, labels, task_name, dataset_name, components_test=None, labels_test=None):
    """Evaluate how well each component (shared and modality-specific) can predict the target label.
    """
    # Determine if this is a classification or regression task
    if labels_test is not None:
        y = labels.detach().cpu().numpy() if hasattr(labels, "detach") else np.array(labels)
        y_test = labels_test.detach().cpu().numpy() if hasattr(labels_test, "detach") else np.array(labels_test)
    else:
        y = labels.detach().cpu().numpy() if hasattr(labels, "detach") else np.array(labels)
        y_test = None
    
    unique_values = np.unique(y)
    n_unique = len(unique_values)
    
    # Handle edge cases
    if n_unique == 1:
        print(f"Warning: Label has only one unique value. Skipping evaluation.")
        return
    
    # Determine task type based on both number of unique values and their nature
    is_classification = (n_unique <= 20 and 
                        np.all(np.mod(y, 1) == 0))  # Check if all values are integers
    
    if is_classification:
        n_classes = len(np.unique(y))
        metric_name = ["roc_auc_ovr", "silhouette_score"]

        if n_classes == 2:
            task_type = "binary"
        else:
            task_type = "multiclass"
    else:
        if y.ndim > 1 and y.shape[1] > 1:
            model = None
            task_type = "neural_multihead"
            metric_name = ["MSE"]
        else:
            model = LinearRegression()
            task_type = "regression"
            metric_name = ["MSE"]
    
    performance_scores = []
    component_names = []
    results_dict = {}
    for ind, (name, z) in enumerate(components):
        z = z.detach().cpu().numpy() if torch.is_tensor(z) else z
        if components_test is not None:
            z_test = components_test[ind][1].detach().cpu().numpy() if torch.is_tensor(components_test[ind][1]) else components_test[ind][1]
        else:
            z_test = None
        try:
            reg_model = SklearnTrainer(task_type=task_type)
            if task_type in ["multiclass", "binary"]:
                scores, score_1s = reg_model.train_and_evaluate(z, y, k=5, z_test=z_test, y_test=y_test)
                results_dict[name+'_acc'] = scores
                results_dict[name+'_silhouette'] = score_1s
            else:
                scores = reg_model.train_and_evaluate(z, y, k=5, z_test=z_test, y_test=y_test)
                results_dict[name+'_mse'] = scores
            performance_scores.append((np.mean(scores), np.var(scores)))
            component_names.append(name)
        except Exception as e:
            print(f"Error evaluating {name}: {str(e)}")
            continue
    return results_dict

def load_components(dataloader, projection_model, dataset_name, device):
    labels, labels_2 = [], []
    z1s, z2s, z1m, z2m = [], [], [], []
    phi_1, phi_2 = [], []
    h1, h2, z = [], [], []
    random_sample = random.randint(0, 1000)
    x2_all, x1_all = [], []
    with torch.no_grad():
        count = 0
        for i, batch in enumerate(dataloader):
            if dataset_name == "flickr":
                image_feats, caption_feats = batch[0], batch[1]
                lang_idx = batch[-1]
                captions = batch[3]
                x1 = batch[2]
                text_feats = torch.stack([caption_feats[0], caption_feats[1]], dim=1).gather(
                    1, lang_idx.unsqueeze(1).unsqueeze(2).expand(-1, -1, caption_feats[0].shape[-1])
                ).squeeze(1)
                l2 = torch.stack([caption_feats[0], caption_feats[1]], dim=1).gather(
                    1, abs(1-lang_idx).unsqueeze(1).unsqueeze(2).expand(-1, -1, caption_feats[0].shape[-1])).squeeze(1)
                x2 = [captions[0][i] if idx == 0 else captions[1][i] for i, idx in enumerate(lang_idx)]
                l1 = lang_idx
                label = [l1]#, l2]
                h1.append(image_feats)
                h2.append(text_feats)
                task_names = ['language']#, 'other_caption']
            elif dataset_name=="cremad":
                video_feats, audio_feats, x1, x2, subject_id, sentence_id, emotion, age, sex, race, ethnicity = batch
                sentence_refs = ['IEO', 'TIE', 'IOM', 'IWW', 'TAI', 'MTI', 'IWL', 'ITH', 'DFA', 'ITS', 'TSI', 'WSI']
                emotion_refs = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']
                subject_id = torch.Tensor([int(id) for id in subject_id])
                sentence_id = torch.Tensor([sentence_refs.index(id) for id in sentence_id])
                emotion = torch.Tensor([emotion_refs.index(id) for id in emotion])
                h1.append(video_feats)
                h2.append(audio_feats)
                label = [subject_id, sentence_id, emotion, age, sex, race, ethnicity]
                task_names = ['subject_id', 'sentence_id', 'emotion', 'age', 'sex', 'race', 'ethnicity']
            elif dataset_name == "vqa":
                image_feats, question_feats, x1, x2, answer, answer_feat = batch
                h1.append(image_feats)
                h2.append(question_feats)
                label = [answer_feat]
                task_names = ['answer']
            elif dataset_name=="urfunny":
                feats = batch
                h1.append(feats[0])
                h2.append(feats[1])
                label = [feats[-1]]
                task_names = ['humor']
                x1, x2 = feats[2], feats[3]
            phis = projection_model([h1[-1].to(device), h2[-1].to(device)])
            z.append(projection_model.fuse_representations(phis))
            z_n = projection_model.decouple(phis, full=True)
            z1s.append(torch.Tensor(z_n[0][1]))
            z2s.append(torch.Tensor(z_n[1][1]))
            z1m.append(torch.Tensor(z_n[0][0]))
            z2m.append(torch.Tensor(z_n[1][0]))
            x2_all.append(x2)
            # labels_2.append(l2)
            x1_all.append(x1)
            for i, lbl in enumerate(label):
                if len(labels) <= i:
                    labels.append([])
                labels[i].append(lbl)
            phi_1.append(phis[0])
            phi_2.append(phis[1])
        z1s = torch.cat(z1s, dim=0)
        z2s = torch.cat(z2s, dim=0)
        z1m = torch.cat(z1m, dim=0)
        z2m = torch.cat(z2m, dim=0)
        h1 = torch.cat(h1, dim=0)
        h2 = torch.cat(h2, dim=0)
        z = torch.cat(z, dim=0)
        labels = [torch.cat(label_list, dim=0).unsqueeze(-1) for label_list in labels]
        phi_1 = torch.cat(phi_1, dim=0)
        phi_2 = torch.cat(phi_2, dim=0)
        

        if dataset_name == "flickr" or dataset_name == "urfunny":
            x2_all = np.concatenate(x2_all, axis=0)
            x1_all = np.concatenate([img.cpu().numpy() for img in x1_all], axis=0)
            random_sample = random.randint(0, len(x1_all))
            random_caption = x2_all[random_sample]
            random_image = x1_all[random_sample]
            closest_images_shared = find_closest_samples(z1s, z1s[random_sample], "image shared")
            closest_images_modality_specific = find_closest_samples(z1m, z1m[random_sample], "image modality-specific")
            closest_captions_shared = find_closest_samples(z2s, z2s[random_sample], "caption shared")
            closest_captions_modality_specific = find_closest_samples(z2m, z2m[random_sample], "caption modality-specific")
            print("Reference caption: ", x2_all[random_sample])
            print("Closest captions in modality-specific space:")
            for ind in closest_captions_modality_specific:
                print(x2_all[ind])
            print("Closest captions in shared space:")
            for ind in closest_captions_shared:
                print(x2_all[ind])
            if dataset_name == "flickr":
                plot_closest_images(x1_all, x1_all[random_sample], closest_images_modality_specific, './plots/%s/closest_images_modality_specific.png' % dataset_name)
                plot_closest_images(x1_all, x1_all[random_sample], closest_images_shared, './plots/%s/closest_shared_space.png' % dataset_name)
    return z1s, z1m, z2s, z2m, h1, h2, z, labels, phi_1, phi_2, x2_all, x1_all, task_names

def plot_representations(z_1, z_2, z_s, labels, task_name, dataset_name, save_dir="./plots", modality_names=["I", "II"]):
    """Plot 2D PCA projections of the representations colored by each label.
    
    Args:
        z_1: Modality-specific representation for modality 1
        z_2: Modality-specific representation for modality 2
        z_s: Shared representation
        labels: Target labels
        task_name: Name of the task
        dataset_name: Name of the dataset
        save_dir: Directory to save the plots
        modality_names: Names of the modalities
    """
    fig, axs = plt.subplots(3, 1, figsize=(5, 15))
    titles = [f'Modality-specific {modality_names[0]}', f'Shared', f'Modality-specific {modality_names[1]}']
    
    for ind, title in enumerate(titles):
        ax = axs[ind]
        ax.set_title(title, fontsize=24)
        if ind == 0:
            data = z_1
        elif ind == 1:
            data = z_s
        else:
            data = z_2
        
        if data.shape[1] >= 2:
            reducer = PCA(n_components=2)
            x = reducer.fit_transform(data)
            ax.scatter(x[:, 0], x[:, 1], c=labels.cpu().numpy() if torch.is_tensor(labels) else labels)
        elif data.shape[1] == 1:
            # Handle 1D case by adding a zero column for visualization
            x = np.hstack([data, np.zeros_like(data)])
            ax.scatter(x[:, 0], x[:, 1], c=labels.cpu().numpy() if torch.is_tensor(labels) else labels)
        else:
            # If no valid features, just write "No Data"
            ax.text(0.5, 0.5, "No Data", horizontalalignment='center', verticalalignment='center')

    plt.savefig(f"{save_dir}/{dataset_name}/test_{task_name}.pdf")
    print(f"Saved plot to {save_dir}/{dataset_name}/test_{task_name}.pdf")


def plot_projection_matrices(model, dataset_name, threshold=0.00, save_dir="./plots"):
    """Plot the learned projection matrices and their correlations.
    
    Args:
        model: Trained ProjectionModule
        dataset_name: Name of the dataset
        threshold: Threshold for singular values
        save_dir: Directory to save the plots
    """
    shared_sv = torch.where(torch.linalg.svdvals(model.R_s1) > threshold)
    Rs_1 = model.R_s1[shared_sv].detach().cpu().numpy()
    m1_sv = torch.where(torch.linalg.svdvals(model.R_m1) > threshold)
    Rm1 = model.R_m1[m1_sv].detach().cpu().numpy()
    m2_sv = torch.where(torch.linalg.svdvals(model.R_m2) > threshold)
    Rm2 = model.R_m2[m2_sv].detach().cpu().numpy()

    # Plot matrices
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    matrices = [Rs_1, Rm1, Rm2]
    titles = ["Shared Projection (R_s1)", "Modality-Specific (R_m1)", "Modality-Specific (R_m2)"]
    
    # Find the maximum absolute value for symmetric color range
    max_abs = max(abs(Rs_1).max(), abs(Rm1).max(), abs(Rm2).max())
    vmin, vmax = -max_abs, max_abs

    # Plot matrices
    for ax, matrix, title in zip(axs, matrices, titles):
        sns.heatmap(matrix, ax=ax, cmap="RdBu_r", cbar=True, vmin=vmin, vmax=vmax, center=0)
        if "R_s1" in title:
            sv = torch.linalg.svdvals(model.R_s1).detach().cpu().numpy()
        elif "R_m1" in title:
            sv = torch.linalg.svdvals(model.R_m1).detach().cpu().numpy()
        else:
            sv = torch.linalg.svdvals(model.R_m2).detach().cpu().numpy()
        ax.set_title(f"{title}\nSVs: {sv[min(len(sv)-1, 10)]}")
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{dataset_name}/learned_matrices.pdf")
    plt.close()

    # Plot correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    names = ['R_s1', 'R_m1', 'R_m2']
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
    ax.set_title('Correlation between Projection Matrices', fontsize=20)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{dataset_name}/matrix_correlations.pdf")
    plt.close()


def main(dataset_name, checkpoint_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    components_train = None
    cfg = get_dataset_config(dataset_name)
    
    if "simulated" in dataset_name:
        loaded_data = np.load(cfg["data_path"])
        task_names = cfg["task_names"]
        modality_names = cfg["modality_names"]
        input_dims = cfg["input_dims"]
        h1 = loaded_data["h1"]
        h2 = loaded_data["h2"]
        x1 = loaded_data["x1"]
        x2 = loaded_data["x2"]
        n_train = int(0.8*len(h1))
        n_val = int(0.1*len(h1))
        labels = loaded_data["labels"][:n_train]
        labels_test = loaded_data["labels"][n_train+n_val:]
        # Create dataset
        train_dataset = MultimodalDataset(h1[:n_train], h2[:n_train], x1[:n_train], x2[:n_train], labels[:n_train])  
        test_dataset = MultimodalDataset(h1[n_train+n_val:], h2[n_train+n_val:], x1[n_train+n_val:], x2[n_train+n_val:], labels[n_train+n_val:])  
        # Load model
        projection_model = MultiLoReFT(
            input_dims=input_dims, 
            shared_rank=cfg["shared_rank"], 
            specific_rank=cfg["specific_rank"], 
            pruning_threshold=cfg.get("pruning_threshold", 0.2),
            staging=True,
            pruning=True,
            device=device,
            shared_R_mode="pad"
        ).to(device)
        projection_model = load_checkpoint(filepath=checkpoint_name, model=projection_model)
        projection_model.eval()
        projection_model = projection_model.to(device)
        # Get representations
        h1_test = torch.Tensor(h1[n_train+n_val:]).to(device)
        h2_test = torch.Tensor(h2[n_train+n_val:]).to(device)
        phis_test = projection_model([h1_test,h2_test])
        phi_1_test = phis_test[0]
        phi_2_test = phis_test[1]
        z_test = projection_model.fuse_representations(phis_test)
        z_n_test = projection_model.decouple(phis_test, full=True, th=0.05)
        z1m_test, z1s_test, z2m_test, z2s_test = z_n_test[0][0], z_n_test[0][1], z_n_test[1][0], z_n_test[1][1]
        h1 = torch.Tensor(h1[:n_train]).to(device)
        h2 = torch.Tensor(h2[:n_train]).to(device)
        phis = projection_model([h1,h2])
        phi_1 = phis[0]
        phi_2 = phis[1]
        z = projection_model.fuse_representations(phis)
        z_n = projection_model.decouple(phis, full=True, th=0.05)
        z1m, z1s, z2m, z2s = z_n[0][0], z_n[0][1], z_n[1][0], z_n[1][1]
        prediction_labels = [labels[:,i] for i in range(len(task_names))]
        prediction_labels_test = [labels_test[:,i] for i in range(len(task_names))]
    else:
        modality_names = cfg["modality_names"]
        if dataset_name == "flickr":
            test_dataset = Multi30KMixedLangDataset(split="test", device=device, return_raw=True)
            train_dataset = Multi30KMixedLangDataset(split="train", device=device, return_raw=True)
            input_dims = cfg["input_dims"]
            projection_model = MultiLoReFT(
                input_dims=input_dims,
                shared_rank=cfg["shared_rank"],
                specific_rank=cfg["specific_rank"],
                pruning_threshold=cfg["pruning_threshold"],
                device=device,
                staging=True,
                pruning=True,
                dataset_name="flickr",
                shared_R_mode="pad"
            ).to(device)
            bs = cfg["batch_size"]
            train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=False)
            test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
        elif dataset_name == "vqa":
            test_dataset = VQADataset(split="validation", device=device)
            test_dataset = torch.utils.data.Subset(test_dataset, range(1000))
            train_dataset = VQADataset(split="train", device=device)
            train_dataset = torch.utils.data.Subset(train_dataset, range(1000))
            input_dims = cfg["input_dims"]
            projection_model = MultiLoReFT(
                input_dims=input_dims,
                shared_rank=cfg["shared_rank"],
                specific_rank=cfg["specific_rank"],
                device=device
            )
            bs = cfg.get("batch_size", 256)
            train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=False)
            test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
        elif dataset_name == "cremad":
            test_dataset = CremadDataset(split="test")
            train_dataset = CremadDataset(split="train")
            input_dims = [train_dataset.video_dim, train_dataset.audio_dim]
            projection_model = MultiLoReFT(
                input_dims=input_dims,
                shared_rank=cfg["shared_rank"],
                specific_rank=cfg["specific_rank"],
                pruning_threshold=cfg["pruning_threshold"],
                device=device,
                staging=True,
                pruning=True,
                dataset_name="cremad",
                shared_R_mode="pad"
            ).to(device)
            bs = cfg["batch_size"]
            train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=False)
            test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
        elif dataset_name == "urfunny":
            test_dataset = UrFunnyDataset(split="test", device=device)
            train_dataset = UrFunnyDataset(split="train", device=device)
            input_dims = [train_dataset.video_dim, train_dataset.text_dim]
            shared_rank = cfg.get("shared_rank") or input_dims[0]
            specific_rank = cfg.get("specific_rank") or input_dims[1]
            projection_model = MultiLoReFT(
                input_dims=input_dims,
                shared_rank=shared_rank,
                specific_rank=specific_rank,
                pruning_threshold=cfg["pruning_threshold"],
                device=device,
                staging=True,
                pruning=True,
                dataset_name="urfunny",
                shared_R_mode="pad"
            ).to(device)
            bs = cfg["batch_size"]
            test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
            train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=False)
        projection_model = load_checkpoint(filepath=checkpoint_name, model=projection_model)
        projection_model.eval()
        projection_model = projection_model.to(device)
        z1s, z1m, z2s, z2m, h1, h2, z, prediction_labels, phi_1, phi_2, x2_all, x1_all, task_names = load_components(train_dataloader, projection_model, dataset_name, device)
        z1s_test, z1m_test, z2s_test, z2m_test, h1_test, h2_test, z_test, prediction_labels_test, phi_1_test, phi_2_test, _, _, _ = load_components(test_dataloader, projection_model, dataset_name, device)

    # Evaluate and plot
    for name, param in projection_model.named_parameters():
        if 'R' in name:
            u, s, v = torch.svd(param)
            print(f"Singular values of {name}: {s}")
    plot_projection_matrices(projection_model, dataset_name=dataset_name)

    components = [
        ("Zs", (z1s+z2s).detach().cpu().numpy()/2),
        ("Zm1", z1m.detach().cpu().numpy()),
        ("Zm2", z2m.detach().cpu().numpy()),
        ("Z", torch.concat([z1m, z2m, (z1s+z2s)/2], dim=1).detach().cpu().numpy()),
        ("H1", h1.detach().cpu().numpy()),
        ("H2", h2.detach().cpu().numpy()),
        ("Phi1", phi_1.detach().cpu().numpy()),
        ("Phi2", phi_2.detach().cpu().numpy()),
        ("concatenated phis", torch.concat([phi_1, phi_2], dim=1).detach().cpu().numpy()),
        ("concatenated raw", torch.concat([h1, h2], dim=1).detach().cpu().numpy()),
    ] 
    components_test = [
        ("Zs", (z1s_test+z2s_test).detach().cpu().numpy()/2),
        ("Zm1", z1m_test.detach().cpu().numpy()),  # Modality-specific representation from modality 1
        ("Zm2", z2m_test.detach().cpu().numpy()),
        ("Z", torch.concat([z1m_test, z2m_test, (z1s_test+z2s_test)/2], dim=1).detach().cpu().numpy()),
        ("H1", h1_test.detach().cpu().numpy()),
        ("H2", h2_test.detach().cpu().numpy()),
        ("Phi1", phi_1_test.detach().cpu().numpy()),
        ("Phi2", phi_2_test.detach().cpu().numpy()),
        ("concatenated phis", torch.concat([phi_1_test, phi_2_test], dim=1).detach().cpu().numpy()),
        ("concatenated raw", torch.concat([h1_test, h2_test], dim=1).detach().cpu().numpy()),
    ] 

    if args.baselines:
        max_dim = max(input_dims[0], input_dims[1])
        attn_fuser = AttentionFusion(input_dims[0], input_dims[1], input_dims[0]+input_dims[1]).to(device)
        attn_fuser.eval()
        with torch.no_grad():
            z_fused_raw = attn_fuser(
                torch.tensor(h1.clone().detach()).to(device),
                torch.tensor(h2).to(device)
            )
        components.append(("attention raw", z_fused_raw.cpu().numpy()))
        with torch.no_grad():
            z_fused_test_raw = attn_fuser(
                torch.tensor(h1_test.clone().detach()).to(device),
                torch.tensor(h2_test).to(device)
            )
        components_test.append(("attention raw", z_fused_test_raw.cpu().numpy()))
        mi_scalar = MultiplicativeInteractions(input_dims[0], input_dims[1], out_dim=input_dims[0]+input_dims[1], mode="vector")
        mi_scalar.eval()
        z_fused_mia = mi_scalar(h1.clone().detach().to(device), h2.clone().detach().to(device))
        components.append(("MIA raw", z_fused_mia.detach().cpu().numpy()))
        z_fused_test_mia = mi_scalar(h1_test.clone().detach().to(device), h2_test.clone().detach().to(device))
        components_test.append(("MIA raw", z_fused_test_mia.detach().cpu().numpy()))

    if args.contrastive:
        if args.dataset == "flickr":
            clip_model, preprocess = clip.load("ViT-B/32", device=device)
            random_image_preprocessed = preprocess(Image.fromarray(x1_all)).unsqueeze(0).to(device)
            random_caption_tokenized = clip.tokenize([x2_all]).to(device)
            with torch.no_grad():
                image_features = clip_model.encode_image(random_image_preprocessed)
                text_features = clip_model.encode_text(random_caption_tokenized)
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            components.append(("clip_image_features", image_features.cpu().numpy()))
            components.append(("clip_text_features", text_features.cpu().numpy()))

    z1 = torch.concat([z1m, z1s], dim=1)
    z2 = torch.concat([z2m, z2s], dim=1)
    results_dict = {n: [] for n in task_names}
    for ind, (task, label_task) in enumerate(zip(task_names, prediction_labels)):
        plot_representations(z1m_test.detach().cpu().numpy(),z2m_test.detach().cpu().numpy(), (z1s_test+z2s_test).detach().cpu().numpy()/2, prediction_labels_test[ind], task, dataset_name, modality_names=modality_names)
        results_dict[task] = evaluate_predictability(components, label_task, task, dataset_name, components_test, prediction_labels_test[ind])
    return results_dict, projection_model
    # Evaluate predictability for each label


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='simulated', help='Dataset name (simulated, flickr, cremad, urfunny, vqa)')
    parser.add_argument('--contrastive', action='store_true', help='Whether to benchmark contrastive learning')
    parser.add_argument('--baselines', action='store_true', help='Whether to benchmark baselines')
    parser.add_argument('--config', type=str, default='./configs/datasets.yaml', help='Path to datasets.yaml (optional override)')
    args = parser.parse_args()
    if not os.path.exists('./plots/%s' % args.dataset):
        os.makedirs('./plots/%s' % args.dataset)
    results_across_seeds = {}
    shapes = {'Rm1':[], 'Rm2':[], 'Rs':[]}
    for seed_id in range(3):
        checkpoint_name = get_checkpoint_path(args.dataset, seed_id, config_path=args.config)
        results_dict, projection_model = main(args.dataset, checkpoint_name)
        shapes['Rm1'].append(projection_model.R_m1.shape[0])
        shapes['Rm2'].append(projection_model.R_m2.shape[0])
        shapes['Rs'].append(projection_model.R_s1.shape[0])
        if not results_across_seeds:
            results_across_seeds = {n: {k: [] for k in results_dict[n].keys()} for n in results_dict.keys()}
        for label_name, result in results_dict.items():
            for task_name in result.keys():
                results_across_seeds[label_name][task_name].extend(result[task_name])
    
    print("Final shapes: ")
    print("Rm1: ", np.mean(shapes['Rm1']), np.std(shapes['Rm1']))
    print("Rm2: ", np.mean(shapes['Rm2']), np.std(shapes['Rm2']))
    print("Rs: ", np.mean(shapes['Rs']), np.std(shapes['Rs']))
    for label_name, results in results_across_seeds.items():
        # Calculate mean and variance for each component
        print("Predicting label: ", label_name)
        for task_name in results.keys():
            mean_score = np.mean(results[task_name], axis=0)
            var_score = np.var(results[task_name], axis=0)
            print(f"Component: {task_name}: {mean_score:.3f}$\pm${np.sqrt(var_score):.3f}")
        print("--------------------------------")


