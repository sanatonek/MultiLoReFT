import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.decomposition import PCA
import umap 
from sim_data import generate_multimodal_data
from torch.utils.data import Dataset, DataLoader
from utils import *
import numpy as np
from multimodal_projector import MultiLoReFT
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from flickr import Multi30KMixedLangDataset
from simulation import MultimodalDataset
from vqa import VQADataset
from cremad import CremadDataset
from sklearn.metrics import r2_score, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
import clip
import timm
from torchvision import transforms
from transformers import BertTokenizer, BertModel
# from transformers import AutoTokenizer
import argparse
import random

            

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

def evaluate_cross_modal_retrieval(h0, h1, device, batch_size=512, similarity_model=None, k=10):
    """
    Batched version to evaluate cross-modal retrieval with learned similarity.
    similarity_model: a model taking (query, gallery) → score
    """
    h0 = h0.to(device)
    h1 = h1.to(device)
    similarity_model = similarity_model.to(device)

    def recall_at_k_batched(query_set, gallery_set, k=10):
        correct_count = 0
        num_samples = query_set.shape[0]

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_query = query_set[start:end]  # [B, Dq]
            if batch_query.size(1) < gallery_set.size(1):
                padding = torch.zeros(batch_query.size(0), gallery_set.size(1) - batch_query.size(1), device=batch_query.device)
                batch_query = torch.cat((batch_query, padding), dim=1)
            elif batch_query.size(1) > gallery_set.size(1):
                padding = torch.zeros(gallery_set.size(0), batch_query.size(1) - gallery_set.size(1), device=gallery_set.device)
                gallery_set = torch.cat((gallery_set, padding), dim=1)
            
            # projected_query = similarity_model(batch_query)
            sim_matrix = torch.nn.functional.cosine_similarity(batch_query.unsqueeze(1), gallery_set, dim=2)

            # scores = []
            # for i in range(gallery_set.shape[0]):
            #     # gallery_item = gallery_set[i].unsqueeze(0).expand(batch_query.size(0), -1)  # [B, Dg]
            #     score = similarity_model(batch_query, gallery_item).squeeze(-1)  # [B]
            #     scores.append(score)

            # sim_matrix = torch.stack(scores, dim=1)  # [B, N]
            topk = sim_matrix.topk(k, dim=1).indices
            true_matches = torch.arange(start, end, device=device).unsqueeze(1)
            correct = (topk == true_matches).any(dim=1).float()
            correct_count += correct.sum().item()

        return correct_count / num_samples
    return recall_at_k_batched(h0, h1, k)


def evaluate_predictability(components, labels, task_name, dataset_name):
    """Evaluate how well each component (shared and modality-specific) can predict the target label.
    """
    # Determine if this is a classification or regression task
    y = labels
    y = y.detach().cpu().numpy() if hasattr(y, "detach") else np.array(y)
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
        print(f"Predicting {task_name}: Classification ({n_classes} classes)")
        metric_name = ["roc_auc_ovr", "silhouette_score"]

        if n_classes == 2:
            model = Lasso(alpha=0.1, max_iter=1000)
            task_type = "binary"
        else:
            model = LogisticRegression(max_iter=1000, solver='lbfgs')
            task_type = "multiclass"
    else:
        if y.ndim > 1 and y.shape[1] > 1:
            print(f"Predicting {task_name}: Multidimensional Regression ({y.shape[1]} dimensions)")
            model = None  # Not used for neural_multihead
            task_type = "neural_multihead"
            metric_name = ["MSE"]
        else:
            print(f"Predicting {task_name}: Regression ({n_unique} unique values)")
            model = LinearRegression()
            task_type = "regression"
            metric_name = ["MSE"]
    
    performance_scores = []
    component_names = []
    results_dict = {}
    for name, z in components:
        z = z.detach().cpu().numpy() if torch.is_tensor(z) else z
        try:
            reg_model = SklearnTrainer(model=model, task_type=task_type)
            if task_type in ["multiclass", "binary"]:
                score, score_var, score_1, score_var_1 = reg_model.train_and_evaluate(z, y, k=5)
            else:
                score, score_var = reg_model.train_and_evaluate(z, y, k=5)
            performance_scores.append((score, score_var))
            component_names.append(name)
            results_dict[name] = score
            # print(name, f"-----Predictive performance of {task_name}: ({metric_name[0]}): {score:.3f} (var: {score_var:.3f})")
            # if task_type in ["multiclass", "binary"]:
            #     print(name, f"-----Predictive performance of {task_name}: ({metric_name[1]}): {score_1:.3f} (var: {score_var_1:.3f})")
        except Exception as e:
            print(f"Error evaluating {name}: {str(e)}")
            continue
    return results_dict



def plot_representations(z_n, labels, task_name, dataset_name, save_dir="./plots", modality_names=["A", "B"]):
    """Plot 2D PCA projections of the representations colored by each label.
    
    Args:
        z_n: Tuple of (modality_specific, shared) representations for each modality
        labels: Target labels
        save_dir: Directory to save the plots
    """
    # for l_ind in range(len(labels[0])):
    fig, axs = plt.subplots(2, 2, figsize=(16, 16))    
    titles = [
        (f'Modality-specific {modality_names[0]}', 0, 0),
        (f'Shared {modality_names[0]}', 0, 1),
        (f'Modality-specific {modality_names[1]}', 1, 0),
        (f'Shared {modality_names[1]}', 1, 1)
    ]
    for title, i, j in titles:
        ax = axs[i, j]
        ax.set_title(title, fontsize=18)     
        data = z_n[i*2+j]#.detach().cpu().numpy()
        
        if data.shape[1] >= 2:
            # Normal PCA case (2+ dimensions)
            reducer = PCA(n_components=2)
            # reducer = umap.UMAP()
            x = reducer.fit_transform(data)
            ax.scatter(x[:, 0], x[:, 1], c=labels.cpu().numpy() if torch.is_tensor(labels) else labels)
        # elif data.shape[1] == 2:
        #     reducer = umap.UMAP(n_neighbors=10,min_dist=0.25,random_state=3)
        #     x = reducer.fit_transform(data)
        #     ax.scatter(x[:, 0], x[:, 1], c=labels.cpu().numpy() if torch.is_tensor(labels) else labels)
            # ax.scatter(data[:, 0], data[:, 1], c=labels.cpu().numpy() if torch.is_tensor(labels) else labels)
        elif data.shape[1] == 1:
            # Handle 1D case by adding a zero column for visualization
            x = np.hstack([data, np.zeros_like(data)])
            ax.scatter(x[:, 0], x[:, 1], c=labels.cpu().numpy() if torch.is_tensor(labels) else labels)
        else:
            # If no valid features, just write "No Data"
            ax.text(0.5, 0.5, "No Data", horizontalalignment='center', verticalalignment='center')

    plt.savefig(f"{save_dir}/{dataset_name}/test_{task_name}.pdf")
    print(f"Saved plot to {save_dir}/{dataset_name}/test_{task_name}.pdf")


def plot_projection_matrices(model, threshold=0.00, save_dir="./plots"):
    """Plot the learned projection matrices and their correlations.
    
    Args:
        model: Trained ProjectionModule
        threshold: Threshold for singular values
        save_dir: Directory to save the plots
    """
    # Get singular values
    # print("Shared SVD: ", torch.linalg.svdvals(model.R_s).detach().cpu().numpy())   
    # print("Modality-specific SVD: ", torch.linalg.svdvals(model.R_m1).detach().cpu().numpy())
    # print("Modality-specific SVD: ", torch.linalg.svdvals(model.R_m2).detach().cpu().numpy())
    
    # Get matrices above threshold
    shared_sv = torch.where(torch.linalg.svdvals(model.R_s1) > threshold)
    Rs_1 = model.R_s1[shared_sv].detach().cpu().numpy()
    shared_sv = torch.where(torch.linalg.svdvals(model.R_s2) > threshold)
    Rs_2 = model.R_s2[shared_sv].detach().cpu().numpy()
    m1_sv = torch.where(torch.linalg.svdvals(model.R_m1) > threshold)
    Rm1 = model.R_m1[m1_sv].detach().cpu().numpy()
    m2_sv = torch.where(torch.linalg.svdvals(model.R_m2) > threshold)
    Rm2 = model.R_m2[m2_sv].detach().cpu().numpy()

    # Plot matrices
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    matrices = [Rs_1, Rs_2, Rm1, Rm2]
    titles = ["Shared Projection (R_s1)", "Shared Projection (R_s2)", "Modality-Specific (R_m1)", "Modality-Specific (R_m2)"]
    
    # Find the maximum absolute value for symmetric color range
    max_abs = max(abs(Rs_1).max(), abs(Rs_2).max(), abs(Rm1).max(), abs(Rm2).max())
    vmin, vmax = -max_abs, max_abs

    # Plot matrices
    for ax, matrix, title in zip(axs, matrices, titles):
        sns.heatmap(matrix, ax=ax, cmap="RdBu_r", cbar=True, vmin=vmin, vmax=vmax, center=0)
        # Get the correct matrix based on title
        if "R_s1" in title:
            sv = torch.linalg.svdvals(model.R_s1).detach().cpu().numpy()
        elif "R_s2" in title:
            sv = torch.linalg.svdvals(model.R_s2).detach().cpu().numpy()
        elif "R_m1" in title:
            sv = torch.linalg.svdvals(model.R_m1).detach().cpu().numpy()
        else:
            sv = torch.linalg.svdvals(model.R_m2).detach().cpu().numpy()
        ax.set_title(f"{title}\nSVs: {sv[min(len(sv)-1, 10)]}")
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/learned_matrices.pdf")
    plt.close()

    # Plot correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    names = ['R_s1', 'R_s2', 'R_m1', 'R_m2']
    corr_matrix = np.zeros((4, 4))
    
    for i in range(4):
        for j in range(4):
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
    plt.tight_layout()
    plt.savefig(f"{save_dir}/matrix_correlations.pdf")
    plt.close()


def main(dataset_name, checkpoint_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if dataset_name=="simulated":
        # Load and prepare data
        loaded_data = np.load("./data/simplest_sim_nongaussian.npz")
        h1 = loaded_data["h1"]
        h2 = loaded_data["h2"]
        x1 = loaded_data["x1"]
        x2 = loaded_data["x2"]
        labels = loaded_data["labels"][5000:6000]
        # Create dataset
        dataset = MultimodalDataset(h1[5000:6000], h2[5000:6000], x1[5000:6000], x2[5000:6000], labels[5000:6000])  
        # Load model
        # Initialize model
        projection_model = MultiLoReFT(
            input_dims=[10,10], 
            shared_rank=10, 
            specific_rank=10, 
            pruning_threshold=0.2,
            staging=True,
            pruning=True,
            device=device,
            shared_R_mode="pad"
        ).to(device)
        projection_model = load_checkpoint(filepath=checkpoint_name, model=projection_model)
        projection_model.eval()
        projection_model = projection_model.to(device)
        # Get representations
        h1 = torch.Tensor(h1[5000:6000]).to(device)
        h2 = torch.Tensor(h2[5000:6000]).to(device)
        phis = projection_model([h1,h2])
        phi_1 = phis[0]
        phi_2 = phis[1]
        z = projection_model.fuse_representations(phis)
        z_n = projection_model.decouple(phis, full=True, th=0.05)
        z1m, z1s, z2m, z2s = z_n[0][0], z_n[0][1], z_n[1][0], z_n[1][1]
        prediction_labels = [labels[:,0], labels[:,1], labels[:,2]]
        task_names = ['shared', 'm1', 'm2']
        modality_names = ["A", "B"]
    elif dataset_name=="simulated_apollo":
        # Load and prepare data
        loaded_data = np.load("./data/simulated_data_apollo.npz")
        h1 = loaded_data["h1"]
        h2 = loaded_data["h2"]
        x1 = loaded_data["x1"]
        x2 = loaded_data["x2"]
        labels = loaded_data["labels"]
        n_train = int(0.8*len(h1))
        n_val = int(0.1*len(h1))
        n_test = len(h1) - n_train - n_val
        # Create dataset
        dataset = MultimodalDataset(h1[n_train+n_val:], h2[n_train+n_val:], x1[n_train+n_val:], x2[n_train+n_val:], labels[n_train+n_val:])  
        # Load model
        # Initialize model
        projection_model = MultiLoReFT(
            input_dims=[80,40], 
            shared_rank=40, 
            specific_rank=40, 
            pruning_threshold=0.2,
            staging=True,
            pruning=True,
            device=device,
            shared_R_mode="pad"
        ).to(device)
        projection_model = load_checkpoint(filepath=checkpoint_name, model=projection_model)
        projection_model.eval()
        projection_model = projection_model.to(device)
        # Get representations
        h1 = torch.Tensor(h1[n_train+n_val:]).to(device)
        h2 = torch.Tensor(h2[n_train+n_val:]).to(device)
        phis = projection_model([h1,h2])
        phi_1 = phis[0]
        phi_2 = phis[1]
        z = projection_model.fuse_representations(phis)
        z_n = projection_model.decouple(phis, full=True, th=0.05)
        z1m, z1s, z2m, z2s = z_n[0][0], z_n[0][1], z_n[1][0], z_n[1][1]
        prediction_labels = [labels[n_train+n_val:,0], labels[n_train+n_val:,1]]
        task_names = ['shared', 'm1']
        modality_names = ["A", "B"]
    else:
        # Load CLIP (English only)
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
        clip_model.eval()
        if dataset_name=="flickr":
            test_dataset = Multi30KMixedLangDataset(split="test", device=device)
            check_point = "./ckpts/flickr_model_all.pth"
            projection_model = MultiLoReFT(
                                    input_dims=[768,768], 
                                    shared_rank=768, 
                                    specific_rank=768, 
                                    pruning_threshold=0.1,
                                    device=device,
                                    staging=True,
                                    pruning=True,
                                    dataset_name="flickr"
                                ).to(device)
            modality_names = ["image", "caption"]
        elif dataset_name=="vqa":
            test_dataset = VQADataset(split="validation", device=device)
            check_point = "./ckpts/vqa_model_all.pth"
            test_dataset = torch.utils.data.Subset(test_dataset, range(1000))
            projection_model = MultiLoReFT(
                                        input_dims=[768,768], 
                                        shared_rank=128, 
                                        specific_rank=128, 
                                        device=device
                                    )
            modality_names = ["image", "question"]
        elif dataset_name=="cremad":
            test_dataset = CremadDataset(split='test')
            check_point = "./ckpts/cremad_model_all.pth"
            projection_model = MultiLoReFT(input_dims=[400, 768],   # adjust if needed: video_feat dim, audio_feat dim
                                            shared_rank=768,
                                            specific_rank=768,
                                            pruning_threshold=0.1,
                                            device=device,
                                            staging=True,
                                            pruning=True,
                                            dataset_name="cremad",
                                            shared_R_mode="pad"
                                        ).to(device)
            modality_names = ["video", "audio"]
        test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)
                                
        projection_model = load_checkpoint(filepath=check_point, model=projection_model)
        projection_model.eval()
        projection_model = projection_model.to(device)
        labels, labels_2 = [], []
        z1s, z2s, z1m, z2m = [], [], [], []
        phi_1, phi_2 = [], []
        h1, h2, z = [], [], []
        random_sample = random.randint(0, 1000)
        x2_all, x1_all = [], []
        with torch.no_grad():
            count = 0
            for i, batch in enumerate(test_dataloader):
                if dataset_name=="flickr":
                    image_feats, caption_feats, x1, captions, label = batch
                    # Generate random binary labels for each item in batch
                    lang_idx = torch.randint(0, 2, (len(x1),))
                    # Use the labels to select language for each item
                    text_feats = torch.stack([caption_feats[0], caption_feats[1]], dim=1).gather(
                        1, lang_idx.unsqueeze(1).unsqueeze(2).expand(-1, -1, caption_feats[0].shape[-1])
                    ).squeeze(1)
                    l2 = torch.stack([caption_feats[0], caption_feats[1]], dim=1).gather(
                        1, abs(1-lang_idx).unsqueeze(1).unsqueeze(2).expand(-1, -1, caption_feats[0].shape[-1])).squeeze(1)
                    x2 = [captions[0][i] if idx == 0 else captions[1][i] for i, idx in enumerate(lang_idx)]
                    l1 = lang_idx
                    label = [l1, l2]
                    h1.append(image_feats)
                    h2.append(text_feats)
                    task_names = ['language', 'other_caption']
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
                elif dataset_name=="vqa":
                    image_feats, question_feats, x1, x2, answer, answer_feat = batch
                    h1.append(image_feats)
                    h2.append(question_feats)
                    label = [answer_feat]
                    task_names = ['answer']
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
            x2_all = np.concatenate(x2_all, axis=0)
            # labels_2 = np.concatenate(labels_2, axis=0)
            x1_all = np.concatenate([img.cpu().numpy() for img in x1_all], axis=0)
            random_sample = random.randint(0, len(x1_all))
            random_caption = x2_all[random_sample]
            random_image = x1_all[random_sample]

            if dataset_name=="flickr" or dataset_name=="vqa":
                # Explore the latent space to find the closest samples
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

                plot_closest_images(x1_all, x1_all[random_sample], closest_images_modality_specific, './plots/%s/closest_images_modality_specific.png' % dataset_name)
                plot_closest_images(x1_all, x1_all[random_sample], closest_images_shared, './plots/%s/closest_shared_space.png' % dataset_name)
        prediction_labels = labels

    # Evaluate and plot
    # plot_projection_matrices(projection_model)
    components = [
        ("Zs1", z1s.detach().cpu().numpy()),  # Shared representation from modality 1
        ("Zs2", z2s.detach().cpu().numpy()),  # Shared representation from modality 2
        ("Zm1", z1m.detach().cpu().numpy()),  # Modality-specific representation from modality 1
        ("Zm2", z2m.detach().cpu().numpy()),
        ("Z", z.detach().cpu().numpy()),
        ("H1", h1.detach().cpu().numpy()),
        ("H2", h2.detach().cpu().numpy())
    ] 
    z1 = torch.concat([z1m, z1s], dim=1)
    z2 = torch.concat([z2m, z2s], dim=1)
    res = evaluate_cross_modal_retrieval(z1, h2, device, batch_size=256, similarity_model=SimilarityMLP(z1.shape[1], h2.shape[1]))
    print("Recall@10 predicting caption from z1: ", res)
    res = evaluate_cross_modal_retrieval(z2, h1, device, batch_size=256, similarity_model=SimilarityMLP(z2.shape[1], h1.shape[1]))
    print("Recall@10 predicting image from z2: ", res)
    res = evaluate_cross_modal_retrieval(phi_1, h2, device, batch_size=256, similarity_model=SimilarityMLP(phi_1.shape[1], h2.shape[1]))
    print("Recall@10 predicting caption from phi1: ", res)
    res = evaluate_cross_modal_retrieval(phi_2, h1, device, batch_size=256, similarity_model=SimilarityMLP(phi_2.shape[1], h1.shape[1]))
    print("Recall@10 predicting image from phi2: ", res)
    res = evaluate_cross_modal_retrieval(h1, h2, device, batch_size=256, similarity_model=SimilarityMLP(h1.shape[1], h2.shape[1]))
    print("Recall@10 predicting caption from h1: ", res)
    res = evaluate_cross_modal_retrieval(h2, h1, device, batch_size=256, similarity_model=SimilarityMLP(h2.shape[1], h1.shape[1]))
    print("Recall@10 predicting image from h2: ", res)
    for name, z in components:
        print(name, z.shape)
    results_dict = []
    for task_ind, label_task in enumerate(prediction_labels):
        print(label_task.shape)
        # Fix: check if label_task[0] is a scalar (numpy.float64) or array
        if np.isscalar(label_task[0]) or (hasattr(label_task[0], 'shape') and label_task[0].shape == ()):  # scalar
            # handle scalar case
            plot_representations((z1m.detach().cpu().numpy(), z1s.detach().cpu().numpy(), z2m.detach().cpu().numpy(), z2s.detach().cpu().numpy()), label_task, task_names[task_ind], dataset_name, modality_names=modality_names)
        else:
            # handle array case
            plot_representations((z1m.detach().cpu().numpy(), z1s.detach().cpu().numpy(), z2m.detach().cpu().numpy(), z2s.detach().cpu().numpy()), label_task, task_names[task_ind], dataset_name, modality_names=modality_names)
        results_dict.append(evaluate_predictability(components, label_task, task_names[task_ind], dataset_name))
    return task_names, results_dict
    # Evaluate predictability for each label


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='simulated', help='Dataset name (simulated or flickr)')
    args = parser.parse_args()
    if not os.path.exists('./plots/%s' % args.dataset):
        os.makedirs('./plots/%s' % args.dataset)
    results_across_seeds = {}
    for seed_id in range(4):
        checkpoint_name = "./ckpts/%s_multi_loreft_%d.pth" % (args.dataset, seed_id)
        task_names, results_dict = main(args.dataset, checkpoint_name)
        for task_name, result in zip(task_names, results_dict):
            if task_name not in results_across_seeds:
                results_across_seeds[task_name] = []
            results_across_seeds[task_name].append(result)
    # print(results_across_seeds)
    # Calculate mean and variance of results across seeds
    for task_name, results in results_across_seeds.items():
        # Calculate mean and variance for each component
        component_names = list(results[0].keys())
        
        print(f"Task: {task_name}")
        for component_name in component_names:
            performance_scores = np.array([result[component_name] for result in results])
            
            mean_score = np.mean(performance_scores, axis=0)
            var_score = np.var(performance_scores, axis=0)
            print(f"Component: {component_name}, Mean Score: {mean_score:.3f}, Variance: {var_score:.3f}")
            # Create bar plots of the performances for each component
            # fig, ax = plt.subplots(figsize=(10, 6))
            # x = np.arange(len(component_names))
            # ax.bar(x, mean_scores, yerr=np.sqrt(var_scores), capsize=5, color='skyblue')
            # ax.set_xticks(x)
            # ax.set_xticklabels(component_names, rotation=45, ha='right')
            # ax.set_ylabel('Performance Score')
            # ax.set_title(f'Performance Scores for Task: {task_name}')
            # plt.tight_layout()
            # plt.savefig(f"./plots/{args.dataset}/{task_name}_performance_barplot.pdf")
            # plt.close(fig)
            # print(f"Saved bar plot to ./plots/{args.dataset}/{task_name}_performance_barplot.pdf")
