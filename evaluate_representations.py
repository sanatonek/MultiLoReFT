import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.decomposition import PCA
from sim_data import generate_multimodal_data
from torch.utils.data import Dataset, DataLoader
from utils import *
import numpy as np
from multimodal_projector import MultimodalDataset, MultiLoReFT
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from flickr import Multi30KMixedLangDataset
from vqa import VQADataset
from sklearn.metrics import r2_score, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
import clip
import timm
from torchvision import transforms
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel
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

def evaluate_cross_modal_retrieval(phis0, phis1, projector, device):
    """
    Evaluates cross-modal retrieval performance using cosine similarity.

    Assumes:
    - dataloader yields (h1, h2, x1, x2, l)
    - projector(h1, h2) -> (phi1, phi2)
    - h1: features from modality 1 (e.g., image)
    - h2: features from modality 2 (e.g., text)

    Returns:
        dict with Recall@1, Recall@5, Recall@10 for both directions
    """

    # Compute similarity matrix
    sim_matrix = phis0 @ phis1.T  # (N x N)

    def recall_at_k(sim_matrix, k, labels):
        topk = sim_matrix.topk(k, dim=1).indices
        
        # For each query i, its true match is at index i
        # Create a column vector of indices [0,1,2,...,N-1]
        true_matches = torch.arange(sim_matrix.shape[0], device=sim_matrix.device).unsqueeze(1)
        
        # Check if true matching index appears in top k predictions
        correct = (topk == true_matches).any(dim=1).float()
        return correct.mean().item()

    results = {
        'Image→Text R@1': recall_at_k(sim_matrix, 1, labels),
        'Image→Text R@5': recall_at_k(sim_matrix, 5, labels),
        'Image→Text R@10': recall_at_k(sim_matrix, 10, labels),
        'Text→Image R@1': recall_at_k(sim_matrix.T, 1, labels),
        'Text→Image R@5': recall_at_k(sim_matrix.T, 5, labels),
        'Text→Image R@10': recall_at_k(sim_matrix.T, 10, labels),
    }

    return results

def evaluate_predictability(components, labels, task_name):
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
    is_classification = (n_unique <= 10 and 
                        np.all(np.mod(y, 1) == 0))  # Check if all values are integers
    
    if is_classification:
        n_classes = len(np.unique(y))
        print(f"Task type: Classification ({n_classes} classes)")

        if n_classes == 2:
            model = Lasso(alpha=0.1)
            metric_name = "roc_auc"
            task_type = "binary"
        else:
            model = OneVsRestClassifier(LogisticRegression(solver='liblinear'))
            metric_name = "roc_auc_ovr"
            task_type = "multiclass"
    else:
        if y.ndim > 1 and y.shape[1] > 1:
            print(f"Task type: Multidimensional Regression ({y.shape[1]} dimensions)")
            model = None  # Not used for neural_multihead
            task_type = "neural_multihead"
            metric_name = "MSE"
        else:
            print(f"Task type: Regression ({n_unique} unique values)")
            model = LinearRegression()
            task_type = "regression"
            metric_name = "MSE"
    
    performance_scores = []
    component_names = []
    for name, z in components:
        try:
            reg_model = SklearnTrainer(model=model, task_type=task_type)
            score, score_var = reg_model.train_and_evaluate(z.detach().cpu(), y, k=5)
            performance_scores.append((score, score_var))
            component_names.append(name)
            print(name, f"-----Predictive performance ({metric_name}): {score:.3f} (var: {score_var:.3f})")
        except Exception as e:
            print(f"Error evaluating {name}: {str(e)}")
            continue

    # Create bar plot
    scores, variances = zip(*performance_scores)
    x_pos = np.arange(len(component_names))

    plt.figure(figsize=(10, 6))
    plt.bar(x_pos, scores, yerr=variances, align='center', alpha=0.7, capsize=10)
    plt.xticks(x_pos, component_names, rotation=45, ha='right')
    plt.ylabel(f'Predictive Performance ({metric_name})')
    plt.title('Predictive Performance of Each Component')
    plt.tight_layout()
    plt.savefig(f"plots/predictability_plot_{task_name}.png")



def plot_representations(z_n, labels, task_name, save_dir="./plots"):
    """Plot 2D PCA projections of the representations colored by each label.
    
    Args:
        z_n: Tuple of (modality_specific, shared) representations for each modality
        labels: Target labels
        save_dir: Directory to save the plots
    """
    # for l_ind in range(len(labels[0])):
    fig, axs = plt.subplots(2, 2, figsize=(16, 16))    
    titles = [
        ('Modality-specific A', 0, 0),
        ('Shared A', 0, 1),
        ('Modality-specific B', 1, 0),
        ('Shared B', 1, 1)
    ]
    for title, i, j in titles:
        ax = axs[i, j]
        ax.set_title(title)     
        data = z_n[i*2+j].detach().cpu().numpy()
        
        if data.shape[1] >= 2:
            # Normal PCA case (2+ dimensions)
            pca = PCA(n_components=2)
            x = pca.fit_transform(data)
            ax.scatter(x[:, 0], x[:, 1], c=labels.cpu().numpy() if torch.is_tensor(labels) else labels)
        elif data.shape[1] == 1:
            # Handle 1D case by adding a zero column for visualization
            x = np.hstack([data, np.zeros_like(data)])
            ax.scatter(x[:, 0], x[:, 1], c=labels.cpu().numpy() if torch.is_tensor(labels) else labels)
        else:
            # If no valid features, just write "No Data"
            ax.text(0.5, 0.5, "No Data", horizontalalignment='center', verticalalignment='center')

        plt.savefig(f"{save_dir}/test_{task_name}.pdf")


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
    shared_sv = torch.where(torch.linalg.svdvals(model.R_s) > threshold)
    Rs = model.R_s[shared_sv].detach().cpu().numpy()
    m1_sv = torch.where(torch.linalg.svdvals(model.R_m1) > threshold)
    Rm1 = model.R_m1[m1_sv].detach().cpu().numpy()
    m2_sv = torch.where(torch.linalg.svdvals(model.R_m2) > threshold)
    Rm2 = model.R_m2[m2_sv].detach().cpu().numpy()

    # Plot matrices
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    matrices = [Rs, Rm1, Rm2]
    titles = ["Shared Projection (R_s)", "Modality-Specific (R_m1)", "Modality-Specific (R_m2)"]
    
    # Find the maximum absolute value for symmetric color range
    max_abs = max(abs(Rs).max(), abs(Rm1).max(), abs(Rm2).max())
    vmin, vmax = -max_abs, max_abs

    # Plot matrices
    for ax, matrix, title in zip(axs, matrices, titles):
        sns.heatmap(matrix, ax=ax, cmap="RdBu_r", cbar=True, vmin=vmin, vmax=vmax, center=0)
        # Get the correct matrix based on title
        if "Shared" in title:
            sv = torch.linalg.svdvals(model.R_s).detach().cpu().numpy()
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
    names = ['R_s', 'R_m1', 'R_m2']
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
    ax.set_title('Correlation between Projection Matrices')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/matrix_correlations.pdf")
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='simulated', help='Dataset name (simulated or flickr)')
    args = parser.parse_args()
    dataset_name = args.dataset
    if not os.path.exists('./plots/%s' % dataset_name):
        os.makedirs('./plots/%s' % dataset_name)
    
    if dataset_name=="simulated":
        # Load and prepare data
        loaded_data = np.load("./data/simplest_sim_nongaussian.npz")
        h1 = loaded_data["h1"]
        h2 = loaded_data["h2"]
        x1 = loaded_data["x1"]
        x2 = loaded_data["x2"]
        labels = loaded_data["labels"][5000:6000]
        # Create dataset
        dataset = MultimodalDataset(h1[5000:6000], h2[5000:6000], x1[5000:6000], x2[5000:6000], labels)  
        # Load model
        # Initialize model
        projection_model = MultiLoReFT(
            input_dims=[10,10], 
            shared_rank=20, 
            specific_rank=10, 
            pruning_threshold=0.2,
            staging=True,
            pruning=True,
            device=device
        )
        projection_model = load_checkpoint(filepath="./ckpts/projection_module.pth", model=projection_model)
        projection_model.eval()
        projection_model = projection_model.to(device)
        # Get representations
        h1 = F.normalize(torch.Tensor(dataset.h1).float(), dim=1).to(device)
        h2 = F.normalize(torch.Tensor(dataset.h2).float(), dim=1).to(device)
        phis = projection_model([h1,h2])
        z = projection_model.fuse_representations(phis)
        z_n = projection_model.decouple(phis, full=True, th=0.05)
        (z1m, z1s, z2m, z2s) = z_n[0][0], z_n[0][1], z_n[1][0], z_n[1][1]
        prediction_labels = [labels[:,0], labels[:,1], labels[:,2]]
        task_names = ['shared', 'm1', 'm2']
    
    else:
        # Load CLIP (English only)
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
        clip_model.eval()
        if dataset_name=="flickr":
            test_dataset = Multi30KMixedLangDataset(split="test", device=device)
            check_point = "./ckpts/flickr_model_all.pth"
        elif dataset_name=="vqa":
            test_dataset = VQADataset(split="validation", device=device)
            check_point = "./ckpts/vqa_model_all.pth"
            test_dataset = torch.utils.data.Subset(test_dataset, range(1000))
        test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        projection_model = MultiLoReFT(
                                    input_dims=[768,768], 
                                    shared_rank=128, 
                                    specific_rank=128, 
                                    device=device
                                )
                                
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
                elif dataset_name=="vqa":
                    image_feats, question_feats, x1, x2, answer, answer_feat = batch
                    h1.append(image_feats)
                    h2.append(question_feats)
                    label = [answer_feat]
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

            # Explore the latent space to find the closest samples
            closest_images_shared = find_closest_samples(z1s, z1s[random_sample], "image shared")
            closest_images_modality_specific = find_closest_samples(z1m, z1m[random_sample], "image modality-specific")
            closest_captions_shared = find_closest_samples(z2m, z2m[random_sample], "caption shared")
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
        task_names = ['language', 'other_caption'] if dataset_name=="flickr" else ['answer']

    # Evaluate and plot
    plot_projection_matrices(projection_model)
    components = [
        ("Zs1", z1s),  # Shared representation from modality 1
        ("Zs2", z2s),  # Shared representation from modality 2
        ("Zm1", z1m),  # Modality-specific representation from modality 1
        ("Zm2", z2m),
        ("Z", z),
        ("H1", h1),
        ("H2", h2)
    ] 
    for name, z in components:
        print(name, z.shape)
    for task_ind, label_task in enumerate(prediction_labels):
        print(label_task.shape)
        label_task = label_task.squeeze()
        if label_task[0].numel() == 1:
            plot_representations((z1m, z1s, z2m, z2s), label_task, task_names[task_ind])
        evaluate_predictability(components, label_task, task_names[task_ind])
    
    # Evaluate predictability for each label


if __name__=="__main__":
    main()
