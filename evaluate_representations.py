import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sim_data import generate_multimodal_data
from torch.utils.data import Dataset, DataLoader
from utils import *
import numpy as np
from multimodal_projector import *
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from flickr import Multi30KMixedLangDataset


def evaluate_predictability(z_n, labels, label_idx):
    """Evaluate how well each component (shared and modality-specific) can predict the target label.
    
    Args:
        z_n: Tuple of (modality_specific, shared) representations for each modality
        labels: Target labels
        label_idx: Index of the label to evaluate
    """
    print(f'Predictability for label {label_idx}')
    components = [
        ("Zs1", z_n[0][1]),  # Shared representation from modality 1
        ("Zs2", z_n[1][1]),  # Shared representation from modality 2
        ("Zm1", z_n[0][0]),  # Modality-specific representation from modality 1
        ("Zm2", z_n[1][0])   # Modality-specific representation from modality 2
    ] 
    
    # Determine if this is a classification or regression task
    y = labels[:,label_idx]
    unique_values = np.unique(y)
    n_unique = len(unique_values)
    
    # Handle edge cases
    if n_unique == 1:
        print(f"Warning: Label {label_idx} has only one unique value. Skipping evaluation.")
        return
    
    # Determine task type based on both number of unique values and their nature
    is_classification = (n_unique <= 10 and 
                        np.all(np.mod(y, 1) == 0))  # Check if all values are integers
    
    if is_classification:
        print(f"Task type: Classification ({n_unique} classes)")
        model = LogisticRegression(max_iter=500)
        task_type = "classification"
        metric_name = "accuracy"
    else:
        print(f"Task type: Regression ({n_unique} unique values)")
        model = LinearRegression()
        task_type = "regression"
        metric_name = "R2 score"
    
    for name, z in components:
        try:
            reg_model = SklearnTrainer(model=model, task_type=task_type)
            score, score_var = reg_model.train_and_evaluate(z.detach().cpu(), y, k=5)
            print(name, f"-----Predictive performance ({metric_name}): {score:.3f} (var: {score_var:.3f})")
        except Exception as e:
            print(f"Error evaluating {name}: {str(e)}")
            continue


def plot_representations(z_n, labels, save_dir="./plots"):
    """Plot 2D PCA projections of the representations colored by each label.
    
    Args:
        z_n: Tuple of (modality_specific, shared) representations for each modality
        labels: Target labels
        save_dir: Directory to save the plots
    """
    for l_ind in range(labels.shape[1]):
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
            data = z_n[i][j].detach().cpu().numpy()
            
            if data.shape[1] >= 2:
                # Normal PCA case (2+ dimensions)
                pca = PCA(n_components=2)
                x = pca.fit_transform(data)
                ax.scatter(x[:, 0], x[:, 1], c=labels[:, l_ind])
            elif data.shape[1] == 1:
                # Handle 1D case by adding a zero column for visualization
                x = np.hstack([data, np.zeros_like(data)])
                ax.scatter(x[:, 0], x[:, 1], c=labels[:, l_ind])
            else:
                # If no valid features, just write "No Data"
                ax.text(0.5, 0.5, "No Data", horizontalalignment='center', verticalalignment='center')

        plt.savefig(f"{save_dir}/test_{l_ind}.pdf")
        plt.close()


def plot_projection_matrices(model, threshold=0.00, save_dir="./plots"):
    """Plot the learned projection matrices and their correlations.
    
    Args:
        model: Trained ProjectionModule
        threshold: Threshold for singular values
        save_dir: Directory to save the plots
    """
    # Get singular values
    print("Shared SVD: ", torch.linalg.svdvals(model.R_s).detach().cpu().numpy())   
    print("Modality-specific SVD: ", torch.linalg.svdvals(model.R_m1).detach().cpu().numpy())
    print("Modality-specific SVD: ", torch.linalg.svdvals(model.R_m2).detach().cpu().numpy())
    
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
        ax.set_title(f"{title}\nSVs: {sv}")
    
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
    dataset_name = "flickr"
    
    if dataset_name=="simulated_data":
        # Load and prepare data
        loaded_data = np.load("./data/simulated_data.npz")
        h1 = loaded_data["h1"]
        h2 = loaded_data["h2"]
        x1 = loaded_data["x1"]
        x2 = loaded_data["x2"]
        labels = loaded_data["labels"][3000:]
        # Create dataset
        dataset = MultimodalDataset(h1[3000:], h2[3000:], x1[3000:], x2[3000:], labels)  
        # Load model
        projection_model = ProjectionModule(input_dims=[5,5], shared_rank=4, specific_rank=4, data_dim={'A':5, 'B':6}).to(device)
        checkpoint = load_checkpoint(filepath="./ckpts/projection_module.pth", model=projection_model)
        # Get representations
        h1 = F.normalize(torch.Tensor(dataset.h1).float(), dim=1).to(device)
        h2 = F.normalize(torch.Tensor(dataset.h2).float(), dim=1).to(device)
        phis = projection_model([h1,h2])
        z_n = projection_model.decouple(phis, full=True, th=0.05)
    
    if dataset_name=="flickr":
        test_dataset = Multi30KMixedLangDataset(split="test", device=device)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        projection_model = ProjectionModule(
                                    input_dims=[512,512], 
                                    shared_rank=256, 
                                    specific_rank=256, 
                                    data_dim=None
                                ).to(device)
        z_n, labels = [], []
        for batch in test_dataloader:
            image_feats, text_feats, images, captions, label = batch
            image_feats = image_feats.to(device)
            text_feats = text_feats.to(device)
            phis = projection_model([image_feats, text_feats])
            z_n.append(projection_model.decouple(phis, full=True))
            labels.extend(label)
        z_n = torch.cat(z_n, dim=0)
    # Evaluate and plot
    plot_projection_matrices(projection_model)
    plot_representations(z_n, labels)
    
    # Evaluate predictability for each label
    for label_idx in range(labels.shape[1]):
        evaluate_predictability(z_n, labels, label_idx)


if __name__=="__main__":
    main()
