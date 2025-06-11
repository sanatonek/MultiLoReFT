import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import torch
import wandb

def plot_representations(z_n, labels, save_dir="./plots", model_name="projection_module"):
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
            
            if data.shape[1] > 2:
                # Normal PCA case (2+ dimensions)
                pca = PCA(n_components=2)
                x = pca.fit_transform(data)
                df_plot = pd.DataFrame(x, columns=['x1', 'x2'])
                df_plot['label'] = labels[:, l_ind]
                sns.scatterplot(data=df_plot, x='x1', y='x2', hue='label', ax=ax)
            elif data.shape[1] == 2:
                # If already 2D, just plot directly
                df_plot = pd.DataFrame(data, columns=['x1', 'x2'])
                df_plot['label'] = labels[:, l_ind]
                sns.scatterplot(data=df_plot, x='x1', y='x2', hue='label', ax=ax)
            elif data.shape[1] == 1:
                # Handle 1D case by adding a zero column for visualization
                x = np.hstack([data, np.zeros_like(data)])
                ax.scatter(x[:, 0], x[:, 1], c=labels[:, l_ind])
            else:
                # If no valid features, just write "No Data"
                ax.text(0.5, 0.5, "No Data", horizontalalignment='center', verticalalignment='center')

        plt.savefig(f"{save_dir}/{model_name}_test_{l_ind}.png")
        plt.close()

def plot_hidden(h1, h2, labels, save_dir="./plots", model_name="projection_module"):
    """Plot 2D PCA projections of the hidden representations colored by each label.
    
    Args:
        h1: Hidden representation from modality 1
        h2: Hidden representation from modality 2
        labels: Target labels
        save_dir: Directory to save the plots
    """
    fig, axs = plt.subplots(3, 2, figsize=(16, 16))    
    titles = ['Modality A', 'Modality B']
    
    for l in range(labels.shape[1]):
        for title, data in zip(titles, [h1, h2]):
            #ax = axs[0 if title == 'Modality A' else 1]
            ax = axs[l, 0] if title == 'Modality A' else axs[l, 1]
            ax.set_title(title)
            if type(data) == torch.Tensor:  
                data = data.detach().cpu().numpy()
            
            if data.shape[1] > 2:
                # Normal PCA case (2+ dimensions)
                pca = PCA(n_components=2)
                x = pca.fit_transform(data)
                df_plot = pd.DataFrame(x, columns=['x1', 'x2'])
                df_plot['label'] = labels[:, l]
                sns.scatterplot(data=df_plot, x='x1', y='x2', hue='label', ax=ax)
                #ax.scatter(x[:, 0], x[:, 1], c=labels[:, l])
            elif data.shape[1] == 2:
                # If already 2D, just plot directly
                df_plot = pd.DataFrame(data, columns=['x1', 'x2'])
                df_plot['label'] = labels[:, l]
                sns.scatterplot(data=df_plot, x='x1', y='x2', hue='label', ax=ax)
                #ax.scatter(data[:, 0], data[:, 1], c=labels[:, l])
            elif data.shape[1] == 1:
                # Handle 1D case by adding a zero column for visualization
                x = np.hstack([data, np.zeros_like(data)])
                ax.scatter(x[:, 0], x[:, 1], c=labels[:, l])
            else:
                # If no valid features, just write "No Data"
                ax.text(0.5, 0.5, "No Data", horizontalalignment='center', verticalalignment='center')

    plt.savefig(f"{save_dir}/{model_name}_hidden.png")
    plt.close()

def plot_projection_matrices(model, threshold=0.00, save_dir="./plots", model_name="projection_module"):
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
    #Rs = model.R_s[shared_sv].detach().cpu().numpy()
    Rs = model.R_s.detach().cpu().numpy()
    m1_sv = torch.where(torch.linalg.svdvals(model.R_m1) > threshold)
    #Rm1 = model.R_m1[m1_sv].detach().cpu().numpy()
    Rm1 = model.R_m1.detach().cpu().numpy()
    m2_sv = torch.where(torch.linalg.svdvals(model.R_m2) > threshold)
    #Rm2 = model.R_m2[m2_sv].detach().cpu().numpy()
    Rm2 = model.R_m2.detach().cpu().numpy()

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
    plt.savefig(f"{save_dir}/{model_name}_learned_matrices.png")
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
    plt.savefig(f"{save_dir}/{model_name}_matrix_correlations.png")
    plt.close()

def plot_representations_wandb(z_n, labels, l_ind=0):
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
        
        if data.shape[1] > 2:
            # Normal PCA case (2+ dimensions)
            pca = PCA(n_components=2)
            x = pca.fit_transform(data)
            df_plot = pd.DataFrame(x, columns=['x1', 'x2'])
            df_plot['label'] = labels[:, l_ind]
            sns.scatterplot(data=df_plot, x='x1', y='x2', hue='label', ax=ax)
        elif data.shape[1] == 2:
            # If already 2D, just plot directly
            df_plot = pd.DataFrame(data, columns=['x1', 'x2'])
            df_plot['label'] = labels[:, l_ind]
            sns.scatterplot(data=df_plot, x='x1', y='x2', hue='label', ax=ax)
        elif data.shape[1] == 1:
            # Handle 1D case by adding a zero column for visualization
            x = np.hstack([data, np.zeros_like(data)])
            ax.scatter(x[:, 0], x[:, 1], c=labels[:, l_ind])
        else:
            # If no valid features, just write "No Data"
            ax.text(0.5, 0.5, "No Data", horizontalalignment='center', verticalalignment='center')
    # return fig
    wandb.log({"Representation Plots": wandb.Image(fig)})
    plt.close(fig)

def plot_projection_matrices_wandb(model):

    # Get matrices
    Rs = model.R_s.detach().cpu().numpy()
    Rm1 = model.R_m1.detach().cpu().numpy()
    Rm2 = model.R_m2.detach().cpu().numpy()

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
    
    wandb.log({"Projection Matrices": wandb.Image(fig)})
    plt.close(fig)

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
    
    wandb.log({"Projection Matrix Correlations": wandb.Image(fig)})
    plt.close(fig)

def plot_losses(losses, loss_names, save_path=None, log_path=None):
    """Plot loss curves in separate horizontal subplots and save loss values."""
    # Convert losses to numpy array if it's not already
    losses = np.array(losses)
    
    # Calculate total loss
    total_loss = np.sum(losses, axis=1)
    all_losses = np.column_stack([losses, total_loss])
    all_names = loss_names + ['Total Loss']
    
    # Create figure with subplots
    num_losses = len(all_names)
    fig, axes = plt.subplots(1, num_losses, figsize=(5*num_losses, 5))
    
    # If there's only one loss, make axes iterable
    if num_losses == 1:
        axes = [axes]
    
    # Plot each loss in its own subplot
    for i, (ax, name) in enumerate(zip(axes, all_names)):
        ax.plot(all_losses[:, i], label=name, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss Value')
        ax.set_title(name)
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        # Save plot
        plt.savefig(save_path)
        plt.close()
        
        # Save loss values to CSV
        csv_path = log_path
        import pandas as pd
        df = pd.DataFrame(all_losses, columns=all_names)
        df.to_csv(csv_path, index_label='epoch')
    else:
        plt.show()

def plot_weights(weights, weight_names, save_path):
    """Plot the evolution of loss weights over time in separate subplots"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    weights = np.array(weights)
    epochs = np.arange(len(weights))
    
    # Create a figure with subplots
    fig, axes = plt.subplots(len(weight_names), 1, figsize=(10, 4*len(weight_names)))
    if len(weight_names) == 1:
        axes = [axes]  # Make it iterable if only one weight
    
    for i, (ax, name) in enumerate(zip(axes, weight_names)):
        ax.plot(epochs, weights[:, i], label=name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Weight Value')
        ax.set_title(f'Evolution of {name}')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()