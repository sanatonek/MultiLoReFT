import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import random


def generate_multimodal_data(n_samples, save_path='./data/', input_dims=[5,5], data_dims=[5,6], shared_dim=2, n_classes=2, class_location='shared'):
    """
    Generates a synthetic dataset with two modalities, each being a transform of shared and modality-specific information.
    
    Parameters:
    - n_samples: Number of data points
    - input_dim: Final representation dimension (for both modalities)
    - shared_dim: Dimensionality of shared information
    - mod1_dim: Dimensionality of modality 1 specific information
    - mod2_dim: Dimensionality of modality 2 specific information

    Returns:
    - h1: Tensor representing Modality 1 features
    - h2: Tensor representing Modality 2 features
    - labels: List indicating which Gaussian component each sample was drawn from
    """
    # Shared Information (2D) - Mixture of Two Gaussians

    if n_classes != 2:
        raise NotImplementedError("Only 2 classes are implemented.")
    
    if class_location == 'shared':
        means = np.random.randn(n_classes, shared_dim) * 2
        #cov = np.eye(shared_dim)
        cov = np.ones((shared_dim, shared_dim))
    
        labels = np.zeros((n_samples, 3))
        X_s = []
        # raise a not implemented error if n_classes > 2
        for i in range(n_samples):
            if np.random.rand() < 0.5:
                X_s.append(np.random.multivariate_normal(means[0], cov))
                labels[i,0]=0
            else:
                X_s.append(np.random.multivariate_normal(means[1], cov))
                labels[i,0]=1
        X_s = np.vstack(X_s)  # (n_samples, shared_dim)
    
        # Modality-specific Information
        mean_m1 = np.random.randn(input_dims[0] - shared_dim) * 3
        X_m1 = np.random.multivariate_normal(mean_m1, np.eye(input_dims[0] - shared_dim), size=n_samples) 
        
        mean_m2 = np.random.randn(input_dims[1] - shared_dim) * 3
        X_m2 = np.random.multivariate_normal(mean_m2, np.eye(input_dims[1] - shared_dim), size=n_samples)
    else:
        mean = np.random.randn(shared_dim) * 2
        cov = np.eye(shared_dim)
        X_s = np.random.multivariate_normal(mean, cov, size=n_samples)

        means_m1 = np.random.randn(n_classes, input_dims[0] - shared_dim) * 3
        X_m1 = []
        means_m2 = np.random.randn(n_classes, input_dims[1] - shared_dim) * 3
        X_m2 = []
        labels = np.zeros((n_samples, 3))
        for i in range(n_samples):
            if np.random.rand() < 0.5:
                X_m1.append(np.random.multivariate_normal(means_m1[0], np.eye(input_dims[0] - shared_dim)))
                X_m2.append(np.random.multivariate_normal(means_m2[0], np.eye(input_dims[1] - shared_dim)))
                labels[i,0]=0
            else:
                X_m1.append(np.random.multivariate_normal(means_m1[1], np.eye(input_dims[0] - shared_dim)))
                X_m2.append(np.random.multivariate_normal(means_m2[1], np.eye(input_dims[1] - shared_dim)))
                labels[i,0]=1
        X_m1 = np.vstack(X_m1)
        X_m2 = np.vstack(X_m2)
    
    labels[:,1] = X_m1.sum(-1)#(X_m1.sum(-1)>6).astype(int)
    labels[:,2] = X_m2.sum(-1)#(X_m2.sum(-1)<0).astype(int)

    # Transform Shared & Modality-Specific Information to Higher Dimensional Space
    W_m1 = np.random.randn(input_dims[0], data_dims[0]) * 0.1  # Modality 1 transformation
    W_m2 = np.random.randn(input_dims[1], data_dims[1]) * 0.1  # Modality 2 transformation

    X_1 = np.concatenate([X_m1, X_s], -1)
    X_2 = np.concatenate([X_m2, X_s], -1)

    # Final Representations for Each Modality
    h1 = np.dot(X_1, W_m1) # Modality 1 representation
    h2 = np.dot(X_2, W_m2)  # Modality 2 representation
    
    data_name = save_path + f"sim_{n_samples}_in{input_dims[0]}-{input_dims[1]}_data{data_dims[0]}-{data_dims[1]}_shared{shared_dim}_c{n_classes}_{class_location}.npz"
    np.savez_compressed(data_name, h1=h1, h2=h2, x1=X_1, x2=X_2, labels=labels)
    print(f"Dataset saved to {data_name}")
    return torch.tensor(h1, dtype=torch.float32), torch.tensor(h2, dtype=torch.float32), torch.tensor(X_1, dtype=torch.float32), torch.tensor(X_2, dtype=torch.float32), labels


def generate_simplest_multimodal_data(n_samples, save_path='./data/', version='v1'):
    """
    Generates a synthetic dataset with two modalities, each being a transform of shared and modality-specific information.
    
    Parameters:
    - n_samples: Number of data points
    - input_dim: Final representation dimension (for both modalities)
    - shared_dim: Dimensionality of shared information
    - mod1_dim: Dimensionality of modality 1 specific information
    - mod2_dim: Dimensionality of modality 2 specific information

    Returns:
    - h1: Tensor representing Modality 1 features
    - h2: Tensor representing Modality 2 features
    - labels: List indicating which Gaussian component each sample was drawn from
    """
    # Shared Information (2D) - Mixture of Two Gaussians

    if version == 'v1':
        means = np.array([[-2, -1], [5, 3]])
    elif version == 'v2':
        means = np.array([[0, 1], [0, 0]])
    cov = np.ones((2, 2))
    labels = np.zeros((n_samples, 3))
    X_s = []
    for i in range(n_samples):
        if np.random.rand() < 0.5:
            X_s.append(np.random.multivariate_normal(means[0], cov))
            labels[i,0]=0
        else:
            X_s.append(np.random.multivariate_normal(means[1], cov))
            labels[i,0]=1
    X_s = np.vstack(X_s)  # (n_samples, shared_dim)
    # Modality-specific Information
    mean_m1 = np.array([-2, 0])
    mean_m2 = np.array([6, 1])
    X_m1 = np.random.multivariate_normal(mean_m1, np.eye(2), size=n_samples)
    #X_m2 = np.random.multivariate_normal(mean_m2, np.eye(3), size=n_samples)
    # use a poisson distribution for modality 2
    X_m2 = np.random.poisson(mean_m2, size=(n_samples, 2))
    
    labels[:,1] = X_m1.sum(-1)#(X_m1.sum(-1)>6).astype(int)
    labels[:,2] = X_m2.sum(-1)#(X_m2.sum(-1)<0).astype(int)

    # Transform Shared & Modality-Specific Information to Higher Dimensional Space
    W_m1 = np.random.randn(4, 4) * 0.2  # Modality 1 transformation
    W_m2 = np.random.randn(4, 4) * 0.1  # Modality 2 transformation

    X_1 = np.concatenate([X_m1, X_s], -1)
    X_2 = np.concatenate([X_m2, X_s], -1)

    # Final Representations for Each Modality
    h1 = np.dot(X_1, W_m1) # Modality 1 representation
    h2 = np.dot(X_2, W_m2)  # Modality 2 representation
    
    if version == 'v1':
        data_name = save_path + "simplest_sim.npz"
    else:
        data_name = save_path + f"simplest_sim_{version}.npz"
    np.savez_compressed(data_name, h1=h1, h2=h2, x1=X_1, x2=X_2, labels=labels)
    print(f"Dataset saved to {data_name}")
    return torch.tensor(h1, dtype=torch.float32), torch.tensor(h2, dtype=torch.float32), torch.tensor(X_1, dtype=torch.float32), torch.tensor(X_2, dtype=torch.float32), labels

def generate_simplest_multimodal_data_nongaussian(n_samples, save_path='./data/', seed=5):
    """
    Generates a synthetic dataset with two modalities using non-Gaussian distributions.
    
    Parameters:
    - n_samples: Number of data points
    - save_path: Directory to save the data
    
    Returns:
    - h1: Tensor representing Modality 1 features
    - h2: Tensor representing Modality 2 features
    - X_1: Original features for Modality 1 
    - X_2: Original features for Modality 2
    - labels: Labels for classification and regression tasks
    """

    np.random.seed(seed)
    random.seed(seed)

    # Define dimensions
    n_hidden_shared = 2
    n_hidden_specific = [2, 2]
    n_out_features = [10, 10]  # Match original output dimensions
    
    # Helper function for sampling from different distributions
    def sample_hidden(n_samples, n_hidden, distrib):
        if distrib == 'binomial':
            return np.random.binomial(1, 0.5, size=(n_samples, n_hidden))
        elif distrib == 'poisson':
            return np.random.poisson(1, size=(n_samples, n_hidden)) + 1
        elif distrib == 'beta':
            return np.random.beta(3, 2, size=(n_samples, n_hidden))
        elif distrib == 'uniform':
            return np.random.uniform(0, 1, size=(n_samples, n_hidden))
        elif distrib == 'gumbel':
            return np.random.gumbel(0, 1, size=(n_samples, n_hidden))
        elif distrib == 'weibull':
            return np.random.weibull(1.5, size=(n_samples, n_hidden)) * 0.3
        else:
            raise ValueError(f"Unknown distribution: {distrib}")
    
    # Sample shared hidden variables (binomial distribution)
    shared_hidden_nonoise = sample_hidden(n_samples, n_hidden_shared, 'binomial')
    shared_hidden = shared_hidden_nonoise.copy() + np.random.normal(0, 0.01, size=(n_samples, n_hidden_shared))
    
    # Sample specific hidden variables (poisson for mod1, beta for mod2)
    X_m1 = sample_hidden(n_samples, n_hidden_specific[0], 'weibull')
    X_m2 = sample_hidden(n_samples, n_hidden_specific[1], 'beta')
    
    # Create labels
    labels = np.zeros((n_samples, 3))
    
    # Get class labels from shared hidden variables (based on unique combinations)
    shared_unique = np.unique(shared_hidden_nonoise, axis=0)
    for i, sh in enumerate(shared_hidden_nonoise):
        labels[i, 0] = np.where((shared_unique == sh).all(axis=1))[0][0]
    
    # Continuous labels from modality-specific variables
    labels[:, 1] = X_m1.sum(axis=1)
    labels[:, 2] = X_m2.sum(axis=1)
    
    # Sample projection matrices from uniform distribution
    W_m1 = np.random.uniform(-1, 1, size=(n_hidden_shared + n_hidden_specific[0], n_out_features[0]))
    W_m2 = np.random.uniform(-1, 1, size=(n_hidden_shared + n_hidden_specific[1], n_out_features[1]))
    
    # Combine shared and modality-specific information
    X_1 = np.concatenate([X_m1, shared_hidden], axis=1)
    X_2 = np.concatenate([X_m2, shared_hidden], axis=1)
    
    # Transform to output features
    h1 = X_1 @ W_m1
    h2 = X_2 @ W_m2
    
    # Save the dataset
    data_name = save_path + "simplest_sim_nongaussian.npz"
    np.savez_compressed(data_name, h1=h1, h2=h2, x1=X_1, x2=X_2, labels=labels)
    print(f"Dataset saved to {data_name}")
    
    # also save plots of the data
    # Create visualization directory if it doesn't exist
    plot_dir = os.path.join(save_path, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot hidden variable distributions
    def plot_hidden_distributions(hidden_vars, title):
        plt.figure(figsize=(10, 6))
        for i, hidden in enumerate(hidden_vars):
            plt.subplot(2, 3, i + 1)
            plt.hist(hidden.flatten(), bins=30, alpha=0.7)
            plt.title(f"{title} {i + 1}")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            # Scatter plot (pca if dim > 2)
            plt.subplot(2, 3, i + 4)
            if hidden.shape[1] > 2:
                pca = PCA(n_components=2)
                reduced = pca.fit_transform(hidden)
                plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.5)
            else:
                plt.scatter(hidden[:, 0], hidden[:, 1], alpha=0.5)
        plt.suptitle(title)
        plt.subplots_adjust(top=0.85)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{title.replace(' ', '_')}.png"))
        plt.close()
    
    # Plot output features
    def plot_output_features(features, title, c_vector=None):
        plt.figure(figsize=(10, 5))
        for i, feature in enumerate(features):
            plt.subplot(1, 2, i + 1)
            if feature.shape[1] > 2:
                pca = PCA(n_components=2)
                reduced = pca.fit_transform(feature)
                if c_vector is not None:
                    plt.scatter(reduced[:, 0], reduced[:, 1], c=c_vector, alpha=0.5)
                else:
                    plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.5)
            else:
                if c_vector is not None:
                    plt.scatter(feature[:, 0], feature[:, 1], c=c_vector, alpha=0.5)
                else:
                    plt.scatter(feature[:, 0], feature[:, 1], alpha=0.5)
            plt.title(f"{title} {i + 1}")
            plt.xlabel("Feature Dimension 1")
            plt.ylabel("Feature Dimension 2")
            plt.colorbar(label='Class')
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{title.replace(' ', '_')}.png"))
        plt.close()
    
    # Generate and save plots
    plot_hidden_distributions([shared_hidden, X_m1, X_m2], "sim_nongaussian_hidden_variables")
    plot_output_features([h1, h2], "sim_nongaussian_output_features_cs", c_vector=labels[:, 0])
    plot_output_features([h1, h2], "sim_nongaussian_output_features_cm1", c_vector=labels[:, 1])
    plot_output_features([h1, h2], "sim_nongaussian_output_features_cm2", c_vector=labels[:, 2])

    # Generate summary statistics
    mod1_mean = np.mean(h1, axis=1)
    mod2_mean = np.mean(h2, axis=1)
    
    # Additional plot: raw input features
    plot_output_features([X_1, X_2], "sim_nongaussian_raw_input_features_cs", c_vector=labels[:, 0])
    plot_output_features([X_1, X_2], "sim_nongaussian_raw_input_features_cm1", c_vector=labels[:, 1])
    plot_output_features([X_1, X_2], "sim_nongaussian_raw_input_features_cm2", c_vector=labels[:, 2])

    # calculate clustering accuracy of labels[:,0]
    from sklearn.metrics import adjusted_rand_score
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=4, random_state=seed)
    kmeans.fit(h1)
    clustering_labels = kmeans.labels_
    clustering_accuracy = adjusted_rand_score(labels[:, 0], clustering_labels)
    print(f"Clustering accuracy in modality 1: {clustering_accuracy:.4f}")
    kmeans.fit(h2)
    clustering_labels = kmeans.labels_
    clustering_accuracy = adjusted_rand_score(labels[:, 0], clustering_labels)
    print(f"Clustering accuracy in modality 2: {clustering_accuracy:.4f}")

    return torch.tensor(h1, dtype=torch.float32), torch.tensor(h2, dtype=torch.float32), torch.tensor(X_1, dtype=torch.float32), torch.tensor(X_2, dtype=torch.float32), labels