import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import random
import torch.nn as nn


def generate_multimodal_data_apollo(n_samples, save_path):
    """
    Generates synthetic data with two modalities (h1, h2) and labels derived from latent variables Z₁–Z₅.

    - Z₁, Z₃: Binary labels from Bernoulli(0.5)
    - Z₂: Noisy continuous version of Z₁
    - h1 (Z₄): Depends on Z₂ and Z₃ + small noise
    - h2 (Z₅): Noisy version of Z₂

    Returns:
    - h1: Modality 1 tensor (n_samples, mod_dim)
    - h2: Modality 2 tensor (n_samples, mod_dim)
    - x1: Raw input to h1 transformation (Z₂, Z₃)
    - x2: Raw input to h2 transformation (Z₂ only)
    - labels: [Z₁, Z₃]
    """

    np.random.seed(42)

    # # Step 1: Generate Z₁ ~ Ber(0.5)
    # Z1 = np.random.binomial(n=1, p=0.5, size=n_samples)
    # # Step 2: Z₂ = Z₁ + Gamma(5, 1) * sqrt(0.0045)
    # Z2 = Z1 + np.random.gamma(shape=5, scale=1, size=n_samples) * np.sqrt(0.0045)
    # # Step 3: Z₃ ~ Ber(0.5)
    # Z3 = np.random.binomial(n=1, p=0.5, size=n_samples)
    # # Step 4: Z₄ = 2 * Z₂ + Z₃ + Gamma(2, 2) * sqrt(0.00125)
    # Z4 = 2 * Z2 + Z3 + np.random.gamma(shape=2, scale=2, size=n_samples) * np.sqrt(0.00125)
    # # Step 5: Z₅ = Z₂ + Gamma(3, 2) * sqrt(0.0075)
    # Z5 = Z2 + np.random.gamma(shape=3, scale=2, size=n_samples) * np.sqrt(0.0075)

    Z1 = np.random.binomial(1, 0.5, size=n_samples)

    gamma2 = np.random.gamma(shape=5, scale=1, size=n_samples)
    Z2 = Z1 + gamma2 * np.sqrt(0.0045) 

    Z3 = np.random.binomial(1, 0.5, size=n_samples)

    gamma4 = np.random.gamma(shape=2, scale=2, size=n_samples)
    Z4 = 2 * Z2 + Z3 + gamma4 * np.sqrt(0.00125)

    gamma5 = np.random.gamma(shape=3, scale=2, size=n_samples)
    Z5 = Z2 + gamma5 * np.sqrt(0.0075)

    # Prepare labels: columns [Z1, Z3]
    labels = np.stack([Z1, Z3, Z5], axis=1).astype(np.float32)

    # random_matrix = np.random.rand(5, 10)  # Create a random matrix of shape (5, 10)
    # h1 = np.dot(np.stack([Z1, Z2, Z3, Z4, Z5], axis=1), random_matrix)  # Multiply stack by random matrix
    # random_matrix = np.random.rand(3, 10)  # Create a random matrix of shape (5, 10)
    # h2 = np.dot(np.stack([Z1, Z2, Z5], axis=1), random_matrix)  # h2 from Z2 only
    h1 = np.zeros((n_samples, 80))
    h2 = np.zeros((n_samples, 40))

    Z_X = np.stack([Z1, Z2, Z3, Z4], axis=1)
    Z_Y = np.stack([Z1, Z2, Z5], axis=1)

    # X1-X10: pure children of Z4
    G1 = np.random.randn(10, 1)
    h1[:, 0:10] = Z_X[:, [3]] @ G1.T
    # X11-X20: pure children of Z3
    G2 = np.random.randn(10, 1)
    h1[:, 10:20] = Z_X[:, [2]] @ G2.T
    # X21-X30: pure children of Z2
    G3 = np.random.randn(10, 1)
    h1[:, 20:30] = Z_X[:, [1]] @ G3.T
    # X31-X40: children of Z2 and Z3
    G4 = np.random.randn(10, 2)
    h1[:, 30:40] = Z_X[:, [1, 2]] @ G4.T
    # X41-X50: children of Z2 and Z4
    G5 = np.random.randn(10, 2)
    h1[:, 40:50] = Z_X[:, [1, 3]] @ G5.T
    # X51-X60: children of Z3 and Z4
    G6 = np.random.randn(10, 2)
    h1[:, 50:60] = Z_X[:, [2, 3]] @ G6.T
    # X61-X70: children of Z2, Z3, Z4
    G7 = np.random.randn(10, 3)
    h1[:, 60:70] = Z_X[:, [1, 2, 3]] @ G7.T
    G8 = np.random.randn(10, 4)
    h1[:, 70:80] = Z_X @ G8.T



    GY1 = np.random.randn(10, 1)
    h2[:, 0:10] = Z_Y[:, [2]] @ GY1.T
    GY2 = np.random.randn(10, 1)
    h2[:, 10:20] = Z_Y[:, [1]] @ GY2.T
    GY3 = np.random.randn(10, 2)
    h2[:, 20:30] = Z_Y[:, [1, 2]] @ GY3.T
    GY4 = np.random.randn(10, 3)
    h2[:, 30:40] = Z_Y @ GY4.T

    # Save dataset
    np.savez_compressed(save_path, h1=h1, h2=h2, x1=h1, x2=h2, labels=labels)
    print(f"Dataset saved to {save_path}")

    return (
        torch.tensor(h1, dtype=torch.float32),
        torch.tensor(h2, dtype=torch.float32),
        torch.tensor(h1, dtype=torch.float32),
        torch.tensor(h2, dtype=torch.float32),
        labels
    )


def generate_simplest_multimodal_data_nongaussian(n_samples, save_path='./data/', seed=4):
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
    n_hidden_specific = [2,2]
    n_out_features = [10, 10]  # Match original output dimensions
    
    # Helper function for sampling from different distributions
    def sample_hidden(n_samples, n_hidden, distrib):
        if distrib == 'binomial':
            return np.random.binomial(1, 0.5, size=(n_samples, n_hidden))
        elif distrib == 'poisson':
            return np.random.poisson(1, size=(n_samples, n_hidden)) + 1
        elif distrib == 'beta':
            return np.random.beta(3, 2, size=(n_samples, n_hidden)) * 1
        elif distrib == 'uniform':
            return np.random.uniform(0, 1, size=(n_samples, n_hidden))
        elif distrib == 'gumbel':
            return np.random.gumbel(0, 1, size=(n_samples, n_hidden))
        elif distrib == 'weibull':
            return np.random.weibull(1.5, size=(n_samples, n_hidden))*0.3
        else:
            raise ValueError(f"Unknown distribution: {distrib}")
    
    # Sample shared hidden variables (binomial distribution)
    shared_hidden_nonoise = sample_hidden(n_samples, n_hidden_shared, 'binomial')
    shared_hidden = shared_hidden_nonoise.copy() + np.random.normal(0, 0.01, size=(n_samples, n_hidden_shared))
    
    # Sample specific hidden variables (poisson for mod1, beta for mod2)
    X_m1 = sample_hidden(n_samples, n_hidden_specific[0], 'weibull')
    X_m2 = sample_hidden(n_samples, n_hidden_specific[1], 'beta')
    
    # Create labels
    labels = np.zeros((n_samples, 4))
    
    # Get class labels from shared hidden variables (based on unique combinations)
    shared_unique = np.unique(shared_hidden_nonoise, axis=0)
    for i, sh in enumerate(shared_hidden_nonoise):
        labels[i, 0] = np.where((shared_unique == sh).all(axis=1))[0][0]
    
    # Continuous labels from modality-specific variables
    m1_thresh = np.quantile(X_m1, 0.5, axis=0)  # per-dim median
    m2_thresh = np.quantile(X_m2, 0.5, axis=0)

    X_m1_code = (X_m1 > m1_thresh).astype(np.int8)  # shape: (n_samples, n_hidden_specific[0])
    X_m2_code = (X_m2 > m2_thresh).astype(np.int8)  # shape: (n_samples, n_hidden_specific[1])

    _, labels[:, 1] = np.unique(X_m1_code, axis=0, return_inverse=True)
    _, labels[:, 2] = np.unique(X_m2_code, axis=0, return_inverse=True)

    # -------------------------
    # Label 3 (joint): bucket unique (label0,label1,label2) triples into fewer classes
    # -------------------------
    n_joint_classes = 8  

    triple = np.stack([labels[:, 0], labels[:, 1], labels[:, 2]], axis=1)
    uniq_triples, inv = np.unique(triple, axis=0, return_inverse=True)
    n_joint_classes = min(n_joint_classes, len(uniq_triples))

    # Deterministic (seeded) random assignment of each unique triple to a bucket
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq_triples))

    # Map each sample's triple -> permuted ID -> bucket
    labels[:, 3] = (perm[inv] % n_joint_classes).astype(np.int64)

    # labels[:, 1] = X_m1.sum(axis=1)
    # labels[:, 2] = X_m2.sum(axis=1)
    # labels[:,3] = np.where(labels[:,0] == 1, X_m1.sum(axis=1), X_m2.sum(axis=1))
    
    # Sample projection matrices from uniform distribution
    W_m1 = np.random.uniform(-1, 1, size=(n_hidden_shared + n_hidden_specific[0], n_out_features[0]))
    W_m2 = np.random.uniform(-1, 1, size=(n_hidden_shared + n_hidden_specific[1], n_out_features[1]))
    
    # Combine shared and modality-specific information
    X_1 = np.concatenate([X_m1, shared_hidden], axis=1)
    X_2 = np.concatenate([X_m2, shared_hidden], axis=1)
    
    
    x1_encoder = nn.Sequential(
        nn.Linear(n_hidden_specific[0]+n_hidden_shared,  n_out_features[0]),
    )
    x2_encoder = nn.Sequential(
        nn.Linear(n_hidden_specific[1]+n_hidden_shared, n_out_features[1]),
    )

    h1 = x1_encoder(torch.tensor(X_1, dtype=torch.float32)).detach().numpy()
    h2 = x2_encoder(torch.tensor(X_2, dtype=torch.float32)).detach().numpy()

    torch.save(x1_encoder, "./ckpts/simulated_x1_encoder.pth")
    torch.save(x2_encoder, "./ckpts/simulated_x2_encoder.pth")
    print(f"Encoders saved to ./ckpts/")
    
    # Save the dataset
    data_name = save_path + "simplest_sim_nongaussian.npz"
    np.savez_compressed(data_name, h1=h1, h2=h2, x1=X_1, x2=X_2, labels=labels)
    print(h1.shape, h2.shape, X_1.shape, X_2.shape, labels.shape)
    print(f"Dataset saved to {data_name}")
    
    # also save plots of the data
    # Create visualization directory if it doesn't exist
    plot_dir = "./plots"
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
                unique_values = np.unique(c_vector)[:min(10, len(np.unique(c_vector)))]
                # print(f"Number of unique values in c_vector: {unique_values_count}")
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
    plot_output_features([X_m1, X_m2], "sim_nongaussian_raw_input_features_cs", c_vector=labels[:, 0])
    plot_output_features([X_m1, X_m2], "sim_nongaussian_raw_input_features_cm1", c_vector=labels[:, 1])
    plot_output_features([X_m1, X_m2], "sim_nongaussian_raw_input_features_cm2", c_vector=labels[:, 2])

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


save_path = "./data/"
if not os.path.exists(save_path):
    os.makedirs(save_path)
generate_simplest_multimodal_data_nongaussian(n_samples=6000, save_path=save_path, seed=4)
generate_multimodal_data_apollo(n_samples=60000, save_path=save_path+"simulated_data_apollo.npz")