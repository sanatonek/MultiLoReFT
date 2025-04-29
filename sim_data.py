import numpy as np
import torch


def generate_multimodal_data(n_samples, mod_dim, save_path):
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
    mean1, mean2 = np.array([-1, 0]), np.array([1, 0])
    cov = np.array([[1, 1], [1, 1]])
    
    labels = np.zeros((n_samples, 3))
    X_s = []
    for i in range(n_samples):
        if np.random.rand() < 0.5:
            X_s.append(np.random.multivariate_normal(mean1, cov))
            labels[i,0]=0
        else:
            X_s.append(np.random.multivariate_normal(mean2, cov))
            labels[i,0]=1
    X_s = np.vstack(X_s)  # (n_samples, shared_dim)
    
    # Modality-specific Information
    mean_m1 = np.array([1, 2, 3])
    X_m1 = np.random.multivariate_normal(mean_m1, np.eye(3), size=n_samples) 
    labels[:,1] = X_m1.sum(-1)#(X_m1.sum(-1)>6).astype(int)
    
    mean_m2 = np.array([1, -2, 0, 2])
    X_m2 = np.random.multivariate_normal(mean_m2, np.eye(4), size=n_samples)
    labels[:,2] = X_m2.sum(-1)#(X_m2.sum(-1)<0).astype(int)

    # Transform Shared & Modality-Specific Information to Higher Dimensional Space
    W_m1 = np.random.randn(5, mod_dim) * 0.1  # Modality 1 transformation
    W_m2 = np.random.randn(6, mod_dim) * 0.1  # Modality 2 transformation

    X_1 = np.concat([X_m1, X_s], -1)
    X_2 = np.concat([X_m2, X_s], -1)

    # Final Representations for Each Modality
    h1 = np.dot(X_1, W_m1) # Modality 1 representation
    h2 = np.dot(X_2, W_m2)  # Modality 2 representation
    
    np.savez_compressed(save_path, h1=h1, h2=h2, x1=X_1, x2=X_2, labels=labels)
    print(f"Dataset saved to {save_path}")
    return torch.tensor(h1, dtype=torch.float32), torch.tensor(h2, dtype=torch.float32), torch.tensor(X_1, dtype=torch.float32), torch.tensor(X_2, dtype=torch.float32), labels
