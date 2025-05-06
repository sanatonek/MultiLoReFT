import numpy as np
import torch


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