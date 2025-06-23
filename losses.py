import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class AdaptiveLossWeights(nn.Module):
    """Adaptive Loss Weighting based on Kendall et al. (2018)."""
    def __init__(self, num_losses):
        super().__init__()
        self.log_sigmas = nn.Parameter(torch.zeros(num_losses))
    
    def forward(self, losses):
        """
        Compute weighted loss using learned weights.
        
        Args:
            losses: List of loss values [l1, l2, l3, ...]
            
        Returns:
            weighted_loss: Combined weighted loss
            weights: List of learned weights
        """
        losses = torch.stack([l if isinstance(l, torch.Tensor) else torch.tensor(l) for l in losses])
        weights = 1.0 / (2 * torch.exp(self.log_sigmas))
        weighted_loss = torch.sum(weights * losses) + torch.sum(self.log_sigmas)
        return weighted_loss, weights

class GradientNormalizedLoss:
    """Gradient Normalization for loss balancing (Chen et al., 2018)."""
    def __init__(self, num_losses):
        self.num_losses = num_losses
        self.running_losses = torch.zeros(num_losses)
        self.running_count = 0
    
    def __call__(self, losses, trainable_params):
        """
        Compute gradient-normalized loss weights.
        
        Args:
            losses: List of loss values
            model: The model being trained
            trainable_params: List of trainable parameters
            
        Returns:
            weighted_loss: Combined weighted loss
            weights: List of gradient-normalized weights
        """
        losses = torch.stack([l if isinstance(l, torch.Tensor) else torch.tensor(l) for l in losses])
        
        # Compute gradients for each loss
        grads = []
        for loss in losses:
        #     model.zero_grad()
            loss.backward(retain_graph=True)
            
            # Collect gradients for trainable parameters
            param_grads = []
            for p in trainable_params:
                if p.grad is not None:
                    param_grads.append(p.grad.view(-1))
            
            if param_grads:
                grad = torch.cat(param_grads)
                grads.append(grad)
            else:
                grads.append(torch.zeros_like(trainable_params[0].view(-1)))
        
        # Normalize gradients and compute weights
        grad_norms = [torch.norm(g) for g in grads]
        weights = [1.0 / (norm + 1e-8) for norm in grad_norms]

        # Normalize weights so they sum to 1
        weights = torch.tensor(weights)
        weights = weights / weights.sum()

        weighted_loss = sum(w * l for w, l in zip(weights, losses))
        return weighted_loss, weights

def loss_reconstruction(h, x, decoder):
    """Compute reconstruction loss between input and decoded representation."""
    return F.l1_loss(x, decoder(h))

def loss_mutual_info(h1, h2, z_components, all=True):
    """
    Maximize mutual information between original and projected representations.
    Supports mismatched dimensions by projecting to a common space.
    """
    if all:
    # Concatenate modality-specific and shared components
        z1 = torch.cat([z_components[0][0], z_components[1][1]], dim=1) 
        z2 = torch.cat([z_components[1][0], z_components[0][1]], dim=1) 
    else:
        z1 = z_components[1][1]
        z2 = z_components[0][1]
    
    # Normalize representations
    h1 = F.normalize(h1, dim=1)
    h2 = F.normalize(h2, dim=1)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # Handle dimension mismatch
    if h1.shape[1] != z1.shape[1]:
        proj_dim = min(h1.shape[1], z1.shape[1])
        h1 = F.normalize(h1[:, :proj_dim], dim=1)
        z1 = F.normalize(z1[:, :proj_dim], dim=1)
    # Handle dimension mismatch
    if h2.shape[1] != z2.shape[1]:
        proj_dim = min(h2.shape[1], z2.shape[1])
        h2 = F.normalize(h2[:, :proj_dim], dim=1)
        z2 = F.normalize(z2[:, :proj_dim], dim=1)
    
    # Compute InfoNCE-style similarity
    temp = 0.1
    logits1 = torch.mm(h1, z1.T) / temp
    logits2 = torch.mm(h2, z2.T) / temp
    
    # Compute InfoNCE losses
    loss1 = -torch.mean(torch.diagonal(logits1) - torch.logsumexp(logits1, dim=1))
    loss2 = -torch.mean(torch.diagonal(logits2) - torch.logsumexp(logits2, dim=1))
    return (loss1 + loss2) / 2

def loss_invariance_m(phi1, phi2, model):
    """
    Ensure invariance of modality-specific representations under shared space transformations.
    """
    perm = torch.randperm(phi1.size(0))
    h1_shuffled = phi1[perm]
    h2_shuffled = phi2[perm]
    
    phi1_recons = phi1 + torch.matmul(
        (torch.matmul(h1_shuffled, model.R_s.T) - torch.matmul(phi1, model.R_s.T)),
        model.R_s
    )
    phi2_recons = phi2 + torch.matmul(
        (torch.matmul(h2_shuffled, model.R_s.T) - torch.matmul(phi2, model.R_s.T)),
        model.R_s
    )
    
    return F.mse_loss(
        torch.matmul(phi1, model.R_m[0].T),
        torch.matmul(phi1_recons, model.R_m[0].T)
    ) + F.mse_loss(
        torch.matmul(phi2, model.R_m[1].T),
        torch.matmul(phi2_recons, model.R_m[1].T)
    )

def contrastive_alignment(phi1, phi2, model):
    """
    Ensure alignment between shared representations of different modalities.
    """
    phi1_recons = phi1 + torch.matmul(
        (torch.matmul(phi2, model.R_s.T) - torch.matmul(phi1, model.R_s.T)),
        model.R_s
    )
    phi2_recons = phi2 + torch.matmul(
        (torch.matmul(phi1, model.R_s.T) - torch.matmul(phi2, model.R_s.T)),
        model.R_s
    )
    return F.mse_loss(phi1_recons, phi1) + F.mse_loss(phi2_recons, phi2)

def loss_orthonormality(R_s, R_m1, R_m2):
    """
    Ensure orthonormality of projection matrices.
    """
    R1 = torch.concat([R_m1, R_s], 0)
    R2 = torch.concat([R_m2, R_s], 0)
    
    l1 = torch.norm(torch.mm(R1, R1.T), p="fro")
    l2 = torch.norm(torch.mm(R2, R2.T), p="fro")
    l3 = torch.norm(torch.mm(R_m1, R_m2.T), p="fro")
    
    return l1 + l2 + l3

def loss_full_rank(R_s, R_m1, R_m2):
    """
    Ensure full-rank behavior in the combined projection matrices.
    """
    R1 = torch.cat([R_s, R_m1, R_m2], dim=0)
    singular_values = torch.svd(R1)[1]
    return torch.sum(1.0 / (torch.abs(singular_values) + 1e-6))

def loss_low_rank_frobenius(*mats, weight=1e-3):
    """Compute Frobenius norm loss for low-rank regularization."""
    return weight * sum(torch.norm(mat, p='fro') for mat in mats)

def loss_cross_covariance(z_s1, z_s2, z_m1, z_m2):
    """
    VICReg-style covariance loss to ensure decorrelation between shared and modality-specific embeddings.
    """
    loss = 0
    batch_size = z_s1.shape[0]
    
    for Z1, Z2 in [(z_s1, z_m1), (z_s2, z_m2)]:
        # Center representations
        Z1 = Z1 - Z1.mean(dim=0, keepdim=True)
        Z2 = Z2 - Z2.mean(dim=0, keepdim=True)
        
        # Compute covariance
        cov = torch.matmul(Z1.T, Z2) / (batch_size - 1)
        loss += torch.norm(cov, p="fro")
    
    return loss

def loss_orthogonality(R_s, R_m1, R_m2):
    """
    Ensure orthogonality between shared and modality-specific spaces.
    """
    # loss_ortho_1 = torch.norm(torch.mm(R_s, R_m1.T), p="fro")**2/ R_s.numel()
    # loss_ortho_2 = torch.norm(torch.mm(R_m1, R_m2.T), p="fro")**2/ R_m1.numel()
    # loss_ortho_3 = torch.norm(torch.mm(R_s, R_m2.T), p="fro")**2/ R_s.numel()
    # return loss_ortho_1 + loss_ortho_2 + loss_ortho_3
    def safe_normalize(x):
        return x / (x.norm(dim=-1, keepdim=True) + 1e-8)

    R_s = safe_normalize(R_s)
    R_m1 = safe_normalize(R_m1)
    R_m2 = safe_normalize(R_m2)

    # Use mean of squared cosine similarities instead of Frobenius norm directly
    def ortho_pair(A, B):
        prod = torch.mm(A, B.T)
        return (prod ** 2).mean()

    loss_ortho_1 = ortho_pair(R_s, R_m1)
    loss_ortho_2 = ortho_pair(R_m1, R_m2)
    loss_ortho_3 = ortho_pair(R_s, R_m2)

    return (loss_ortho_1 + loss_ortho_2 + loss_ortho_3)

def loss_shared_consistency(z_s1, z_s2):
    """
    Ensure consistency between shared representations of different modalities.
    """
    # Normalize each representation
    # z_s1_norm = F.normalize(z_s1, p=2, dim=1)
    # z_s2_norm = F.normalize(z_s2, p=2, dim=1)
    
    # # Compute MSE between normalized representations
    # return F.mse_loss(z_s1_norm, z_s2_norm)
    
    # Center representations
    z_s1_centered = z_s1 - z_s1.mean(dim=0, keepdim=True)
    z_s2_centered = z_s2 - z_s2.mean(dim=0, keepdim=True)
    
    # Compute correlation matrix
    batch_size = z_s1.shape[0]
    corr = torch.matmul(z_s1_centered.T, z_s2_centered) / (batch_size - 1)
    
    # Normalize by standard deviations
    std1 = torch.sqrt(torch.var(z_s1_centered, dim=0, unbiased=True) + 1e-8)
    std2 = torch.sqrt(torch.var(z_s2_centered, dim=0, unbiased=True) + 1e-8)
    corr = corr / torch.outer(std1, std2)
    
    # Return negative correlation to minimize
    return -torch.mean(torch.diagonal(corr))
    # return 1 - linear_cka(z_s1, z_s2)
    # cos_similarity = 1 - F.cosine_similarity(z_s1, z_s2, dim=1)
    # return torch.mean(cos_similarity)

def _rbf_kernel(x: torch.Tensor, sigma: float = None, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute the RBF (Gaussian) kernel matrix.
    """
    N = x.shape[0]
    x_norm = (x ** 2).sum(dim=1).view(N, 1)
    dist_sq = x_norm + x_norm.T - 2 * (x @ x.T)

    if sigma is None:
        # Use median heuristic for bandwidth
        dist = dist_sq.detach().clone()
        dist = dist[~torch.eye(N, dtype=bool, device=x.device)]
        sigma = torch.sqrt(torch.median(dist) + eps)

    gamma = 1.0 / (2 * sigma ** 2 + eps)
    K = torch.exp(-gamma * dist_sq)
    return K

def hsic_rbf(x: torch.Tensor, y: torch.Tensor, sigma_x: float = None, sigma_y: float = None, unbiased: bool = True) -> torch.Tensor:
    """
    HSIC with Gaussian RBF kernel.
    
    Args:
        x: (N, d_x)
        y: (N, d_y)
        sigma_x: bandwidth for RBF kernel on x
        sigma_y: bandwidth for RBF kernel on y
        unbiased: whether to use unbiased estimator
    
    Returns:
        Scalar HSIC value
    """
    N = x.shape[0]
    assert N == y.shape[0], "x and y must have the same number of samples"

    K = _rbf_kernel(x, sigma_x)
    L = _rbf_kernel(y, sigma_y)

    if unbiased:
        K = K - torch.diag_embed(torch.diagonal(K, dim1=-2, dim2=-1))
        L = L - torch.diag_embed(torch.diagonal(L, dim1=-2, dim2=-1))

        hsic = torch.sum(K * L) / (N * (N - 3)) \
             - 2 * torch.sum(K.sum(dim=0) * L.sum(dim=0)) / (N * (N - 2) * (N - 3)) \
             + torch.sum(K) * torch.sum(L) / (N * (N - 1) * (N - 2) * (N - 3))
    else:
        H = torch.eye(N, device=x.device) - (1.0 / N) * torch.ones((N, N), device=x.device)
        Kc = H @ K @ H
        Lc = H @ L @ H
        hsic = torch.trace(Kc @ Lc) / ((N - 1) ** 2)

    return hsic

def hsic(x, y, sigma_x=None, sigma_y=None):
    """
    Compute HSIC between two tensors.
    Args:
        x: [n, d1]
        y: [n, d2]
    Returns:
        Scalar HSIC value.
    """
    n = x.size(0)
    K = rbf_kernel(x, sigma_x)
    L = rbf_kernel(y, sigma_y)

    H = torch.eye(n, device=x.device) - (1.0 / n) * torch.ones((n, n), device=x.device)
    Kc = torch.mm(H, torch.mm(K, H))
    Lc = torch.mm(H, torch.mm(L, H))
    hsic_val = torch.trace(torch.mm(Kc, Lc)) / ((n - 1)**2)
    return hsic_val

def loss_independence(z_s1, z_s2, z_m1, z_m2):
    """
    Compute independence loss between shared and modality-specific representations.
    """
    return hsic_rbf(z_s1, z_m1) + hsic_rbf(z_s2, z_m2)+ hsic_rbf(z_m1, z_m2)


def center_gram(gram):
    """Center a Gram matrix."""
    n = gram.size(0)
    unit = torch.ones(n, n, device=gram.device)
    identity = torch.eye(n, device=gram.device)
    H = identity - unit / n
    return torch.matmul(H, torch.matmul(gram, H))

def linear_cka(z1, z2):
    """
    Compute linear CKA between two representations z1 and z2.
    """
    # Centered Gram matrices
    K = center_gram(torch.matmul(z1, z1.T))
    L = center_gram(torch.matmul(z2, z2.T))

    # HSIC (Hilbert-Schmidt Independence Criterion)
    hsic = torch.sum(K * L)

    # Normalization
    norm_K = torch.norm(K)
    norm_L = torch.norm(L)

    return hsic / (norm_K * norm_L + 1e-8)


