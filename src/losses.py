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
    
    def __call__(self, losses, trainable_params=None, weights=None):
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
        #grads = []
        #for loss in losses:
        #     model.zero_grad()
        #    loss.backward(retain_graph=True)
            
            # Collect gradients for trainable parameters
        #    param_grads = []
        #    for p in trainable_params:
        #        if p.grad is not None:
        #            param_grads.append(p.grad.view(-1))
            
        #    if param_grads:
        #        grad = torch.cat(param_grads)
        #        grads.append(grad)
        #    else:
        #        grads.append(torch.zeros_like(trainable_params[0].view(-1)))
        
        # Normalize gradients and compute weights
        #grad_norms = [torch.norm(g) for g in grads]
        #weights = [1.0 / (norm + 1e-8) for norm in grad_norms]

        # Normalize weights so they sum to 1
        #weights = torch.tensor(weights)
        #weights = weights / weights.sum()

        if weights is None:
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

'''
def loss_mutual_info(h1, h2, z_components, all=True):
    """
    Maximize mutual information between original and projected representations.
    Supports mismatched dimensions by projecting to a common space.
    """
    # Concatenate modality-specific and shared components
    if all:
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
'''

def loss_mutual_info(h1, h2, z_components, mode="all"):
    """
    Maximize mutual information between original and projected representations.
    Supports mismatched dimensions by projecting to a common space.
    """
    class FixedProjector(torch.nn.Module):
        def __init__(self, d_in, k, ortho=True, seed=0):
            super().__init__()
            g = torch.Generator().manual_seed(seed)
            W = torch.randn(d_in, k, generator=g) / (d_in**0.5)
            if 0:#ortho:
                # QR for approximate orthonormal columns
                Q, _ = torch.linalg.qr(W, mode='reduced')
                W = Q
            self.register_buffer('W', W, persistent=False)  # not learnable
        def forward(self, x):
            return x @ self.W  # [B,k]
    #if all:
    # Concatenate modality-specific and shared components
    #    z1 = torch.cat([z_components[0][0], z_components[1][1]], dim=1)
    #    z2 = torch.cat([z_components[1][0], z_components[0][1]], dim=1)
    #else:
    #    z1 = z_components[1][1]
    #    z2 = z_components[0][1]

    class FixedProjector(torch.nn.Module):
        def __init__(self, d_in, k, ortho=True, seed=0):
            super().__init__()
            g = torch.Generator().manual_seed(seed)
            W = torch.randn(d_in, k, generator=g) / (d_in**0.5)
            if 0:#ortho:
                # QR for approximate orthonormal columns
                Q, _ = torch.linalg.qr(W, mode='reduced')
                W = Q
            self.register_buffer('W', W, persistent=False)  # not learnable

        def forward(self, x):
            return x @ self.W  # [B,k]
    
    if mode=="all":
        rnd = 1#torch.randint(0, 2, (1,)).item()
        z1 = torch.cat([z_components[rnd][1], z_components[0][0]], dim=1) 
        z2 = torch.cat([z_components[1-rnd][1], z_components[1][0]], dim=1) 
    elif mode=="shared":
        rnd = torch.randint(0, 2, (1,)).item()
        z1 = z_components[rnd][1]
        z2 = z_components[1-rnd][1]
    elif mode=="private":
        z1 = z_components[1][0]
        z2 = z_components[0][0]
    # Handle dimension mismatch
    if h1.shape[1] != z1.shape[1]:
        proj_dim = max(h1.shape[1], z1.shape[1])
        if h1.shape[1] < proj_dim:
            # padding = torch.zeros(h1.size(0), proj_dim - h1.shape[1], device=h1.device)
            # h1 = torch.cat((h1, padding), dim=1)
            h1 = FixedProjector(h1.shape[1], k=proj_dim, seed=123).to(h1.device)(h1)
        if z1.shape[1] < proj_dim:
            # padding = torch.zeros(z1.size(0), proj_dim - z1.shape[1], device=z1.device)
            # z1 = torch.cat((z1, padding), dim=1)
            z1 = FixedProjector(z1.shape[1], k=proj_dim, seed=223).to(z1.device)(z1)
    if h2.shape[1] != z2.shape[1]:
        proj_dim = max(h2.shape[1], z2.shape[1])
        if h2.shape[1] < proj_dim:
            # padding = torch.zeros(h2.size(0), proj_dim - h2.shape[1], device=h2.device)
            # h2 = torch.cat((h2, padding), dim=1)
            h2 = FixedProjector(h2.shape[1], k=proj_dim, seed=124).to(h2.device)(h2)
        if z2.shape[1] < proj_dim:
            # padding = torch.zeros(z2.size(0), proj_dim - z2.shape[1], device=z2.device)
            # z2 = torch.cat((z2, padding), dim=1)
            z2 = FixedProjector(z2.shape[1], k=proj_dim, seed=224).to(z2.device)(z2)
    # Normalize representations
    h1 = F.normalize(h1, dim=1)
    h2 = F.normalize(h2, dim=1)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    # Compute InfoNCE-style similarity
    temp = 0.1
    # logits1 = torch.mm(h1, z1.T) / temp
    # logits2 = torch.mm(h2, z2.T) / temp
    logits1 = (h1 @ z1.T) / 0.1
    logits2 = (h2 @ z2.T) / 0.1
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
    #def safe_normalize(x):
    #    return x / (x.norm(dim=-1, keepdim=True) + 1e-8)

    #R_s = safe_normalize(R_s)
    #R_m1 = safe_normalize(R_m1)
    #R_m2 = safe_normalize(R_m2)

    # Use mean of squared cosine similarities instead of Frobenius norm directly
    def ortho_pair(A, B):
        A = A / (A.norm(dim=1, keepdim=True) + 1e-6)
        B = B / (B.norm(dim=1, keepdim=True) + 1e-6)
        prod = torch.mm(A, B.T)
        return (prod ** 2).mean()

    loss_ortho_1 = ortho_pair(R_s, R_m1)
    loss_ortho_2 = ortho_pair(R_m1, R_m2)
    loss_ortho_3 = ortho_pair(R_s, R_m2)

    return (loss_ortho_1 + loss_ortho_2 + loss_ortho_3) # Sana changed loss 2 to three, but I am not sure if that is correct cause then there is loss 3 twice

def loss_shared_consistency(z_s1, z_s2):
    """
    Ensure consistency between shared representations of different modalities.
    """
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
    # cos_similarity = 1 - F.cosine_similarity(z_s1, z_s2, dim=1)
    # return torch.mean(cos_similarity)

def loss_reconstruction_m(x, z, decoder):
    """Compute reconstruction loss between input and decoded representation."""
    return

def loss_reconstruction_m(x, z, decoder):
    """Compute reconstruction loss between input and decoded representation."""

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

def rbf_kernel(x, sigma=None, eps=1e-12, return_sigma=True):
    n = x.size(0)
    x_norm = (x ** 2).sum(dim=1, keepdim=True)
    dist_sq = (x_norm + x_norm.t() - 2 * (x @ x.t())).clamp_min_(0)
    if sigma is None:
        n = x.size(0)
        iu = torch.triu_indices(n, n, 1, device=x.device)
        dists = dist_sq[iu[0], iu[1]]
        nz = dists[dists > 0]
        median_val = torch.median(nz) if nz.numel() > 0 else torch.mean(dists)
        sigma = torch.sqrt(median_val / 2.0 + eps)

    K = torch.exp(-dist_sq / (2.0 * (sigma ** 2) + eps))
    return K, sigma

def hsic_rbf(X, Y, sigma_x=None, sigma_y=None, unbiased=False):
    K, sigma_x = rbf_kernel(X, sigma=sigma_x) # <- unpack
    L, sigma_y = rbf_kernel(Y, sigma=sigma_y)  # <- unpack
    n = X.size(0)
    H = torch.eye(n, device=X.device, dtype=X.dtype) - (1.0 / n)
    Kc = H @ K @ H
    Lc = H @ L @ H

    if unbiased:
        hsic = torch.trace(Kc @ Lc) / ((n - 3) * (n - 2) + 1e-12)  # or your preferred unbiased estimator
    else:
        hsic = torch.trace(Kc @ Lc) / ((n - 1) ** 2 + 1e-12)
    return hsic

def hsic_linear(X, Y, unbiased=False):
    """
    Linear HSIC (i.e., cross-covariance Frobenius norm squared).
    X: (n, d_x)
    Y: (n, d_y)
    """
    n = X.size(0)
    # Center the features
    Xc = X - X.mean(dim=0, keepdim=True)
    Yc = Y - Y.mean(dim=0, keepdim=True)

    # Linear kernels = centered Gram matrices
    Kc = Xc @ Xc.T
    Lc = Yc @ Yc.T

    if unbiased:
        hsic = torch.trace(Kc @ Lc) / ((n - 3) * (n - 2) + 1e-12)
    else:
        hsic = torch.trace(Kc @ Lc) / ((n - 1)**2 + 1e-12)

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
    K = rbf_kernel(x, sigma_x)[0]
    L = rbf_kernel(y, sigma_y)[0]

    H = torch.eye(n, device=x.device) - (1.0 / n) * torch.ones((n, n), device=x.device)
    Kc = torch.mm(H, torch.mm(K, H))
    Lc = torch.mm(H, torch.mm(L, H))
    hsic_val = torch.trace(torch.mm(Kc, Lc)) / ((n - 1)**2)
    return hsic_val

def loss_independence(z_s1, z_s2, z_m1, z_m2):
    """
    Compute independence loss between shared and modality-specific representations.
    """
    #return hsic_rbf(z_s1, z_m1) + hsic_rbf(z_s2, z_m2)+ hsic_rbf(z_m1, z_m2)
    return hsic_rbf(z_s1, z_m1, unbiased=False) + hsic_rbf(z_s2, z_m2, unbiased=False)+ hsic_rbf(z_m1, z_m2, unbiased=False)


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