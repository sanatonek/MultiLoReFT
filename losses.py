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
    
    def __call__(self, losses, model, trainable_params):
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
            model.zero_grad()
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
        weighted_loss = sum(w * l for w, l in zip(weights, losses))
        
        return weighted_loss, weights

def loss_reconstruction(h, x, decoder):
    """Compute reconstruction loss between input and decoded representation."""
    return F.l1_loss(x, decoder(h))

def loss_mutual_info(h1, h2, z_components):
    """
    Maximize mutual information between original and projected representations.
    Supports mismatched dimensions by projecting to a common space.
    """
    # Concatenate modality-specific and shared components
    z1 = torch.cat([z_components[0][0], z_components[0][1]], dim=1)
    z2 = torch.cat([z_components[1][0], z_components[1][1]], dim=1)
    
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
    loss_ortho_1 = torch.norm(torch.mm(R_s, R_m1.T), p="fro")**2
    loss_ortho_2 = torch.norm(torch.mm(R_m1, R_m2.T), p="fro")**2
    loss_ortho_3 = torch.norm(torch.mm(R_s, R_m2.T), p="fro")**2
    return loss_ortho_1 + loss_ortho_2 + loss_ortho_3

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
def compute_stage_losses(model, h1, h2, z_components, stage):
    """
    Compute losses based on the current training stage.
    
    Args:
        model: The projection model
        h1, h2: Input representations
        z_components: Decomposed representations
        stage: Current training stage ("shared", "private", or "joint")
        
    Returns:
        losses_list: List of losses to optimize
        loss_names: Names of the losses
        all_losses: All computed losses
        all_loss_names: Names of all losses
    """
    # Compute all losses
    l_shared = loss_shared_consistency(z_components[0][1], z_components[1][1])
    l_orthogonal = loss_orthogonality(model.R_s, model.R_m1, model.R_m2)
    l_mi = loss_mutual_info(h1, h2, z_components)
    
    all_losses = [l_shared.item(), l_orthogonal.item(), l_mi.item()]
    all_loss_names = ["Shared Loss", "Orthogonal Loss", "Mutual Info Loss"]
    
    # Return appropriate losses based on stage
    if stage == "shared":
        return [l_shared, l_mi], ["Shared Loss", "Mutual Info Loss"], all_losses, all_loss_names
    elif stage == "private":
        return [l_orthogonal], ["Orthogonal Loss"], all_losses, all_loss_names
    elif stage == "joint":
        return [l_orthogonal, l_shared, l_mi], ["Orthogonal Loss", "Shared Loss", "Mutual Info Loss"], all_losses, all_loss_names
    else:
        raise ValueError(f"Unknown training stage: {stage}")