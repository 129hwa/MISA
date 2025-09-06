# misa_components.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# ============ Gradient Reversal Layer ============
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return GradientReversal.apply(x, alpha)

# ============ Domain Information Encoder ============
class DomainInformationEncoder(nn.Module):
    def __init__(self, feature_dim, output_dim, dropout_rate=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.BatchNorm1d(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, output_dim)
        )
    
    def forward(self, features):
        return self.encoder(features)

# ============ Feature Attention Module ============
class FeatureAttention(nn.Module):
    def __init__(self, feature_dim, context_dim, num_heads, output_dim, dropout_rate=0.3):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            kdim=context_dim,
            vdim=context_dim,
            batch_first=True,
            dropout=dropout_rate
        )
        self.norm1 = nn.LayerNorm(feature_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim * 2, output_dim)
        )
        self.norm2 = nn.LayerNorm(output_dim)
    
    def forward(self, z_query_features, context_kv_features):
        z_q_unsqueeze = z_query_features.unsqueeze(1)
        context_kv_unsqueeze = context_kv_features.unsqueeze(1)
        attn_output, _ = self.attention(z_q_unsqueeze, context_kv_unsqueeze, context_kv_unsqueeze)
        attn_output = attn_output.squeeze(1)
        z_after_attn_norm = self.norm1(z_query_features + attn_output)
        mlp_output = self.mlp(z_after_attn_norm)
        return self.norm2(mlp_output)

# ============ Learnable Gabor Bank ============
class LearnableGaborBank(nn.Module):
    def __init__(self, num_filters=8, kernel_size=11):
        super().__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.sigma = nn.Parameter(torch.full((num_filters,), 4.0))
        init_theta = torch.linspace(0, math.pi * (1 - 1/num_filters), steps=num_filters)
        self.theta = nn.Parameter(init_theta)
        self.lambd = nn.Parameter(torch.full((num_filters,), 10.0))
        self.gamma = nn.Parameter(torch.full((num_filters,), 0.5))
        self.psi = nn.Parameter(torch.zeros(num_filters))
    
    def forward(self, device):
        kernels = []
        half_size = self.kernel_size // 2
        y, x = torch.meshgrid(torch.arange(-half_size, half_size+1),
                              torch.arange(-half_size, half_size+1), indexing='ij')
        x = x.float().to(device)
        y = y.float().to(device)
        
        for i in range(self.num_filters):
            sigma = self.sigma[i]
            theta = self.theta[i]
            lambd = self.lambd[i]
            gamma = self.gamma[i]
            psi = self.psi[i]
            
            x_theta = x * torch.cos(theta) + y * torch.sin(theta)
            y_theta = -x * torch.sin(theta) + y * torch.cos(theta)
            exp_term = torch.exp(-0.5 * (x_theta**2 + (gamma**2) * (y_theta**2)) / (sigma**2))
            cos_term = torch.cos(2 * math.pi * x_theta / lambd + psi)
            kernel = exp_term * cos_term
            kernels.append(kernel.unsqueeze(0).unsqueeze(0))
        
        gabor_bank = torch.cat(kernels, dim=0)
        return gabor_bank

# ============ Spectral Loss Functions ============
def features_to_image(features, height=None, width=None):
    batch_size, feature_dim = features.shape
    
    if height is None or width is None:
        side_length = int(math.sqrt(feature_dim))
        if side_length * side_length < feature_dim:
            side_length += 1
        height = width = side_length
    
    padded_size = height * width
    if padded_size > feature_dim:
        padding = torch.zeros(batch_size, padded_size - feature_dim, device=features.device)
        features_padded = torch.cat([features, padding], dim=1)
    else:
        features_padded = features[:, :padded_size]
    
    return features_padded.view(batch_size, 1, height, width)

def spectral_loss_gabor_edge_features(features, domain_labels, learnable_gabor, epsilon=1e-8):
    img_size = int(math.ceil(math.sqrt(features.shape[1])))
    x_gray = features_to_image(features, img_size, img_size)
    
    # Gabor response
    gabor_bank = learnable_gabor(features.device)
    pad = learnable_gabor.kernel_size // 2
    gabor_response = F.conv2d(x_gray, gabor_bank, padding=pad)
    gabor_response = torch.abs(gabor_response)
    
    # Edge detection
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], 
                          device=features.device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], 
                          device=features.device).unsqueeze(0).unsqueeze(0)
    edge_x = F.conv2d(x_gray, sobel_x, padding=1)
    edge_y = F.conv2d(x_gray, sobel_y, padding=1)
    edge_map = torch.sqrt(edge_x**2 + edge_y**2 + epsilon)
    
    # Combine features
    features_combined = torch.cat([gabor_response, edge_map], dim=1)
    features_flat = features_combined.mean(dim=[2,3])
    
    # Normalize
    feat_mean = features_flat.mean(dim=1, keepdim=True)
    feat_std = features_flat.std(dim=1, keepdim=True) + epsilon
    features_norm = (features_flat - feat_mean) / feat_std
    
    # Calculate domain differences
    unique_domains = torch.unique(domain_labels)
    domain_means = []
    
    for d in unique_domains:
        mask = (domain_labels == d)
        if mask.sum() == 0:
            continue
        domain_mean = features_norm[mask].mean(dim=0)
        domain_means.append(domain_mean)
    
    if len(domain_means) < 2:
        return 0.0 * features.sum()
    
    # Calculate loss
    loss = 0.0
    count = 0
    for i in range(len(domain_means)):
        for j in range(i+1, len(domain_means)):
            diff = domain_means[i] - domain_means[j]
            loss += torch.sum(diff**2) / features_norm.shape[1]
            count += 1
    
    return loss / count if count > 0 else 0.0 * features.sum()

def invert_spectral_loss(loss_value):
    return 1.0 / (1.0 + 10.0 * loss_value)
