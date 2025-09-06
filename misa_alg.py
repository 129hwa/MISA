# misa_algorithm.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import spectral_norm

from domainbed.algorithms import Algorithm
from domainbed import networks
from misa_components import (
    grad_reverse,
    DomainInformationEncoder,
    FeatureAttention,
    LearnableGaborBank,
    spectral_loss_gabor_edge_features,
    invert_spectral_loss
)

class MISA(Algorithm):
    """
    Mutual Information Spectral Attention for Domain Generalization
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        
        # Feature extractor
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.feature_dim = self.featurizer.n_outputs
        self.latent_dim = hparams['latent_dim']
        self.num_heads = hparams['attention_heads']
        self.dropout_rate = hparams['dropout_rate']
        
        # MISA components
        self.die = DomainInformationEncoder(
            self.feature_dim, 
            self.feature_dim // 4, 
            dropout_rate=self.dropout_rate
        )
        
        self.ifa = FeatureAttention(
            self.feature_dim, 
            self.feature_dim // 4, 
            self.num_heads, 
            self.latent_dim, 
            dropout_rate=self.dropout_rate
        )
        
        self.sfa = FeatureAttention(
            self.feature_dim, 
            self.feature_dim // 4, 
            self.num_heads, 
            self.latent_dim, 
            dropout_rate=self.dropout_rate
        )
        
        # Classifiers
        self.main_classifier = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim // 2),
            nn.BatchNorm1d(self.latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.latent_dim // 2, num_classes)
        )
        
        self.domain_classifier_inv = nn.Sequential(
            spectral_norm(nn.Linear(self.latent_dim, self.latent_dim // 2)),
            nn.BatchNorm1d(self.latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            spectral_norm(nn.Linear(self.latent_dim // 2, num_domains))
        )
        
        self.domain_classifier_spc = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim // 2),
            nn.BatchNorm1d(self.latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.latent_dim // 2, num_domains)
        )
        
        # Reconstructor
        self.reconstructor = nn.Sequential(
            nn.Linear(self.latent_dim * 2, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, self.feature_dim)
        )
        
        # Spectral loss components
        if hparams['use_spectral_loss']:
            self.learnable_gabor = LearnableGaborBank(
                num_filters=hparams['gabor_num_filters'],
                kernel_size=hparams['gabor_kernel_size']
            )
        
        # Optimizer setup
        self._setup_optimizer(hparams)
        
        # Register update counter
        self.register_buffer('update_count', torch.tensor([0]))
    
    def _setup_optimizer(self, hparams):
        """Setup optimizer with optional CLIP-specific learning rates"""
        if hparams.get('use_clip', False) and not hparams.get('freeze_clip', True):
            clip_params = []
            other_params = []
            
            for name, param in self.named_parameters():
                if 'featurizer.visual_encoder' in name:
                    clip_params.append(param)
                else:
                    other_params.append(param)
            
            self.optimizer = torch.optim.Adam([
                {'params': clip_params, 'lr': hparams.get('lr_clip', 1e-5)},
                {'params': other_params, 'lr': hparams["lr"]}
            ], weight_decay=hparams['weight_decay'])
        else:
            self.optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )
    
    def forward(self, x, grl_alpha=2.0):
        # Extract features
        z_intermediate = self.featurizer(x)
        
        # Convert to float32 if needed
        if z_intermediate.dtype == torch.float16:
            z_intermediate = z_intermediate.float()
        
        # Generate domain context
        domain_context = self.die(z_intermediate)
        
        # Generate invariant and specific features
        z_inv = self.ifa(z_intermediate, domain_context)
        z_spc = self.sfa(z_intermediate, domain_context)
        
        # Task prediction
        task_logits = self.main_classifier(z_inv)
        
        # Domain predictions
        z_inv_reversed = grad_reverse(z_inv, grl_alpha)
        domain_logits_inv = self.domain_classifier_inv(z_inv_reversed)
        domain_logits_spc = self.domain_classifier_spc(z_spc)
        
        # Reconstruction
        z_combined = torch.cat([z_inv, z_spc], dim=1)
        z_reconstructed = self.reconstructor(z_combined)
        
        return {
            'task_logits': task_logits,
            'domain_logits_inv': domain_logits_inv,
            'domain_logits_spc': domain_logits_spc,
            'z_inv': z_inv,
            'z_spc': z_spc,
            'z_intermediate': z_intermediate,
            'z_reconstructed': z_reconstructed
        }
    
    def update(self, minibatches, unlabeled=None):
        # Calculate adaptive GRL alpha
        if self.hparams['grl_warmup_epochs'] > 0:
            progress = min(1.0, self.update_count.item() / self.hparams['grl_warmup_epochs'])
            grl_alpha = (2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0) * self.hparams['grl_alpha_max']
        else:
            grl_alpha = self.hparams['grl_alpha_max']
        
        # Prepare data
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_d = torch.cat([
            torch.full((x.shape[0],), i, dtype=torch.int64, device=all_x.device)
            for i, (x, y) in enumerate(minibatches)
        ])
        
        # Forward pass
        outputs = self.forward(all_x, grl_alpha)
        
        # Calculate losses
        losses = self._calculate_losses(outputs, all_y, all_d)
        
        # Total loss
        total_loss = self._combine_losses(losses)
        
        # Optimization
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.update_count += 1
        
        # Prepare return dict
        return_dict = {'loss': total_loss.item()}
        for key, value in losses.items():
            return_dict[key] = value.item()
        
        return return_dict
    
    def _calculate_losses(self, outputs, all_y, all_d):
        """Calculate all loss components"""
        losses = {}
        
        # Task loss
        losses['task_loss'] = F.cross_entropy(outputs['task_logits'], all_y)
        
        # Domain adversarial loss for z_inv
        losses['inv_adv_loss'] = F.cross_entropy(outputs['domain_logits_inv'], all_d)
        
        # Domain classification loss for z_spc
        losses['spc_clf_loss'] = F.cross_entropy(outputs['domain_logits_spc'], all_d)
        
        # Disentanglement loss
        z_inv_norm = F.normalize(outputs['z_inv'], p=2, dim=1)
        z_spc_norm = F.normalize(outputs['z_spc'], p=2, dim=1)
        losses['disentangle_loss'] = torch.mean(torch.sum(z_inv_norm * z_spc_norm, dim=1)**2)
        
        # Reconstruction loss
        losses['reconstruct_loss'] = F.mse_loss(outputs['z_reconstructed'], outputs['z_intermediate'])
        
        # Spectral losses (if enabled)
        if self.hparams['use_spectral_loss']:
            losses['spectral_inv_loss'] = spectral_loss_gabor_edge_features(
                outputs['z_inv'], all_d, self.learnable_gabor
            )
            spectral_spc_direct = spectral_loss_gabor_edge_features(
                outputs['z_spc'], all_d, self.learnable_gabor
            )
            losses['spectral_spc_loss'] = invert_spectral_loss(spectral_spc_direct)
        
        return losses
    
    def _combine_losses(self, losses):
        """Combine losses with weights"""
        total_loss = (
            self.hparams['lambda_task'] * losses['task_loss'] +
            self.hparams['lambda_inv_adv'] * losses['inv_adv_loss'] +
            self.hparams['lambda_spc_clf'] * losses['spc_clf_loss'] +
            self.hparams['lambda_disentangle'] * losses['disentangle_loss'] +
            self.hparams['lambda_reconstruct'] * losses['reconstruct_loss']
        )
        
        if self.hparams['use_spectral_loss']:
            total_loss += (
                self.hparams['lambda_spectral_inv'] * losses['spectral_inv_loss'] +
                self.hparams['lambda_spectral_spc'] * losses['spectral_spc_loss']
            )
        
        return total_loss
    
    def predict(self, x):
        outputs = self.forward(x, grl_alpha=0.0)
        return outputs['task_logits']
