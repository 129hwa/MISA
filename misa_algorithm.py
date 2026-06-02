
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from domainbed import networks
from domainbed.algorithms import Algorithm


# =====================================================================
# Gradient Reversal (original MISA, unchanged)
# =====================================================================
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x, alpha=1.0):
    return GradReverse.apply(x, alpha)


# =====================================================================
# Domain Information Encoder (original MISA, unchanged)
# =====================================================================
class DomainInformationEncoder(nn.Module):
    def __init__(self, feature_dim, output_dim, dropout_rate=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.BatchNorm1d(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, output_dim),
        )

    def forward(self, z_features):
        return self.encoder(z_features)


# =====================================================================
# Feature Attention (original MISA, unchanged)
# =====================================================================
class FeatureAttention(nn.Module):
    def __init__(self, feature_dim, context_dim, num_heads, output_dim,
                 dropout_rate=0.3):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            kdim=context_dim,
            vdim=context_dim,
            batch_first=True,
            dropout=dropout_rate,
        )
        self.norm1 = nn.LayerNorm(feature_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim * 2, output_dim),
        )
        self.norm2 = nn.LayerNorm(output_dim)

    def forward(self, z_query_features, context_kv_features):
        q = z_query_features.unsqueeze(1)
        kv = context_kv_features.unsqueeze(1)
        attn_output, _ = self.attention(q, kv, kv)
        attn_output = attn_output.squeeze(1)
        z_norm = self.norm1(z_query_features + attn_output)
        mlp_out = self.mlp(z_norm)
        return self.norm2(mlp_out)


# =====================================================================
# Learnable Gabor bank + spectral loss (original MISA, unchanged)
# =====================================================================
class LearnableGaborBank(nn.Module):
    def __init__(self, num_filters=8, kernel_size=11):
        super().__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.sigma = nn.Parameter(torch.full((num_filters,), 4.0))
        init_theta = torch.linspace(0, math.pi * (1 - 1 / num_filters),
                                    steps=num_filters)
        self.theta = nn.Parameter(init_theta)
        self.lambd = nn.Parameter(torch.full((num_filters,), 10.0))
        self.gamma = nn.Parameter(torch.full((num_filters,), 0.5))
        self.psi = nn.Parameter(torch.zeros(num_filters))

    def forward(self, device):
        kernels = []
        half = self.kernel_size // 2
        y, x = torch.meshgrid(
            torch.arange(-half, half + 1),
            torch.arange(-half, half + 1),
            indexing='ij',
        )
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
            exp_term = torch.exp(
                -0.5 * (x_theta ** 2 + (gamma ** 2) * (y_theta ** 2)) / (sigma ** 2)
            )
            cos_term = torch.cos(2 * math.pi * x_theta / lambd + psi)
            kernel = exp_term * cos_term
            kernels.append(kernel.unsqueeze(0).unsqueeze(0))
        return torch.cat(kernels, dim=0)


def features_to_image(features, height=None, width=None):
    batch_size, feature_dim = features.shape
    if height is None or width is None:
        side = int(math.sqrt(feature_dim))
        if side * side < feature_dim:
            side += 1
        height = width = side
    padded_size = height * width
    if padded_size > feature_dim:
        pad = torch.zeros(batch_size, padded_size - feature_dim,
                          device=features.device)
        features = torch.cat([features, pad], dim=1)
    else:
        features = features[:, :padded_size]
    return features.view(batch_size, 1, height, width)


def spectral_loss_gabor_edge_features(features, domain_labels, learnable_gabor,
                                       epsilon=1e-8):
    img_size = int(math.ceil(math.sqrt(features.shape[1])))
    x_gray = features_to_image(features, img_size, img_size)

    gabor_bank = learnable_gabor(features.device)
    pad = learnable_gabor.kernel_size // 2
    gabor_response = torch.abs(F.conv2d(x_gray, gabor_bank, padding=pad))

    sobel_x = torch.tensor(
        [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
        device=features.device,
    ).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor(
        [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
        device=features.device,
    ).unsqueeze(0).unsqueeze(0)
    edge_x = F.conv2d(x_gray, sobel_x, padding=1)
    edge_y = F.conv2d(x_gray, sobel_y, padding=1)
    edge_map = torch.sqrt(edge_x ** 2 + edge_y ** 2 + epsilon)

    combined = torch.cat([gabor_response, edge_map], dim=1)
    flat = combined.mean(dim=[2, 3])
    feat_mean = flat.mean(dim=1, keepdim=True)
    feat_std = flat.std(dim=1, keepdim=True) + epsilon
    normed = (flat - feat_mean) / feat_std

    uniq = torch.unique(domain_labels)
    means = []
    for d in uniq:
        mask = (domain_labels == d)
        if mask.sum() == 0:
            continue
        means.append(normed[mask].mean(dim=0))
    if len(means) < 2:
        return 0.0 * features.sum()
    loss = 0.0
    count = 0
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            diff = means[i] - means[j]
            loss = loss + torch.sum(diff ** 2) / normed.shape[1]
            count += 1
    return loss / count if count > 0 else 0.0 * features.sum()


def invert_spectral_loss(loss_value):
    return 1.0 / (1.0 + 10.0 * loss_value)


# =====================================================================
# MISA v2
# =====================================================================
class MISA(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)

        self.num_classes = num_classes
        self.num_domains = num_domains

        # --- Backbone + latent shape ---
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.feature_dim = self.featurizer.n_outputs
        self.latent_dim = hparams['latent_dim']
        self.num_heads = hparams['attention_heads']
        self.dropout_rate = hparams['dropout_rate']

        # --- DIE / IFA / SFA ---
        self.die = DomainInformationEncoder(
            self.feature_dim, self.feature_dim // 4,
            dropout_rate=self.dropout_rate,
        )
        self.ifa = FeatureAttention(
            self.feature_dim, self.feature_dim // 4,
            self.num_heads, self.latent_dim,
            dropout_rate=self.dropout_rate,
        )
        self.sfa = FeatureAttention(
            self.feature_dim, self.feature_dim // 4,
            self.num_heads, self.latent_dim,
            dropout_rate=self.dropout_rate,
        )

        # --- Main classifier ---
        self.main_classifier = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim // 2),
            nn.BatchNorm1d(self.latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.latent_dim // 2, num_classes),
        )

        # --- MARGINAL adversary on z_inv (Theorem-consistent) ---
        self.domain_classifier_inv = nn.Sequential(
            spectral_norm(nn.Linear(self.latent_dim, self.latent_dim // 2)),
            nn.BatchNorm1d(self.latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            spectral_norm(nn.Linear(self.latent_dim // 2, num_domains)),
        )

        # --- z_spc domain classifier ---
        self.domain_classifier_spc = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim // 2),
            nn.BatchNorm1d(self.latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.latent_dim // 2, num_domains),
        )

        # --- Reconstructor (reused by swap-recon auxiliary) ---
        self.reconstructor = nn.Sequential(
            nn.Linear(self.latent_dim * 2, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, self.feature_dim),
        )

        # --- Auxiliary: SupCon projection head (v3: 3-layer, on featurizer feature) ---
        proj_dim = hparams.get('proj_dim', 256)
        proj_hidden = self.feature_dim
        self.proj_head = nn.Sequential(
            nn.Linear(self.feature_dim, proj_hidden),
            nn.BatchNorm1d(proj_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, proj_hidden),
            nn.BatchNorm1d(proj_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, proj_dim),
        )

        # --- Spectral loss components ---
        if hparams['use_spectral_loss']:
            self.learnable_gabor = LearnableGaborBank(
                num_filters=hparams['gabor_num_filters'],
                kernel_size=hparams['gabor_kernel_size'],
            )

        # --- Optimizer ---
        if hparams.get('use_clip', False) and \
        not hparams.get('freeze_clip', True):
            # CLIP fine-tuning 시 (현재는 안 씀): 다른 lr 그룹
            clip_params, other_params = [], []
            for name, param in self.named_parameters():
                if 'featurizer.clip_model' in name or \
                'featurizer.visual_encoder' in name:
                    clip_params.append(param)
                else:
                    other_params.append(param)
            self.optimizer = torch.optim.Adam([
                {'params': clip_params,
                'lr': hparams.get('lr_clip', 1e-5)},
                {'params': other_params,
                'lr': hparams['lr']},
            ], weight_decay=hparams['weight_decay'])
        else:
            # 기본 (ResNet 또는 frozen CLIP)
            self.optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams['lr'],
                weight_decay=self.hparams['weight_decay'],
            )


        # --- v4: cosine LR decay scheduler (theorem-orthogonal) ---
        self.use_cosine_lr = hparams.get('use_cosine_lr', False)
        if self.use_cosine_lr:
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=int(hparams.get('total_steps', 5000)),
                eta_min=float(hparams.get('lr_min', 1e-6)),
            )
        else:
            self.lr_scheduler = None

        # --- EMA teacher (inference only) ---
        self.use_ema = hparams.get('use_ema_teacher', True)
        self.ema_decay = hparams.get('ema_decay', 0.995)
        if self.use_ema:
            self.ema_die = copy.deepcopy(self.die)
            self.ema_ifa = copy.deepcopy(self.ifa)
            self.ema_main_classifier = copy.deepcopy(self.main_classifier)
            for m in (self.ema_die, self.ema_ifa, self.ema_main_classifier):
                for p in m.parameters():
                    p.requires_grad_(False)

        self.register_buffer('update_count', torch.tensor([0]))

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _ema_update(self):
        if not self.use_ema:
            return
        d = self.ema_decay
        for ema_m, m in (
            (self.ema_die, self.die),
            (self.ema_ifa, self.ifa),
            (self.ema_main_classifier, self.main_classifier),
        ):
            for ep, p in zip(ema_m.parameters(), m.parameters()):
                ep.data.mul_(d).add_(p.data, alpha=1.0 - d)
            for eb, b in zip(ema_m.buffers(), m.buffers()):
                eb.data.copy_(b.data)

    def _supcon_loss(self, feats, labels, temperature):
        """Auxiliary supervised contrastive loss (Khosla et al. 2020)."""
        device = feats.device
        B = feats.shape[0]
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        self_mask = torch.eye(B, device=device)
        mask = (mask - self_mask).clamp(min=0.0)

        logits = feats @ feats.T / temperature
        logits_max, _ = logits.max(dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        neg_mask = 1.0 - self_mask
        exp_logits = torch.exp(logits) * neg_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        pos_count = mask.sum(dim=1)
        valid = pos_count > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1)[valid] / pos_count[valid]
        return -mean_log_prob_pos.mean()

    @staticmethod
    def _find_swap_pairs(y, d, max_pairs=None):
        """For each anchor i, find j with y_i == y_j and d_i != d_j."""
        device = y.device
        B = y.shape[0]
        if B < 2:
            return None
        same_y = y.unsqueeze(0) == y.unsqueeze(1)
        diff_d = d.unsqueeze(0) != d.unsqueeze(1)
        valid = same_y & diff_d
        i_idx, j_idx = [], []
        for i in range(B):
            cand = torch.nonzero(valid[i], as_tuple=False).view(-1)
            if cand.numel() == 0:
                continue
            pick = cand[torch.randint(0, cand.numel(), (1,), device=device)]
            i_idx.append(i)
            j_idx.append(int(pick.item()))
        if not i_idx:
            return None
        i_idx = torch.tensor(i_idx, device=device, dtype=torch.long)
        j_idx = torch.tensor(j_idx, device=device, dtype=torch.long)
        if max_pairs is not None and i_idx.numel() > max_pairs:
            perm = torch.randperm(i_idx.numel(), device=device)[:max_pairs]
            i_idx, j_idx = i_idx[perm], j_idx[perm]
        return i_idx, j_idx

    # ------------------------------------------------------------------
    def forward(self, x, grl_alpha=2.0):
        z_intermediate = self.featurizer(x)
        if z_intermediate.dtype == torch.float16:
            z_intermediate = z_intermediate.float()

        domain_context = self.die(z_intermediate)
        z_inv = self.ifa(z_intermediate, domain_context)
        z_spc = self.sfa(z_intermediate, domain_context)

        task_logits = self.main_classifier(z_inv)

        # MARGINAL adversary on z_inv — matches Theorem's MI route I(Z_inv; D)
        z_inv_reversed = grad_reverse(z_inv, grl_alpha)
        domain_logits_inv = self.domain_classifier_inv(z_inv_reversed)

        domain_logits_spc = self.domain_classifier_spc(z_spc)

        # Self-reconstruction: Assumption 5's eps_rec^(S)
        z_combined = torch.cat([z_inv, z_spc], dim=1)
        z_reconstructed = self.reconstructor(z_combined)

        # Auxiliary: SupCon projection (v3: on backbone feature, not z_inv)
        z_inv_proj = F.normalize(self.proj_head(z_intermediate), p=2, dim=1)

        return {
            'task_logits': task_logits,
            'domain_logits_inv': domain_logits_inv,
            'domain_logits_spc': domain_logits_spc,
            'z_inv': z_inv,
            'z_spc': z_spc,
            'z_intermediate': z_intermediate,
            'z_reconstructed': z_reconstructed,
            'z_inv_proj': z_inv_proj,
        }

    def update(self, minibatches, unlabeled=None):
        if self.hparams['grl_warmup_epochs'] > 0:
            progress = min(1.0, self.update_count.item()
                           / self.hparams['grl_warmup_epochs'])
            grl_alpha = (2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0) \
                        * self.hparams['grl_alpha_max']
        else:
            grl_alpha = self.hparams['grl_alpha_max']

        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_d = torch.cat([
            torch.full((x.shape[0],), i, dtype=torch.int64, device=all_x.device)
            for i, (x, y) in enumerate(minibatches)
        ])

        outputs = self.forward(all_x, grl_alpha)

        # === Theorem-bound losses ===
        loss_task    = F.cross_entropy(outputs['task_logits'], all_y)
        loss_inv_adv = F.cross_entropy(outputs['domain_logits_inv'], all_d)
        loss_spc_clf = F.cross_entropy(outputs['domain_logits_spc'], all_d)

        z_inv_n = F.normalize(outputs['z_inv'], p=2, dim=1)
        z_spc_n = F.normalize(outputs['z_spc'], p=2, dim=1)
        loss_disentangle = torch.mean(torch.sum(z_inv_n * z_spc_n, dim=1) ** 2)

        loss_reconstruct = F.mse_loss(outputs['z_reconstructed'],
                                      outputs['z_intermediate'])

        loss_spectral_inv = torch.tensor(0.0, device=all_x.device)
        loss_spectral_spc = torch.tensor(0.0, device=all_x.device)
        if self.hparams['use_spectral_loss']:
            loss_spectral_inv = spectral_loss_gabor_edge_features(
                outputs['z_inv'], all_d, self.learnable_gabor)
            spectral_spc_direct = spectral_loss_gabor_edge_features(
                outputs['z_spc'], all_d, self.learnable_gabor)
            loss_spectral_spc = invert_spectral_loss(spectral_spc_direct)

        # === Auxiliary losses (outside bound) ===
        loss_supcon = torch.tensor(0.0, device=all_x.device)
        if self.hparams.get('use_supcon', True):
            loss_supcon = self._supcon_loss(
                outputs['z_inv_proj'], all_y,
                temperature=self.hparams.get('supcon_temperature', 0.1),
            )

        loss_swap_recon = torch.tensor(0.0, device=all_x.device)
        if self.hparams.get('use_swap_recon', True):
            pairs = self._find_swap_pairs(
                all_y, all_d,
                max_pairs=self.hparams.get('swap_max_pairs', 64),
            )
            if pairs is not None:
                i_idx, j_idx = pairs
                z_swap = torch.cat(
                    [outputs['z_inv'][i_idx], outputs['z_spc'][j_idx]], dim=1
                )
                recon_swap = self.reconstructor(z_swap)
                loss_swap_recon = F.mse_loss(
                    recon_swap, outputs['z_intermediate'][j_idx].detach()
                )

        # === Total ===
        total_loss = (
            self.hparams['lambda_task']          * loss_task
            + self.hparams['lambda_inv_adv']     * loss_inv_adv
            + self.hparams['lambda_spc_clf']     * loss_spc_clf
            + self.hparams['lambda_disentangle'] * loss_disentangle
            + self.hparams['lambda_reconstruct'] * loss_reconstruct
        )
        if self.hparams['use_spectral_loss']:
            total_loss = total_loss + (
                self.hparams['lambda_spectral_inv'] * loss_spectral_inv
                + self.hparams['lambda_spectral_spc'] * loss_spectral_spc
            )
        if self.hparams.get('use_supcon', True):
            total_loss = total_loss + self.hparams['lambda_supcon'] * loss_supcon
        if self.hparams.get('use_swap_recon', True):
            total_loss = total_loss + self.hparams['lambda_swap_recon'] * loss_swap_recon

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self._ema_update()
        self.update_count += 1

        losses = {
            'loss': total_loss.item(),
            'task_loss': loss_task.item(),
            'inv_adv_loss': loss_inv_adv.item(),
            'spc_clf_loss': loss_spc_clf.item(),
            'disentangle_loss': loss_disentangle.item(),
            'reconstruct_loss': loss_reconstruct.item(),
        }
        if self.hparams['use_spectral_loss']:
            losses['spectral_inv_loss'] = loss_spectral_inv.item()
            losses['spectral_spc_loss'] = loss_spectral_spc.item()
        if self.hparams.get('use_supcon', True):
            losses['supcon_loss'] = float(loss_supcon.item())
        if self.hparams.get('use_swap_recon', True):
            losses['swap_recon_loss'] = float(loss_swap_recon.item())
        return losses

    def predict(self, x):
        """Inference via EMA teacher (still in the hypothesis class H)."""
        if self.use_ema:
            with torch.no_grad():
                z = self.featurizer(x)
                if z.dtype == torch.float16:
                    z = z.float()
                ctx = self.ema_die(z)
                z_inv = self.ema_ifa(z, ctx)
                return self.ema_main_classifier(z_inv)
        return self.forward(x, grl_alpha=0.0)['task_logits']

    def train(self, mode=True):
        super().train(mode)
        self.featurizer.train(mode)
        if self.use_ema:
            self.ema_die.eval()
            self.ema_ifa.eval()
            self.ema_main_classifier.eval()
