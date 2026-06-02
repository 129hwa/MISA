
import numpy as np


def get_misa_hparams(dataset, random_state=None):
    h = {}

    # ----- Architecture (unchanged) -----
    h['latent_dim']       = 512
    h['attention_heads']  = 4
    h['dropout_rate']     = 0.3

    # ----- CLIP backbone -----
    h['use_clip']         = True           # True=CLIP, False=ResNet
    h['clip_model_name']  = 'ViT-B/16'     # 'ViT-B/32', 'ViT-B/16', 'ViT-L/14'
    h['freeze_clip']      = True           # 동결 여부 (현재는 항상 True)
    h['lr_clip']          = 1e-5           # CLIP fine-tuning용 lr (참고)


    # ----- Original MISA loss weights -----
    h['lambda_task']        = 1.0
    h['lambda_inv_adv']     = 0.1
    h['lambda_spc_clf']     = 0.1
    h['lambda_disentangle'] = 0.05
    h['lambda_reconstruct'] = 0.05

    # ----- GRL schedule -----
    h['grl_alpha_max']     = 1.0
    h['grl_warmup_epochs'] = 1000

    # ----- Spectral loss (unchanged) -----
    h['use_spectral_loss']    = True
    h['lambda_spectral_inv']  = 0.1
    h['lambda_spectral_spc']  = 0.1
    h['gabor_num_filters']    = 8
    h['gabor_kernel_size']    = 11

    # ----- NEW: Supervised contrastive on z_inv -----
    h['use_supcon']          = True
    h['lambda_supcon']       = 0.3
    h['supcon_temperature']  = 0.07
    h['proj_dim']            = 256

    # ----- NEW: Swap reconstruction -----
    h['use_swap_recon']     = True
    h['lambda_swap_recon']  = 0.05
    h['swap_max_pairs']     = 64

    # ----- NEW: EMA teacher -----
    h['use_ema_teacher'] = True
    h['ema_decay']       = 0.995
    h['use_cosine_lr']    = True
    h['total_steps']      = 5000
    h['lr_min']           = 1e-6

    # ----- CLIP (unchanged) -----
    h['use_clip']         = False
    h['clip_model_name']  = 'ViT-B/32'
    h['freeze_clip']      = True
    h['lr_clip']          = 1e-5

    # ----- Per-dataset overrides -----
    if dataset == 'PACS':
        h['lambda_supcon']       = 0.5
        h['lambda_inv_adv']      = 0.1
        h['grl_warmup_epochs']   = 1000
    elif dataset == 'VLCS':
        h['lambda_supcon']       = 0.3
        h['lambda_inv_adv']      = 0.05          # VLCS labels are noisier
        h['lambda_disentangle']  = 0.02
        h['grl_warmup_epochs']   = 1500
    elif dataset == 'OfficeHome':
        h['lambda_supcon']       = 0.2
        h['lambda_inv_adv']      = 0.1
        h['grl_warmup_epochs']   = 1500
    elif dataset == 'DomainNet':
        h['lambda_supcon']       = 0.1           # few positives per batch
        h['lambda_inv_adv']      = 0.05
        h['lambda_spc_clf']      = 0.05
        h['grl_warmup_epochs']   = 3000
        h['swap_max_pairs']      = 32
        h['ema_decay']           = 0.999

    # ----- Random search (HP sweep mode) -----
    if random_state is not None:
        r = np.random.RandomState(random_state)

        h['latent_dim']        = int(2 ** r.uniform(8, 10))
        h['attention_heads']   = int(r.choice([2, 4, 8]))
        h['dropout_rate']      = float(r.uniform(0.1, 0.5))

        h['lambda_inv_adv']     = float(10 ** r.uniform(-2, -0.3))
        h['lambda_spc_clf']     = float(10 ** r.uniform(-2, -0.3))
        h['lambda_disentangle'] = float(10 ** r.uniform(-2.5, -0.7))
        h['lambda_reconstruct'] = float(10 ** r.uniform(-2.5, -0.7))

        h['grl_alpha_max']      = float(r.uniform(0.5, 1.5))
        h['grl_warmup_epochs']  = int(r.uniform(500, 2500))

        if h['use_spectral_loss']:
            h['lambda_spectral_inv'] = float(10 ** r.uniform(-2, -0.3))
            h['lambda_spectral_spc'] = float(10 ** r.uniform(-2, -0.3))
            h['gabor_num_filters']   = int(r.choice([4, 8, 16]))
            h['gabor_kernel_size']   = int(r.choice([7, 11, 15]))

        # NEW knobs sampled from tight-ish ranges
        h['lambda_supcon']       = float(10 ** r.uniform(-1.3, -0.2))   # ~0.05..0.63
        h['supcon_temperature']  = float(r.choice([0.07, 0.1, 0.2]))
        h['lambda_swap_recon']   = float(10 ** r.uniform(-2.3, -0.7))   # ~0.005..0.2
        h['ema_decay']           = float(r.choice([0.99, 0.995, 0.999]))

    return h
