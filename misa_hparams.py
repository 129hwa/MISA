# misa_hparams.py

def get_misa_hparams(dataset, random_state=None):
    """Get MISA hyperparameters for a given dataset"""
    
    hparams = {}
    
    # Architecture hyperparameters
    hparams['latent_dim'] = 512
    hparams['attention_heads'] = 4
    hparams['dropout_rate'] = 0.3
    
    # Loss weights
    hparams['lambda_task'] = 1.0
    hparams['lambda_inv_adv'] = 0.1
    hparams['lambda_spc_clf'] = 0.1
    hparams['lambda_disentangle'] = 0.05
    hparams['lambda_reconstruct'] = 0.05
    
    # GRL parameters
    hparams['grl_alpha_max'] = 1.0
    hparams['grl_warmup_epochs'] = 1000
    
    # Spectral loss parameters
    hparams['use_spectral_loss'] = True
    hparams['lambda_spectral_inv'] = 0.1
    hparams['lambda_spectral_spc'] = 0.1
    hparams['gabor_num_filters'] = 8
    hparams['gabor_kernel_size'] = 11
    
    # CLIP parameters (optional)
    hparams['use_clip'] = False
    hparams['clip_model_name'] = 'ViT-B/32'
    hparams['freeze_clip'] = True
    hparams['lr_clip'] = 1e-5
    
    # Random hyperparameter search
    if random_state is not None:
        import numpy as np
        r = np.random.RandomState(random_state)
        
        hparams['latent_dim'] = int(2 ** r.uniform(7, 10))
        hparams['attention_heads'] = int(r.choice([2, 4, 8]))
        hparams['dropout_rate'] = r.uniform(0.1, 0.5)
        hparams['lambda_inv_adv'] = 10 ** r.uniform(-2, 0)
        hparams['lambda_spc_clf'] = 10 ** r.uniform(-2, 0)
        hparams['lambda_disentangle'] = 10 ** r.uniform(-3, -1)
        hparams['lambda_reconstruct'] = 10 ** r.uniform(-3, -1)
        hparams['grl_alpha_max'] = r.uniform(0.5, 2.0)
        hparams['grl_warmup_epochs'] = int(r.uniform(500, 2000))
        
        if hparams['use_spectral_loss']:
            hparams['lambda_spectral_inv'] = 10 ** r.uniform(-2, 0)
            hparams['lambda_spectral_spc'] = 10 ** r.uniform(-2, 0)
            hparams['gabor_num_filters'] = int(r.choice([4, 8, 16]))
            hparams['gabor_kernel_size'] = int(r.choice([7, 11, 15]))
    
    return hparams
