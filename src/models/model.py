from src.models.vae_unet import VAEUNet

def get_model(cfg, device):
    """
    Instantiates and returns the VAE-U-Net model based on configuration.
    
    Args:
        cfg (dict): Configuration dictionary defining model hyperparameters.
        device (torch.device): Device to load the model onto.
        
    Returns:
        torch.nn.Module: The instantiated VAE-U-Net model.
    """
    # Check for latent_dim in config, else default to 256
    latent_dim = cfg['model'].get('latent_dim', 256)
    
    model = VAEUNet(
        spatial_dims=cfg['model']['spatial_dims'],
        in_channels=cfg['model']['in_channels'],
        out_channels=cfg['data']['num_classes'],
        channels=tuple(cfg['model']['channels']),
        strides=tuple(cfg['model']['strides']),
        num_res_units=cfg['model']['num_res_units'],
        latent_dim=latent_dim,
        norm="INSTANCE",
    ).to(device)
    return model