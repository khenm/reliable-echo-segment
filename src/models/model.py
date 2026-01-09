from monai.networks.nets import UNet

def get_model(cfg, device):
    """
    Instantiates and returns the U-Net model based on configuration.
    
    Args:
        cfg (dict): Configuration dictionary defining model hyperparameters.
        device (torch.device): Device to load the model onto.
        
    Returns:
        torch.nn.Module: The instantiated U-Net model.
    """
    model = UNet(
        spatial_dims=cfg['model']['spatial_dims'],
        in_channels=cfg['model']['in_channels'],
        out_channels=cfg['data']['num_classes'],
        channels=tuple(cfg['model']['channels']),
        strides=tuple(cfg['model']['strides']),
        num_res_units=cfg['model']['num_res_units'],
        norm="INSTANCE",
    ).to(device)
    return model