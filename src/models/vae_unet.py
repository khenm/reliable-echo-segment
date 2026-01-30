import torch
import torch.nn as nn
from monai.networks.blocks import Convolution, UpSample
from src.registry import register_model

@register_model("VAEUNet")
class VAEUNet(nn.Module):
    """
    VAE-U-Net implementation for segmentation with variational regularization.
    Combines a standard U-Net encoder-decoder structure with a Variational Autoencoder (VAE) bottleneck.
    """

    def __init__(
        self,
        spatial_dims=2,
        in_channels=1,
        out_channels=4,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        latent_dim=256,
        act="PRELU",
        norm="INSTANCE",
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        
        self.blocks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        current_c = in_channels
        
        # Build Encoder
        for i, stride in enumerate(strides):
            out_c = channels[i]
            # Downsampling block
            down_layer = nn.Sequential(
                Convolution(
                    spatial_dims,
                    current_c,
                    out_c,
                    strides=stride,
                    kernel_size=3,
                    act=act,
                    norm=norm,
                ),
                *[
                    Convolution(
                        spatial_dims,
                        out_c,
                        out_c,
                        strides=1,
                        kernel_size=3,
                        act=act,
                        norm=norm,
                    )
                    for _ in range(num_res_units)
                ]
            )
            self.downs.append(down_layer)
            current_c = out_c
            
        # Bottleneck
        self.bottleneck_conv = nn.Sequential(
            Convolution(
                spatial_dims,
                current_c,
                channels[-1],
                strides=1,
                kernel_size=3,
                act=act,
                norm=norm,
            ),
             *[
                Convolution(
                    spatial_dims,
                    channels[-1],
                    channels[-1],
                    strides=1,
                    kernel_size=3,
                    act=act,
                    norm=norm,
                )
                for _ in range(num_res_units)
            ]
        )
        
        enc_out_dim = channels[-1]
        self.latent_dim = latent_dim
        
        # VAE Components
        self.global_avg = nn.AdaptiveAvgPool2d(1)
        self.mu = nn.Linear(enc_out_dim, latent_dim)
        self.logvar = nn.Linear(enc_out_dim, latent_dim)
        
        self.decode_z = nn.Linear(latent_dim, enc_out_dim)
        
        # Decoder
        current_c = channels[-1]
        
        up_channels = [channels[2], channels[1], channels[0], channels[0]//2]
        up_strides = strides[::-1]
        
        for i, (out_c, stride) in enumerate(zip(up_channels, up_strides)):
            self.ups.append(
                UpSample(
                    spatial_dims,
                    current_c,
                    out_c,
                    scale_factor=stride,
                    mode="deconv", 
                    kernel_size=3,
                )
            )
            
            is_last = (i == len(up_strides) - 1)
            in_channels_conv = out_c * 2 if not is_last else out_c
            
            self.blocks.append(
                 nn.Sequential(
                    Convolution(
                        spatial_dims,
                        in_channels_conv,
                        out_c,
                        strides=1,
                        kernel_size=3,
                        act=act,
                        norm=norm,
                    ),
                    *[
                        Convolution(
                            spatial_dims,
                            out_c,
                            out_c,
                            strides=1,
                            kernel_size=3,
                            act=act,
                            norm=norm,
                        )
                        for _ in range(num_res_units)
                    ]
                 )
            )
            current_c = out_c
            
        self.final = Convolution(spatial_dims, current_c, out_channels, kernel_size=1, act=None, norm=None)

    @classmethod
    def from_config(cls, cfg):
        latent_dim = cfg['model'].get('latent_dim', 256)
        return cls(
            spatial_dims=cfg['model']['spatial_dims'],
            in_channels=cfg['model']['in_channels'],
            out_channels=cfg['data']['num_classes'],
            channels=tuple(cfg['model']['channels']),
            strides=tuple(cfg['model']['strides']),
            num_res_units=cfg['model']['num_res_units'],
            latent_dim=latent_dim,
            norm="INSTANCE",
        )

    def reparameterize(self, mu, logvar):
        """
        Performs the reparameterization trick to sample z from the latent distribution.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Forward pass of the VAE-U-Net.
        Returns:
            tuple: (logits, mu, logvar)
        """
        skips = []
        
        for down in self.downs:
            x = down(x)
            skips.append(x)
        
        x = self.bottleneck_conv(x)
        
        B, C, H, W = x.shape
        x_flat = self.global_avg(x).view(B, -1)
        
        mu = self.mu(x_flat)
        logvar = self.logvar(x_flat)
        
        if self.training:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu 
        
        # Project back
        z_feat = self.decode_z(z)
        z_feat = z_feat.view(B, C, 1, 1).expand(B, C, H, W)
        
        # Replace bottleneck feature with VAE feature
        x = z_feat
        
        skips_to_use = skips[:-1][::-1]
        
        for i in range(len(self.ups)):
            x = self.ups[i](x)
            
            if i < len(skips_to_use):
                skip = skips_to_use[i]
                
                if x.shape[2:] != skip.shape[2:]:
                    x = torch.nn.functional.interpolate(x, size=skip.shape[2:], mode='nearest')
                
                x = torch.cat([x, skip], dim=1)
            
            x = self.blocks[i](x)
            
        logits = self.final(x)
        
        return logits, mu, logvar
