
import torch
import torch.nn as nn
from monai.networks.blocks import Convolution, UpSample

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
        """
        Initializes the VAE-U-Net model.

        Args:
            spatial_dims (int): Number of spatial dimensions (e.g., 2 for 2D, 3 for 3D).
            in_channels (int): Number of input channels.
            out_channels (int): Number of output classes/channels.
            channels (tuple): Sequence of channel counts for each encoder level.
            strides (tuple): Sequence of stride values for downsampling at each encoder level.
            num_res_units (int): Number of residual units per block.
            latent_dim (int): Dimensionality of the latent space (z).
            act (str): Activation function name.
            norm (str): Normalization type.
        """
        super().__init__()
        self.spatial_dims = spatial_dims
        
        # Encoder
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
            # UpSample
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

    def reparameterize(self, mu, logvar):
        """
        Performs the reparameterization trick to sample z from the latent distribution.

        Args:
            mu (torch.Tensor): Mean of the latent Gaussian.
            logvar (torch.Tensor): Log variance of the latent Gaussian.

        Returns:
            torch.Tensor: Sampled latent vector z.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Forward pass of the VAE-U-Net.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: (logits, mu, logvar)
                - logits (torch.Tensor): Segmentation output.
                - mu (torch.Tensor): Latent mean.
                - logvar (torch.Tensor): Latent log variance.
        """
        skips = []
        
        # Encoder
        for down in self.downs:
            x = down(x)
            skips.append(x)
        
        # Bottleneck
        x = self.bottleneck_conv(x)
        
        # VAE
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
        
        # Decoder
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
