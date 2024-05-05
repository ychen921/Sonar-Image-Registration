import torch
import torch.nn as nn
from math import pi
import sys
from Networks.ConvBlock import ConvBlock
from Mics.utils import ScaledTanH, ScalingAF
from Networks.spatial_transformer import AffineTransformer

sys.dont_write_bytecode = True

class AIRNet(nn.Module):
    def __init__(self, 
                 kernel_size=3, 
                 kernels=32, 
                 linear_nodes=64,
                 num_conv_layers=4,
                 num_dense_layers=2,
                 num_downsamplings=3,
                 ndim=2
                 ):
        
        super().__init__()
        
        assert (num_dense_layers >= 1), "Number of dense layers should at least be 1 (excluding the final dense output layer)."
        
        self.ndim = ndim
        AdaptiveAvgPool = (nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d)[ndim - 2]

        AF = nn.ELU

        self.max_scaling = 2
        self.max_rotation = 0.5 * pi
        self.max_shearing = 0.25 * pi

        in_channels = 1
        conv_layers = [
            ConvBlock(
                in_channels,
                kernels,
                kernel_size=kernel_size,
                downsample=(2,) * ndim,
                af=AF,
            )
        ]  # TODO: clean this hacky stuff
        for i in range(1, num_conv_layers):
            downsample = (
                (2,) * ndim if i < num_downsamplings else (1,) * ndim
            )  # TODO: clean this hacky stuff
            conv_layers.append(
                ConvBlock(
                    kernels,
                    kernels,
                    kernel_size=kernel_size,
                    downsample=downsample,
                    af=AF,
                )
            )
        conv_layers.append(AdaptiveAvgPool(1))
        self.convnet_features = nn.Sequential(*conv_layers)

        dense_layers = list()

        dense_layers.append(nn.Linear(2 * kernels, linear_nodes))
        dense_layers.append(AF(inplace=True))
        for i in range(1, num_dense_layers):
            dense_layers.append(nn.Linear(linear_nodes, linear_nodes))
            dense_layers.append(AF(inplace=True))

        self.regression_features = nn.Sequential(*dense_layers)

        self.translation = nn.Linear(linear_nodes, ndim)

        self.rotation = nn.Sequential(
            nn.Linear(linear_nodes, 1 if ndim == 2 else 3),
            ScaledTanH(self.max_rotation),
        )

        self.scaling = nn.Sequential(nn.Linear(linear_nodes, ndim), ScalingAF(2))

        self.shearing = nn.Sequential(
            nn.Linear(linear_nodes, (ndim - 1) * ndim), ScaledTanH(self.max_shearing)
        )

        self.transformer = AffineTransformer(ndim=2)

    def forward(self, fixed, moving):
        f = self.convnet_features(fixed)
        m = self.convnet_features(moving)
        x = torch.cat((f.flatten(1), m.flatten(1)), dim=1)
        x = self.regression_features(x)
        translation = self.translation(x)
        rotation = self.rotation(x)
        scale = self.scaling(x)
        shear = self.shearing(x)

        parameters = (translation, rotation, scale, shear)
        wraped = self.transformer(parameters, fixed, moving)

        return wraped, parameters