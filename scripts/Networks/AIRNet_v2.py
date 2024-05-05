import torch
import torch.nn as nn
from math import pi
import sys
from Mics.utils import ScaledTanH, ScalingAF
from Networks.spatial_transformer import AffineTransformer


class AIRNet_v2(nn.Module):
  def __init__(self, linear_nodes=64, ndim=2):
    super(AIRNet_v2, self).__init__()

    self.max_scaling = 2
    self.max_rotation = 0.5 * pi
    self.max_shearing = 0.25 * pi

    self.conv_layers = nn.Sequential(

        # Conv layer with batch normalization
        # and maxpool layer.
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.MaxPool2d(kernel_size=2),

        # Conv layer with batch normalization
        # and maxpool layer.
        nn.Conv2d(32, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.MaxPool2d(kernel_size=2),

        # Conv layer with batch normalization
        # and maxpool layer.
        nn.Conv2d(32, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.MaxPool2d(kernel_size=2),)

        # Flatten then pass to 3 fully connected
        # layers, each FCL follow with a dropout
        # layer
    self.regression_layers = nn.Sequential(        
        nn.Linear(14400, 512),
        nn.ReLU(),
        nn.Dropout(0.7),
        
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.7),
        
        # Output 10 classes
        nn.Linear(128, linear_nodes)
        )
    
    self.translation = nn.Linear(linear_nodes, ndim)

    self.rotation = nn.Sequential(
        nn.Linear(linear_nodes, 1 if ndim == 2 else 3),
        ScaledTanH(self.max_rotation),
    )

    self.scaling = nn.Sequential(nn.Linear(linear_nodes, ndim), ScalingAF(2))

    self.shearing = nn.Sequential(
        nn.Linear(linear_nodes, (ndim - 1) * ndim), ScaledTanH(self.max_shearing)
    )

    self.transformer = AffineTransformer(ndim=ndim)

  def forward(self, fixed, moving):
    f = self.conv_layers(fixed)
    m = self.conv_layers(moving)
    x = torch.cat((f.flatten(1), m.flatten(1)), dim=1)
    x = self.regression_layers(x)

    translation = self.translation(x)
    rotation = self.rotation(x)
    scale = self.scaling(x)
    shear = self.shearing(x)

    parameters = (translation, rotation, scale, shear)
    wraped = self.transformer(parameters, fixed, moving)

    return wraped, parameters