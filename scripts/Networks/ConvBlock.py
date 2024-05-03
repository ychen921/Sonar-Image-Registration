from math import pi
from collections import OrderedDict
import numpy as np
import torch
import sys
from torch import nn

sys.dont_write_bytecode = True

class ConvBlock(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        downsample=(False,),
        af=nn.ELU,
        ndim=2,
    ):
        Conv = (nn.Conv2d, nn.Conv3d)[ndim - 2]
        AvgPool = (nn.AvgPool2d, nn.AvgPool3d)[ndim - 2]
        padding = kernel_size // 2
        layers = OrderedDict()
        layers["conv"] = Conv(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding
        )
        if af:
            layers["af"] = af()
        if any(downsample):
            if any(np.asarray(downsample) > 1):  # TODO: improve checking
                layers["downsample"] = AvgPool(tuple(downsample))
        super().__init__(layers)