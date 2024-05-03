import torch
import sys
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union
from torch import nn, Tensor, Size
from Mics.utils import *

sys.dont_write_bytecode = True

class Transformer(ABC, nn.Module):
    def __init__(self, ndim: int, coord_dim: int = 1):
        super().__init__()
        self.ndim = ndim
        self.coord_dim = coord_dim
        self._resampler = Resampler(coord_dim=coord_dim)

    @abstractmethod
    def apply_transform(
        self,
        parameters: Tensor,
        fixed_image: Tensor,
        moving_image: Tensor,
        coordinate_grid: Optional[Tensor] = None,
    ) -> Tensor:
        """apply the parameters to get the transformed coord_grid"""
        pass

    def forward(
        self,
        parameters: Tensor,
        fixed_image: Tensor,
        moving_image: Tensor,
        coordinate_grid: Optional[Tensor] = None,
        return_coordinate_grid: bool = False,
    ) -> Tuple[Tensor]:

        coordinate_grid = self.apply_transform(
            parameters, fixed_image, moving_image, coordinate_grid
        )

        ret = self._resampler(moving_image, coordinate_grid)
        if return_coordinate_grid:
            ret = (ret, coordinate_grid)
        return ret


class AffineTransformer(Transformer):
    def apply_transform(
        self,
        parameters: Tuple[Tensor],
        fixed_image: Tensor,
        moving_image: Tensor,
        coordinate_grid: Optional[Tensor] = None,
    ) -> Tensor:
        """
        :param parameters: translation, rotation, scale, shear
        :param fixed_image:
        :param moving_image:
        :param coordinate_grid:
        :return:

        Note: If ndim == 3, number of parameters are 3 for translation, rotation, and scale; and
        6 for shear. If ndim == 2, number of parameters are 2 for translation, scale, and shear;
        and 1 for rotation
        """
        translation, rotation, scale, shear = parameters

        if self.ndim == 2:
            rot_mat = rotation_matrix_eff(rotation, ndim=self.ndim)
        elif self.ndim == 3:
            rot_mat = (
                rotation_matrix_eff(rotation[:, 0], axis=2, ndim=self.ndim)
                @ rotation_matrix_eff(rotation[:, 1], axis=1, ndim=self.ndim)
                @ rotation_matrix_eff(rotation[:, 2], axis=0, ndim=self.ndim)
            )

        Tmat = (
            scaling_matrix_eff(scale, ndim=self.ndim)
            @ shearing_matrix_eff(shear, ndim=self.ndim)
            @ rot_mat
        )

        f_origin = -(
            torch.tensor(
                fixed_image.shape[2:],
                dtype=fixed_image.dtype,
                device=fixed_image.device,
            )[None]
            / 2
        )
        m_origin = -(
            torch.tensor(
                moving_image.shape[2:],
                dtype=moving_image.dtype,
                device=moving_image.device,
            )[None]
            / 2
        )

        if coordinate_grid is None:
            coordinate_grid = identity_grid(
                fixed_image.shape[2:],
                stackdim=0,
                dtype=fixed_image.dtype,
                device=fixed_image.device,
            )[None].movedim(1, self.coord_dim)

        coordinate_grid = batch_transform_efficient(
            coordinate_grid, self.coord_dim, Tmat, translation, f_origin, m_origin
        )

        return coordinate_grid