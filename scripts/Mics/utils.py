import torch
import torch.nn as nn
import torch.nn.functional as F


class Resampler(nn.Module):
    """
    Generic resampler for 2D and 3D images.
    Expects voxel coordinates as coord_field
    Args:
        input (Tensor): input batch (N x C x IH x IW) or (N x C x ID x IH x IW)
        grid (Tensor): flow-field of size (N x OH x OW x 2) or (N x OD x OH x OW x 3)
        padding_mode (str): padding mode for outside grid values
            'zeros' | 'border'. Default: 'zeros'
    """

    def __init__(
        self, coord_dim: int = 1, mode: str = "bilinear", padding_mode: str = "border"
    ):
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode
        self.coord_dim = coord_dim

    def forward(self, input, coord_grid):
        im_shape = input.shape[2:]
        assert coord_grid.shape[self.coord_dim] == len(
            im_shape
        )  # number of coordinates should match image dimension

        coord_grid = coord_grid.movedim(self.coord_dim, -1)

        # scale for pytorch grid_sample function
        max_extent = (
            torch.tensor(
                im_shape[::-1], dtype=coord_grid.dtype, device=coord_grid.device
            )
            - 1
        )
        coord_grid = 2 * (coord_grid / max_extent) - 1
        return F.grid_sample(
            input,
            coord_grid,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=True,
        )


def scaling_matrix_eff(x, ndim=2):
    assert (
        x.ndim == 2
    ), "Input should be a tensor of (m, n), where m is number of instances, and n number of parameters."
    assert ndim in (2, 3), "Only 2D and 3D implemented."
    assert ndim == x.shape[-1], f"Number of parameters for {ndim}D should be {ndim}."
    dtype, device = x.dtype, x.device
    T = torch.zeros((len(x), ndim, ndim), dtype=dtype, device=device)
    sel_mask = torch.eye(ndim, device=device, dtype=torch.bool)
    T[:, sel_mask] = x
    return T


def rotation_matrix_eff(x, axis=0, ndim=2):
    """
    For 3D axis = x: 2, y: 1, z: 0.
    """
    assert (
        x.ndim == 2 and x.shape[-1] == 1
    ), "Input should be a tensor of (m, 1), where m is number of instances."
    assert ndim in (2, 3), "Only 2D and 3D implemented."
    dtype, device = x.dtype, x.device
    T = torch.eye(ndim, dtype=dtype, device=device)[None].repeat(len(x), 1, 1)
    lidx, hidx = ((0, 1), (0, 2), (1, 2))[axis]
    c = torch.cos(x)
    s = torch.sin(x)
    T[:, lidx, lidx] = c.squeeze()
    T[:, lidx, hidx] = s.squeeze()
    T[:, hidx, lidx] = -s.squeeze()
    T[:, hidx, hidx] = c.squeeze()
    return T


def shearing_matrix_eff(x, ndim=2):
    assert (
        x.ndim == 2
    ), "Input should be a tensor of (m, n), where m is number of instances, and n number of parameters."
    assert ndim in (2, 3), "Only 2D and 3D implemented."
    assert (ndim - 1) * ndim == x.shape[
        -1
    ], f"Number of parameters for {ndim}D should be {(ndim - 1) * ndim}"
    dtype, device = x.dtype, x.device
    T = torch.eye(ndim, dtype=dtype, device=device)[None].repeat(len(x), 1, 1)
    T[:, ~torch.eye(ndim, device=device, dtype=torch.bool)] = torch.tan(x)
    return T


def translation_matrix(x, ndim=2):  # not used for efficient transforms
    assert (
        x.ndim == 2
    ), "Input should be a tensor of (m, n), where m is number of instances, and n number of parameters."
    assert ndim in (2, 3), "Only 2D and 3D implemented."
    assert ndim == x.shape[-1], f"Number of parameters for {ndim}D should be {ndim}."
    dtype, device = x.dtype, x.device
    T = torch.eye(ndim + 1, dtype=dtype, device=device)[None].repeat(len(x), 1, 1)
    T[:, :ndim, ndim] = x
    return T


def batch_transform_efficient(
    coord_grid, coord_dim, Tmat, translation, Forigin, Morigin
):
    # print(coord_grid.shape)
    coord_grid = coord_grid.movedim(coord_dim, -1)
    # print(coord_grid.shape)
    shape = coord_grid.shape
    coord_grid = coord_grid.view(shape[0], -1, shape[-1])  # flatten
    # print(coord_grid.shape, Forigin.shape)
    coord_grid = coord_grid + Forigin[:, None]
    coord_grid = coord_grid @ Tmat.transpose(
        2, 1
    )  # transpose because of switched order
    coord_grid = coord_grid + (translation[:, None] - Morigin[:, None])
    shape = (len(coord_grid),) + shape[1:]  # enable broadcasting
    coord_grid = coord_grid.view(shape)  # recover original shape
    coord_grid = coord_grid.movedim(-1, coord_dim)
    return coord_grid

def identity_grid(shape, stackdim, dtype=torch.float32, device="cpu"):
    """Create an nd identity grid."""
    tensors = (torch.arange(s, dtype=dtype, device=device) for s in shape)
    return torch.stack(
        torch.meshgrid(*tensors)[::-1], stackdim
    )  # z,y,x shape and flip for x, y, z coords