import os
import sys
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.dont_write_bytecode = True

def plot_loss(loss_values, dice_loss):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
    
    # Plot loss over epoch
    ax1.plot(loss_values)
    ax1.set_ylabel('NCC Loss', fontsize=13)
    
    # Plot accuracy over epoch
    ax2.plot(dice_loss)
    ax2.set_xlabel('Epochs', fontsize=13)
    ax2.set_ylabel('Dice', fontsize=13)
    
    plt.suptitle('Training NCC loss & Dice', fontsize=18)
    # plt.savefig('../Save_fig/'+'TrainLossAccuracy.png')
    plt.show()

def FindLatestModel(CheckPointPath):
    """
    Finds Latest Model in CheckPointPath
    Inputs:
    CheckPointPath - Path where you have stored checkpoints
    Outputs:
    LatestFile - File Name of the latest checkpoint
    """
    FileList = glob.glob(CheckPointPath + '*.pt') # * means all if need specific format then *.csv
    LatestFile = max(FileList, key=os.path.getctime)
    # Strip everything else except needed information
    # LatestFile = LatestFile.replace(CheckPointPath, '')
    # LatestFile = LatestFile.replace('.ckpt.index', '')
    return LatestFile

class StableStd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        assert tensor.numel() > 1
        ctx.tensor = tensor.detach()
        res = torch.std(tensor).detach()
        ctx.result = res.detach()
        return res

    @staticmethod
    def backward(ctx, grad_output):
        tensor = ctx.tensor.detach()
        result = ctx.result.detach()
        e = 1e-6
        assert tensor.numel() > 1
        return (
            (2.0 / (tensor.numel() - 1.0))
            * (grad_output.detach() / (result.detach() * 2 + e))
            * (tensor.detach() - tensor.mean().detach())
        )


stablestd = StableStd.apply

class ScaledTanH(nn.Module):
    def __init__(self, scaling):
        super().__init__()
        self.scaling = scaling

    def forward(self, input):
        return torch.tanh(input) * self.scaling

    def __repr__(self):
        return self.__class__.__name__ + "(" + "scaling = " + str(self.scaling) + ")"
    

class ScalingAF(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, input):
        return self.scale_factor ** torch.tanh(input)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "scale_factor="
            + str(self.scale_factor)
            + ")"
        )

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