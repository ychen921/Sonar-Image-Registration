import math
from typing import Optional, Tuple
import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss
import sys
from Mics.utils import stablestd

sys.dont_write_bytecode = True

# TODO: add masks.
def ncc(x1, x2, e=1e-10):
    assert x1.shape == x2.shape, "Inputs are not of equal shape"
    cc = ((x1 - x1.mean()) * (x2 - x2.mean())).mean()
    std = stablestd(x1) * stablestd(x2)
    ncc = cc / (std + e)
    return ncc


def ncc_mask(x1, x2, mask, e=1e-10):  # TODO: calculate ncc per sample
    assert x1.shape == x2.shape, "Inputs are not of equal shape"
    x1 = torch.masked_select(x1, mask)
    x2 = torch.masked_select(x2, mask)
    cc = ((x1 - x1.mean()) * (x2 - x2.mean())).mean()
    std = stablestd(x1) * stablestd(x2)
    ncc = cc / (std + e)
    return ncc

class NCC(_Loss):
    def __init__(self, use_mask: bool = False):
        super().__init__()
        if use_mask:
            self.forward = self.masked_metric
        else:
            self.forward = self.metric

    def metric(self, fixed: Tensor, warped: Tensor) -> Tensor:
        return -ncc(fixed, warped)

    def masked_metric(self, fixed: Tensor, warped: Tensor, mask: Tensor) -> Tensor:
        return -ncc_mask(fixed, warped, mask)
    

def dice_score(pred, target):
    """This definition generalize to real valued pred and target vector.
This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    top = 2 *  torch.sum(pred * target, [1, 2, 3])
    union = torch.sum(pred + target, [1, 2, 3])
    eps = torch.ones_like(union) * 1e-5
    bottom = torch.max(union, eps)
    dice = torch.mean(top / bottom)
    #print("Dice score", dice)
    return dice