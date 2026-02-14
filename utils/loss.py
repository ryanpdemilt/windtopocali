import torch
import torch.nn as nn
from torch.nn import Module

class _Loss(Module):

    """Class copied from Pytorch needed for the class CustomLoss"""

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = nn.functional._Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


def customloss(input, target, size_average=None, reduce=None, reduction='mean'):
    """Loss function designed for Wind-Topo

    "input" is a (n_sample, 2) pytorch tensor and contains predicted u and v values (standardized)
    "target" is a (n_sample, 2) pytorch tensor and contains the real (observed) u and v values (standardized)
    keyword arguments are copied from pytorch
    """

    # Copied from pytorch
    if size_average is not None or reduce is not None:
        reduction = nn.functional._Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction = nn.functional._Reduction.get_enum(reduction)

    # Custom loss function
    #    Parameters
    epsilon = 4
    tau = 0.425
    #    Scaling u and v to avoid squeezed distributions of predicted velocity
    vel_target = torch.sqrt(torch.sum(target**2, 1, keepdim=True))
    vel_input = torch.sqrt(torch.sum(input**2, 1, keepdim=True))
    beta = (epsilon + vel_target) / (epsilon + vel_input)
    err = torch.sum((input - target * beta)**2, 1)
    #    Pinball term to reduce the bias of the prediction
    L = err * tau
    ind_neg = (vel_input - vel_target) < 0
    ind_neg = ind_neg[:, 0]
    L[ind_neg] = err[ind_neg] * (1 - tau)
    L = torch.mean(L)

    # Alternative or as simple example: Mean Squared Error would be:
    # L = torch.mean(torch.sum((input - target)**2, 1))

    return L


class CustomLoss(_Loss):

    """Class for the custom loss function of Wind-Topo, copied from pytorch"""

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(CustomLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        return customloss(input, target, reduction=self.reduction)