import torch
import torch.nn.functional as F
from utils import EPS


def soft_cross_entropy(predict, soft_target):
    return -(soft_target * torch.log(predict + EPS)).sum(dim=1).mean()


def kl_div(predict, soft_target):
    return F.kl_div(torch.log(soft_target), predict, reduction="batchmean")


def squared_error(predict, soft_target):
    return torch.norm(soft_target - predict, p=2, dim=1)
