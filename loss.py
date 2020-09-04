import torch
from torch import nn


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        # target is a scalar
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def confidence_penalty(logit):
    return (-logit * torch.log(logit + EPS)).sum(dim=-1).mean()


def laplacian_reg(edge_index, y):
    row, col = edge_index
    y_r, y_c = y[row], y[col]
    l2_norm = torch.norm(y_r - y_c, dim=1, p=2)
    reg = l2_norm.mean()
    return edge_index.size(0) * reg * reg

