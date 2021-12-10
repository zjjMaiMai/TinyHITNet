import torch
from torchmetrics import AverageMeter


class EPEMetric(AverageMeter):
    def __init__(self):
        super().__init__()

    def update(self, pred, target, mask):
        error = torch.abs(target - pred) * mask
        error = torch.flatten(error, 1).sum(-1)
        count = torch.flatten(mask, 1).sum(-1)
        epe = error / count
        super().update(epe[count > 0])