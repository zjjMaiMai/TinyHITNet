import torch
from torchmetrics import AverageMeter


class RateMetric(AverageMeter):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def update(self, pred, target, mask):
        error = torch.abs(target - pred)
        error = (error < self.threshold) & mask
        error = torch.flatten(error, 1).float().sum(-1)
        count = torch.flatten(mask, 1).sum(-1)
        rate = error / count * 100
        super().update(rate[count > 0])
