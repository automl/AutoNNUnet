from __future__ import annotations

from torch.optim.lr_scheduler import _LRScheduler


class DummyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]