"""A dummy learning rate scheduler that does nothing."""
from __future__ import annotations

from typing import TYPE_CHECKING

from torch.optim.lr_scheduler import _LRScheduler

if TYPE_CHECKING:
    from torch.optim.optimizer import Optimizer


class DummyLRScheduler(_LRScheduler):
    """A dummy learning rate scheduler that does nothing."""
    def __init__(self, optimizer: Optimizer, last_epoch: int = -1):
        """Initialize the dummy learning rate scheduler.

        Parameters
        ----------
        optimizer : Optimizer
            The optimizer.

        last_epoch : int, optional
            The index of the last epoch. Defaults to -1.
        """
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Get the learning rate of the optimizer."""
        return [group["lr"] for group in self.optimizer.param_groups]