from __future__ import annotations

import warnings

from smac.intensifier.successive_halving import SuccessiveHalving

warnings.filterwarnings("ignore")

import numpy as np
from typing import TYPE_CHECKING

import hydra
import sys
import logging

if TYPE_CHECKING:
    from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs", config_name="train")
def run(cfg: DictConfig):
    lower = 1 - cfg.hp_config.num_epochs / 1000
    return np.random.uniform(lower, 1)

if __name__  == "__main__":
    sys.exit(run())