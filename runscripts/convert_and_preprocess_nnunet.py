"""Convert and preprocess a dataset for nnU-Net."""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import logging
import sys
from typing import TYPE_CHECKING

import hydra

from autonnunet.datasets import MSDDataset

if TYPE_CHECKING:
    from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="configs", config_name="convert_and_preprocess_nnunet")
def run(cfg: DictConfig):
    logging.basicConfig(format="%(asctime)s %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    dataset = MSDDataset(
        name=cfg.dataset.name,
    )

    dataset.convert()
    dataset.preprocess()


if __name__  == "__main__":
    sys.exit(run())