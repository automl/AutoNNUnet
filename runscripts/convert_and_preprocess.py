"""Conver and preprocess a dataset."""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import logging
import sys
from typing import TYPE_CHECKING

import hydra

if TYPE_CHECKING:
    from autonnunet.datasets import Dataset
    from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs", config_name="convert_and_preprocess")
def run(cfg: DictConfig):
    logging.basicConfig(format="%(asctime)s %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    dataset: Dataset = hydra.utils.instantiate(cfg.dataset)

    dataset.convert()
    dataset.preprocess()


if __name__  == "__main__":
    sys.exit(run())