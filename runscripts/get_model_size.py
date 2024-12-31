from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import logging
import shutil
import sys
import os
from pathlib import Path
from typing import TYPE_CHECKING
import time

import hydra
import torch
from codecarbon import OfflineEmissionsTracker

if TYPE_CHECKING:
    from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs", config_name="train")
def run(cfg: DictConfig):
    # --------------------------------------------------------------------------------------------
    # GENERAL SETUP
    # --------------------------------------------------------------------------------------------
    # We have to do lazy imports here to make everything pickable for SLURM.
    # Also, we need to ensure that environment variables are set
    # before importing AutoNNUNet
    from autonnunet.training import AutoNNUNetTrainer

    logger = logging.getLogger()
    
    logger.setLevel(logging.INFO)
    logger.info("Starting model size calculation script")
    logger.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")


    logger.info("Creating trainer")
    nnunet_trainer = AutoNNUNetTrainer.from_config(cfg=cfg)
    nnunet_trainer.initialize()

    model = nnunet_trainer.network
    model_size = sum(p.numel() for p in model.parameters())
    logger.info(f"Model size: {model_size}")


if __name__  == "__main__":
    sys.exit(run())