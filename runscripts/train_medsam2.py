from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

from typing import TYPE_CHECKING
import logging
import torch
import hydra
from codecarbon import OfflineEmissionsTracker
import subprocess
import os
import sys
from pathlib import Path

if TYPE_CHECKING:
    from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs", config_name="train_medsam2")
def run(cfg: DictConfig):
    logger = logging.getLogger()

    logger.setLevel(logging.INFO)
    logger.info("Starting training script")
    logger.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

    # --------------------------------------------------------------------------------------------
    # CODECARBON SETUP
    # --------------------------------------------------------------------------------------------
    logger.info("Setting up emissions tracker")
    tracker = OfflineEmissionsTracker(
        country_iso_code="DEU",
        log_level="WARNING"
    )

    sam_dir = Path(__file__).parent.parent / "submodules" / "Medical-SAM2"
    command_dir = sam_dir / cfg.command.target
    checkpoint_dir = sam_dir / "checkpoints" / cfg.command.sam_ckpt

    command = (
        f"python {command_dir} "
        f"-net {cfg.command.net} "
        f"-prompt {cfg.command.prompt} "
        f"-exp_name {cfg.command.exp_name} "
        f"-vis {cfg.command.vis} "
        f"-sam_ckpt {checkpoint_dir} "
        f"-sam_config {cfg.command.sam_config} "
        f"-image_size {cfg.command.image_size} "
        f"-out_size {cfg.command.out_size} "
        f"-b {cfg.command.batch_size} "
        f"-val_freq {cfg.command.val_freq} "
        f"-dataset {cfg.command.dataset} "
        f"-fold {cfg.command.fold} "
        f"-seed {cfg.command.seed} "
        f"-n_train_workers {cfg.command.n_train_workers} "
        f"-n_test_workers {cfg.command.n_test_workers} "
        f"-n_epochs {cfg.command.n_epochs} "
        f"-mode {cfg.command.mode} "
    )

    logger.info("Executing command:")
    logger.info(command)
    tracker.start()
    subprocess.run(command, shell=True, check=True, env=os.environ)
    tracker.stop()


if __name__  == "__main__":
    sys.exit(run())