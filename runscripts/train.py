from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import torch
from autonnunet.utils import (
    check_if_job_already_done,
    read_metrics,
    seed_everything,
    write_performance,
)
from autonnunet.utils.paths import NNUNET_PREPROCESSED
from codecarbon import OfflineEmissionsTracker

if TYPE_CHECKING:
    from autonnunet.training import AutoNNUNetTrainer
    from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs", config_name="train")
def run(cfg: DictConfig):
    # We have to do lazy imports here to make everything pickable for SLURM
    from autonnunet.training import AutoNNUNetTrainer
    from torch.backends import cudnn
    
    if torch.cuda.is_available():
        cudnn.deterministic = True
        cudnn.benchmark = True
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info("Starting training script")
    logger.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

    performance = check_if_job_already_done()
    if performance is not None and cfg.pipeline.return_if_done:
        logger.info(f"Job already done. Returning performance: {performance}")
        return performance

    logger.info("Setting seed")
    seed_everything(cfg.seed)

    logger.info("Setting up emissions tracker")
    tracker = OfflineEmissionsTracker(
        country_iso_code="DEU",
        log_level="WARNING"
    )
    tracker.start()

    logger.info("Creating trainer")
    nnunet_trainer = AutoNNUNetTrainer.from_config(cfg)

    if cfg.pipeline.run_training:
        logger.info("Starting training")
        nnunet_trainer.run_training()
        logger.info("Training finished")

    if cfg.pipeline.run_validation:
        logger.info("Starting validation")
        if cfg.pipeline.validate_with_best:
            nnunet_trainer.load_checkpoint(
                str(Path(nnunet_trainer.output_folder) /  "checkpoint_best.pth")
            )

        nnunet_trainer.perform_actual_validation(cfg.trainer.export_validation_probabilities)
        metrics = read_metrics()
        logger.info("Validation finished")

        val_dice_score = metrics["foreground_mean"]["Dice"]
        performance = 1 - val_dice_score

        write_performance(performance)

        logger.info(f"Mean Validation Dice Score: {val_dice_score}")
        logger.info(f"Performance: {performance}")

    tracker.stop()
    return performance


if __name__  == "__main__":
    sys.exit(run())