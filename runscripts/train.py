from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import logging
import shutil
import sys
import os
from pathlib import Path
from typing import TYPE_CHECKING

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
    from autonnunet.utils import (
        read_performance,
        seed_everything,
        write_performance,
    )
    from torch.backends import cudnn

    
    if torch.cuda.is_available():
        cudnn.deterministic = False
        cudnn.benchmark = True
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info("Starting training script")
    logger.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

    logger.info("Setting seed")
    seed_everything(cfg.seed)

    # --------------------------------------------------------------------------------------------
    # READ PEFROMANCE TO CHECK IF JOB IS DONE
    # --------------------------------------------------------------------------------------------
    performance = read_performance()
    if performance is not None and cfg.pipeline.return_if_done:
        logger.info(f"Job already done. Returning performance: {performance}")
        return performance

    # --------------------------------------------------------------------------------------------
    # CODECARBON SETUP
    # --------------------------------------------------------------------------------------------
    logger.info("Setting up emissions tracker")
    tracker = OfflineEmissionsTracker(
        country_iso_code="DEU",
        log_level="WARNING"
    )
    logger.info("Creating trainer")

    # --------------------------------------------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------------------------------------------
    nnunet_trainer = AutoNNUNetTrainer.from_config(cfg)

    if cfg.pipeline.run_training:
        if Path("./checkpoint_final.pth").exists() and cfg.pipeline.continue_training:
            logger.info("Found checkpoint_final.pth. Skipping training.")
        else:
            tracker.start()

            logger.info("Starting training")
            nnunet_trainer.run_training()
            logger.info("Training finished")

            tracker.stop()

            if cfg.save:
                logger.info("Saving model to checkpoint dir.")
                save_path_best = Path(cfg.save + f"_fold_{cfg.fold}_best.pth").resolve()
                save_path_final = Path(cfg.save + f"_fold_{cfg.fold}_final.pth").resolve()
                checkpoint_final_path = Path(".").resolve() / "checkpoint_final.pth"
                checkpoint_best_path = Path(".").resolve() / "checkpoint_best.pth"

                shutil.copy(checkpoint_final_path, save_path_final)
                shutil.copy(checkpoint_best_path, save_path_best)
                logger.info(f"Saved model to {save_path_best} and {save_path_final}")

    # --------------------------------------------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------------------------------------------
    if cfg.pipeline.run_validation:
        logger.info("Starting validation")
        if cfg.pipeline.validate_with_best:
            nnunet_trainer.load_checkpoint(
                str(Path(nnunet_trainer.output_folder) /  "checkpoint_best.pth")
            )

        nnunet_trainer.perform_actual_validation(
            save_probabilities=cfg.trainer.export_validation_probabilities
        )
        logger.info("Validation finished")

        if cfg.pipeline.remove_validation_files:
            logger.info("Removing validation files")
            # remove all .nii.gz files in the validation folder
            for file_path in (Path(nnunet_trainer.output_folder) /  "validation").iterdir():
                if ".nii.gz" in file_path.name:
                    os.remove(file_path.resolve())

        performance = read_performance()
        assert performance is not None

        write_performance(performance)

        logger.info(f"Mean Validation Dice Score: {1 - performance}")
        logger.info(f"Performance: {performance}")

    return performance


if __name__  == "__main__":
    sys.exit(run())