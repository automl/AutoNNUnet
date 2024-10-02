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
    get_device,
    read_metrics,
    seed_everything,
    write_performance,
)
from autonnunet.utils.paths import NNUNET_PREPROCESSED
from codecarbon import OfflineEmissionsTracker

if TYPE_CHECKING:
    from autonnunet.training import AutoNNUNetTrainer
    from omegaconf import DictConfig


def get_trainer(
    cfg: DictConfig
) -> Any:
    # We do lazy imports here to make everything pickable for SLURM
    from autonnunet.training import AutoNNUNetTrainer
    from batchgenerators.utilities.file_and_folder_operations import load_json
    from nnunetv2.run.run_training import maybe_load_checkpoint
    from torch.backends import cudnn

    preprocessed_dataset_folder_base = NNUNET_PREPROCESSED / cfg.dataset.name  # noqa: SIM112
    plans_file = preprocessed_dataset_folder_base / f"{cfg.trainer.plans_identifier}.json"
    plans = load_json(plans_file)
    dataset_json = load_json(preprocessed_dataset_folder_base / "dataset.json")

    nnunet_trainer = AutoNNUNetTrainer(
        plans=plans,
        configuration=cfg.trainer.configuration,
        fold=cfg.fold,
        dataset_json=dataset_json,
        unpack_dataset=not cfg.trainer.use_compressed_data,
        device=get_device(cfg.device),
    )
    nnunet_trainer.set_hp_config(cfg.hp_config)

    nnunet_trainer.disable_checkpointing = cfg.trainer.disable_checkpointing

    maybe_load_checkpoint(
        nnunet_trainer=nnunet_trainer,
        continue_training=cfg.pipeline.continue_training,
        validation_only=cfg.pipeline.run_validation and not cfg.pipeline.run_training,
        pretrained_weights_file=cfg.trainer.pretrained_weights_file,
    )

    if torch.cuda.is_available():
        cudnn.deterministic = True
        cudnn.benchmark = True

    return nnunet_trainer


@hydra.main(version_base=None, config_path="configs", config_name="train")
def run(cfg: DictConfig):

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
    nnunet_trainer: AutoNNUNetTrainer = get_trainer(cfg)

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