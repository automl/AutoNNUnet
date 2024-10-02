from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import torch
from automis.utils import get_device, set_environment_variables
from automis.utils.paths import AUTONNUNET_OUTPUT, AUTONNUNET_PREDCITIONS, NNUNET_PREPROCESSED, NNUNET_RAW
from omegaconf import DictConfig, OmegaConf


def get_trainer(
    cfg: DictConfig
) -> Any:
    # We do lazy imports here to make everything pickable for SLURM
    from automis.training import CustomNNUNetTrainer
    from batchgenerators.utilities.file_and_folder_operations import load_json
    from torch.backends import cudnn

    preprocessed_dataset_folder_base = NNUNET_PREPROCESSED / cfg.dataset.name
    plans_file = preprocessed_dataset_folder_base / f"{cfg.trainer.plans_identifier}.json"
    plans = load_json(plans_file)
    dataset_json = load_json(preprocessed_dataset_folder_base / "dataset.json")

    nnunet_trainer = CustomNNUNetTrainer(
        plans=plans,
        configuration=cfg.trainer.configuration,
        fold=0,
        dataset_json=dataset_json,
        unpack_dataset=not cfg.trainer.use_compressed_data,
        device=get_device(cfg.device),
    )
    nnunet_trainer.set_hp_config(cfg.hp_config)

    if torch.cuda.is_available():
        cudnn.deterministic = True
        cudnn.benchmark = True

    return nnunet_trainer


def run_prediction(
        dataset_name: str,
        approach: str,
        configuration: str,
        use_folds: tuple[int]
    ):
    from automis.inference import CustomNNUNetPredictor

    if approach == "baseline":
        model_base_output_dir = AUTONNUNET_OUTPUT / approach / dataset_name / configuration
    else:
        model_base_output_dir = AUTONNUNET_OUTPUT / approach / dataset_name / configuration / "incumbent"

    # Read config from yaml
    cfg = DictConfig(OmegaConf.load(model_base_output_dir / "fold_0" / "config.yaml"))

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.info("Starting prediction")

    predictor = CustomNNUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False
    )

    predictor.initialize_from_config(
        hp_config=cfg.hp_config,
        model_training_output_dir=str(model_base_output_dir),
        use_folds=use_folds,
        checkpoint_name="checkpoint_best.pth",
    )

    source_folder = str(NNUNET_RAW / cfg.dataset.name / "imagesTs")
    target_folder = str(AUTONNUNET_PREDCITIONS / approach / dataset_name / configuration)

    predictor.predict_from_files(
        source_folder,
        target_folder,
        save_probabilities=False,
        overwrite=False,
        num_processes_preprocessing=int(os.environ["nnUNet_n_proc_DA"]) // 2,       # noqa: SIM112
        num_processes_segmentation_export=int(os.environ["nnUNet_n_proc_DA"]) // 2, # noqa: SIM112
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0
    )

    logger.info("Finished prediction")


if __name__  == "__main__":
    set_environment_variables()
    os.environ["nnUNet_results"] = "."      # noqa: SIM112
    os.environ["nnUNet_n_proc_DA"] = "20"   # noqa: SIM112

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset_name", type=str, required=True)
    argparser.add_argument("--approach", type=str, default="baseline")
    argparser.add_argument("--configuration", type=str, default="3d_fullres")
    argparser.add_argument("--use_folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = argparser.parse_args()

    run_prediction(
        dataset_name=args.dataset_name,
        approach=args.approach,
        configuration=args.configuration,
        use_folds=args.use_folds
    )
