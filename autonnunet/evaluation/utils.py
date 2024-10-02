from __future__ import annotations
import warnings

warnings.filterwarnings("ignore")
import os

import pandas as pd
import zipfile

from autonnunet.utils import dataset_name_to_msd_task


import logging
from typing import Any

import torch
from autonnunet.utils import get_device, set_environment_variables
from autonnunet.utils.paths import AUTONNUNET_OUTPUT, AUTONNUNET_PREDCITIONS, NNUNET_PREPROCESSED, NNUNET_RAW
from omegaconf import DictConfig, OmegaConf


# According to the MSD, these predictions should not be included in the submission
IGNORE_PREDICTIONS = [
    "liver_141.nii.gz",
    "liver_156.nii.gz",
    "liver_160.nii.gz",
    "liver_161.nii.gz",
    "liver_162.nii.gz",
    "liver_164.nii.gz",
    "liver_167.nii.gz",
    "liver_182.nii.gz",
    "liver_189.nii.gz",
    "liver_190.nii.gz",
    "hepaticvessel_247.nii.gz"
]


def get_prediction_trainer(
    cfg: DictConfig
) -> Any:
    # We do lazy imports here to make everything pickable for SLURM
    from autonnunet.training import AutoNNUNetTrainer
    from batchgenerators.utilities.file_and_folder_operations import load_json
    from torch.backends import cudnn

    preprocessed_dataset_folder_base = NNUNET_PREPROCESSED / cfg.dataset.name
    plans_file = preprocessed_dataset_folder_base / f"{cfg.trainer.plans_identifier}.json"
    plans = load_json(plans_file)
    dataset_json = load_json(preprocessed_dataset_folder_base / "dataset.json")

    nnunet_trainer = AutoNNUNetTrainer(
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
    from autonnunet.inference import AutoNNUNetPredictor
    
    set_environment_variables()
    os.environ["nnUNet_results"] = "."      # noqa: SIM112
    os.environ["nnUNet_n_proc_DA"] = "20"   # noqa: SIM112

    if approach == "baseline":
        model_base_output_dir = AUTONNUNET_OUTPUT / approach / dataset_name / configuration
    else:
        model_base_output_dir = AUTONNUNET_OUTPUT / approach / dataset_name / configuration / "incumbent"

    # Read config from yaml
    cfg = DictConfig(OmegaConf.load(model_base_output_dir / "fold_0" / "config.yaml"))

    logger = logging.getLogger(__name__)
    logger.info("Starting prediction")

    predictor = AutoNNUNetPredictor(
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


def extract_incumbent(dataset_name: str, approach: str, configuration: str, smac_seed: int) -> None:
    output_dir = AUTONNUNET_OUTPUT / approach / dataset_name / configuration / str(smac_seed)
    target_dir = AUTONNUNET_OUTPUT / approach / dataset_name / configuration / "incumbent"

    incumbent_df = pd.read_csv(output_dir / "incumbent.csv")
    incumbent_config_id = int(incumbent_df["config_id"].values[-1])

    for fold in range(5):
        run_id = incumbent_config_id * 5 + fold
        target_fold_dir = target_dir / f"fold_{fold}"
        target_fold_dir.mkdir(exist_ok=True, parents=True)

        # Copy the run to the incumbent directory
        os.system(f"cp -r {run_id}/ {target_fold_dir}/")


def compress_msd_submission(approach: str, configuration: str):
    predictions_dir = AUTONNUNET_PREDCITIONS / approach
    target_path = AUTONNUNET_MSD_SUBMISSIONS / f"{approach}_{configuration}.zip"
    AUTONNUNET_MSD_SUBMISSIONS.mkdir(exist_ok=True)

    with zipfile.ZipFile(target_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for dataset_name in os.listdir(predictions_dir):
            task_name = dataset_name_to_msd_task(dataset_name)
            dataset_dir = predictions_dir / dataset_name / configuration
            for file in os.listdir(dataset_dir):
                if file in IGNORE_PREDICTIONS:
                    continue

                if not file.endswith(".nii.gz"):
                    continue

                file_path = dataset_dir / file

                # Save the file in the zip with the task name as a subdirectory
                zipf.write(file_path, task_name + "/" + file)

