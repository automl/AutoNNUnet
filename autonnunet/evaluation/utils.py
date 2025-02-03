from __future__ import annotations

import ast
import os
import zipfile

import pandas as pd
import torch
import yaml
from omegaconf import OmegaConf

from autonnunet.utils import dataset_name_to_msd_task, load_json
from autonnunet.utils.paths import (AUTONNUNET_CONFIGS,
                                    AUTONNUNET_MSD_SUBMISSIONS,
                                    AUTONNUNET_OUTPUT, AUTONNUNET_PREDICTIONS,
                                    NNUNET_RAW)

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


def run_prediction(
        dataset_name: str,
        approach: str,
        configuration: str,
        use_folds: tuple[int]
    ):
    from autonnunet.inference import AutoNNUNetPredictor

    os.environ["nnUNet_results"] = "."      # noqa: SIM112
    os.environ["nnUNet_n_proc_DA"] = "20"   # noqa: SIM112

    if "baseline" in approach:
        model_base_output_dir = AUTONNUNET_OUTPUT / approach / dataset_name / configuration
    else:
        model_base_output_dir = AUTONNUNET_OUTPUT / approach / dataset_name / configuration / "0" / "incumbent"

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
        model_training_output_dir=str(model_base_output_dir),
        use_folds=use_folds,
        checkpoint_name="checkpoint_best.pth",
    )

    source_folder = str(NNUNET_RAW / dataset_name / "imagesTs")
    target_folder = str(AUTONNUNET_PREDICTIONS / approach / dataset_name / configuration)

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


def extract_incumbent(
        dataset_name: str,
        approach: str,
        configuration: str,
        hpo_seed: int,
    ) -> None:
    output_dir = AUTONNUNET_OUTPUT / approach / dataset_name / configuration / str(hpo_seed)
    target_dir = AUTONNUNET_CONFIGS / "incumbent"

    target_dir.mkdir(exist_ok=True, parents=True)

    incumbent_df = pd.read_csv(output_dir / "incumbent_loss.csv")
    incumbent_config_id = int(incumbent_df["run_id"].values[-1])

    run_id = incumbent_config_id * 5
    debug_info_path = output_dir / str(run_id) / "debug.json"

    debug_info = load_json(debug_info_path)
    hp_config = ast.literal_eval(debug_info["hp_config"])
    actual_hp_config = {}

    search_space = OmegaConf.load(AUTONNUNET_CONFIGS / "search_space" / f"{approach}.yaml")
    search_space = dict(search_space.hyperparameters)

    for hp in hp_config:
        if hp == "num_epochs" or f"hp_config.{hp}" in search_space:
            actual_hp_config[hp] = hp_config[hp]

    yaml_dict = {
        "hp_config": actual_hp_config,
    }

    hydra_outpur_dir = f"output/{approach}/{dataset_name}/{configuration}/{hpo_seed}/incumbent"
    yaml_dict["hydra"] = {}
    yaml_dict["hydra"]["output_subdir"] = "'.'"
    yaml_dict["hydra"]["job_logging"] = {}
    yaml_dict["hydra"]["job_logging"]["stdout"] = True
    yaml_dict["hydra"]["job_logging"]["stderr"] = True
    yaml_dict["hydra"]["run"] = {}
    yaml_dict["hydra"]["run"]["dir"] = hydra_outpur_dir
    yaml_dict["hydra"]["sweep"] = {}
    yaml_dict["hydra"]["sweep"]["dir"] = hydra_outpur_dir
    yaml_dict["hydra"]["sweep"]["subdir"] = "fold_${fold}"
    yaml_dict["hydra"]["job"] = {}
    yaml_dict["hydra"]["job"]["chdir"] = True

    output_path = target_dir / f"{dataset_name}_{approach}.yaml"
    with open(output_path, "w") as f:
        f.write("# @package _global_\n")
        yaml.dump(yaml_dict, f)

    # Now we need to fix the output_subdir
    with open(output_path) as f:
        content = f.read()

    content = content.replace("'''.'''", '"."')

    with open(output_path, "w") as f:
        f.write(content)


def extract_incumbent_directories(dataset_name: str, approach: str, configuration: str, hpo_seed: int) -> None:
    output_dir = AUTONNUNET_OUTPUT / approach / dataset_name / configuration / str(hpo_seed)
    target_dir = AUTONNUNET_OUTPUT / approach / dataset_name / configuration / str(hpo_seed) / "incumbent"

    if target_dir.exists():
        return

    incumbent_df = pd.read_csv(output_dir / "incumbent_loss.csv")
    incumbent_run_id = int(incumbent_df["run_id"].values[-1])

    for fold in range(5):
        run_id = incumbent_run_id * 5 + fold
        source_dir_fold_dir = output_dir / str(run_id)
        target_fold_dir = target_dir / f"fold_{fold}"

        if target_fold_dir.exists():
            continue

        target_fold_dir.mkdir(exist_ok=True, parents=True)

        # Copy the run to the incumbent directory
        os.system(f"cp -r {source_dir_fold_dir}/* {target_fold_dir}/")


def compress_msd_submission(approach: str, configuration: str):
    predictions_dir = AUTONNUNET_PREDICTIONS / approach
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

