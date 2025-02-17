"""Helper functions."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch

from autonnunet.utils.paths import NNUNET_PREPROCESSED, NNUNET_RAW

if TYPE_CHECKING:
    from omegaconf import DictConfig


def seed_everything(seed: int) -> None:
    """Seed all the random number generators.

    Parameters
    ----------
    seed : int
        The seed value.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)    # noqa: NPY002
    random.seed(seed)


def read_objectives(*, skip_loss: bool = False) -> dict | None:
    """Read the objectives from the progress.csv and validation/summary.json files.

    Parameters
    ----------
    skip_loss : bool, optional
        Whether to skip the loss computation, by default False.

    Returns:
    -------
    dict | None
        The objectives (loss and runtime) or None if the files do not exist.
    """
    if (skip_loss or Path("./validation/summary.json").exists()) and \
        Path("./progress.csv").exists():
        if skip_loss:
            loss = 1.0
        else:
            with open("./validation/summary.json") as f:
                metrics = json.load(f)
                loss = 1 - metrics["foreground_mean"]["Dice"]

        progress = pd.read_csv("./progress.csv")
        progress["runtime"] = (
            progress["epoch_end_timestamps"] - progress["epoch_start_timestamps"]
        )
        runtime = float(progress["runtime"].sum())

        # Seconds to hours
        runtime = runtime / 3600

        return {
            "loss": loss,
            "runtime": runtime
        }
    return None


def set_environment_variables() -> None:
    """Set the environment variables for the nnUNet directories."""
    os.environ["nnUNet_datasets"] = str( # noqa: SIM112
        Path("./data/nnUNet_datasets").resolve().as_posix()
    )
    os.environ["nnUNet_raw"] = str( # noqa: SIM112
        Path("./data/nnUNet_raw").resolve().as_posix()
    )
    os.environ["nnUNet_preprocessed"] = str( # noqa: SIM112
        Path("./data/nnUNet_preprocessed").resolve().as_posix()
    )
    os.environ["nnUNet_results"] = str( # noqa: SIM112
        Path("./data/nnUNet_results").resolve().as_posix()
    )

def dataset_name_to_msd_task(dataset_name: str) -> str:
    """Convert the dataset name to the MSD task name.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset.

    Returns:
    -------
    str
        The MSD task name.
    """
    return f"Task{dataset_name[8:]}"

def msd_task_to_dataset_name(msd_task: str) -> str:
    """Convert the MSD task name to the dataset name.

    Parameters
    ----------
    msd_task : str
        The MSD task name.

    Returns:
    -------
    str
        The dataset name.
    """
    return f"Dataset0{msd_task[4:]}"

def format_dataset_name(dataset_name: str) -> str:
    """Format the dataset name to a more readable format.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset.

    Returns:
    -------
    str
        The formatted dataset name.
    """
    # Extract number from Dataset002_Heart
    name = dataset_name[11:]
    dataset_id = dataset_name[8:10]

    return f"D{dataset_id} ({name})"

def load_json(file: Path) -> dict:
    """Load a JSON file.

    Parameters
    ----------
    file : Path
        The path to the JSON file.

    Returns:
    -------
    dict
        The content of the JSON file.
    """
    with open(file) as f:
        return json.load(f)

def get_train_val_test_names(cfg: DictConfig) -> tuple[list[str], list[str], list[str]]:
    """Get the names of the training, validation and test images for a given dataset.

    Parameters
    ----------
    cfg : DictConfig
        The configuration object.

    Returns:
    -------
    tuple[list[str], list[str], list[str]]
        The names of the training, validation and test images.
    """
    raw_folder = NNUNET_RAW / cfg.dataset.name
    preprocessed_folder = NNUNET_PREPROCESSED / cfg.dataset.name
    dataset_info = load_json(preprocessed_folder / "dataset.json")
    n_channels = len(dataset_info["channel_names"])
    splits = load_json(preprocessed_folder / "splits_final.json")

    train_imgs = []
    for name in splits[cfg.fold]["train"]:
        for i in range(n_channels):
            _name = f"{name}_{'%04d' % i}.nii.gz"
            if (NNUNET_RAW / cfg.dataset.name / "imagesTr" / _name).exists():
                train_imgs += [_name]

    val_imgs = []
    for name in splits[cfg.fold]["val"]:
        for i in range(n_channels):
            _name = f"{name}_{'%04d' % i}.nii.gz"
            if (NNUNET_RAW / cfg.dataset.name / "imagesTr" / _name).exists():
                val_imgs += [_name]

    test_imgs = []
    for name in raw_folder.glob("imagesTs/*.nii.gz"):
        test_imgs.append(name.name)

    return train_imgs, val_imgs, test_imgs