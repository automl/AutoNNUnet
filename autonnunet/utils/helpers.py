from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from autonnunet.utils.paths import NNUNET_PREPROCESSED, NNUNET_RAW


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)    # noqa: NPY002
    random.seed(seed)


def read_objectives(skip_loss: bool = False) -> dict | None:
    if (skip_loss or os.path.exists("./validation/summary.json")) and os.path.exists("./progress.csv"):
        if skip_loss:
            loss = 1.0
        else:
            with open("./validation/summary.json") as f:
                metrics = json.load(f)
                loss = 1 - metrics["foreground_mean"]["Dice"]

        progress = pd.read_csv("./progress.csv")
        progress["runtime"] = progress["epoch_end_timestamps"] - progress["epoch_start_timestamps"]
        runtime = float(progress["runtime"].sum())

        # Seconds to hours
        runtime = runtime / 3600

        return {
            "loss": loss,
            "runtime": runtime
        }
    return None


def set_environment_variables() -> None:
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
    return f"Task{dataset_name[8:]}"

def msd_task_to_dataset_name(msd_task: str) -> str:
    return f"Dataset0{msd_task[4:]}"

def format_dataset_name(dataset_name: str) -> str:
    # Extract number from Dataset002_Heart
    name = dataset_name[11:]
    id = dataset_name[8:10]

    return f"D{id} ({name})"

def load_json(file: Path) -> dict:
    with open(file) as f:
        return json.load(f)

def get_train_val_test_names(cfg: DictConfig) -> tuple[list[str], list[str], list[str]]:
    raw_folder = NNUNET_RAW / cfg.dataset.name
    preprocessed_folder = NNUNET_PREPROCESSED / cfg.dataset.name
    dataset_info = load_json(preprocessed_folder / "dataset.json")
    n_channels = len(dataset_info["channel_names"])
    splits = load_json(preprocessed_folder / "splits_final.json")

    train_imgs = []
    for name in splits[cfg.fold]["train"]:
        for i in range(n_channels):
            name = f"{name}_{'%04d' % i}.nii.gz"
            if (NNUNET_RAW / cfg.dataset.name / "imagesTr" / name).exists():
                train_imgs += [name]
        
    val_imgs = []
    for name in splits[cfg.fold]["val"]:
        for i in range(n_channels):
            name = f"{name}_{'%04d' % i}.nii.gz"
            if (NNUNET_RAW / cfg.dataset.name / "imagesTr" / name).exists():
                val_imgs += [name]

    test_imgs = []
    for name in raw_folder.glob("imagesTs/*.nii.gz"):
        test_imgs.append(name.name)

    return train_imgs, val_imgs, test_imgs