from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch


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
    else:
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
