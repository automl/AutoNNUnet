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

def read_objectives() -> dict | None:
    if os.path.exists("./validation/summary.json") and os.path.exists("./progress.csv"):
        with open("./validation/summary.json") as f:
            metrics = json.load(f)
            loss = 1 - metrics["foreground_mean"]["Dice"]

        progress = pd.read_csv("./progress.csv")
        progress["runtime"] = progress["epoch_end_timestamps"] - progress["epoch_start_timestamps"]
        runtime = float(progress["runtime"][1:].mean())

        return {
            "loss": loss,
            "epoch_runtime": runtime
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

def format_dataset_name(dataset_name: str) -> str:
    # Extract number from Dataset002_Heart
    name = dataset_name[11:]
    id = dataset_name[8:10]

    return f"D{id} ({name})"

def load_json(file: Path) -> dict:
    with open(file) as f:
        return json.load(f)
