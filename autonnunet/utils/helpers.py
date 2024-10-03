from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    np.random.seed(seed)    # noqa: NPY002
    random.seed(seed)


def get_device(device: str) -> torch.device:
    # taken from nnunetv2.run.run_training
    assert device in [
        "cpu",
        "cuda",
        "mps",
    ], f"-device must be either cpu, mps or cuda. Got: {device}."
    if device == "cpu":
        # let's allow torch to use hella threads
        import multiprocessing

        torch.set_num_threads(multiprocessing.cpu_count())
        return torch.device("cpu")
    if device == "cuda":
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        return torch.device("cuda")
    return torch.device("mps")


def check_if_job_already_done() -> float | None:
    try:
        with open("./performance.csv") as pf:
            csvreader = csv.reader(pf)
            performance = next(csvreader)
            return float(performance[0])

    except FileNotFoundError:
        return None


def write_performance(performance: float) -> None:
    with open("./performance.csv", "w+") as f:
        f.write(str(performance))


def read_metrics() -> dict:
    with open("./validation/summary.json") as f:
        return json.load(f)


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
