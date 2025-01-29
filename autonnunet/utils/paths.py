from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DATA_DIR = {
    "nnUNet_datasets": Path("./data/nnUNet_datasets").resolve(),
    "nnUNet_raw": Path("./data/nnUNet_raw").resolve(),
    "nnUNet_preprocessed": Path("./data/nnUNet_preprocessed").resolve(),
    "nnUNet_results": Path("./data/nnUNet_results").resolve(),
    "MedSAM2_preprocessed": Path("./data/MedSAM2_preprocessed").resolve(),
}

for k, v in DEFAULT_DATA_DIR.items():
    if not os.environ.get(k):
        os.environ[k] = str(v)

AUTONNUNET_CONFIGS = Path("./runscripts/configs").resolve()
AUTONNUNET_OUTPUT = Path("./output").resolve()
AUTONNUNET_RESULTS = Path("./results").resolve()
AUTONNUNET_PREDICTIONS =  Path("./output/predictions").resolve()
AUTONNUNET_MSD_SUBMISSIONS = Path("./output/msd_submissions").resolve()
AUTONNUNET_MSD_RESULTS = Path("./output/msd_results").resolve()

AUTONNUNET_PLOTS = Path("./thesis/plots").resolve()
AUTONNUNET_TABLES = Path("./thesis/tables").resolve()

NNUNET_DATASETS = Path(os.environ["nnUNet_datasets"]).resolve()    # noqa: SIM112
NNUNET_RAW =  Path(os.environ["nnUNet_raw"]).resolve()              # noqa: SIM112
NNUNET_PREPROCESSED =  Path(os.environ["nnUNet_preprocessed"]).resolve() # noqa: SIM112
NNUNET_RESULTS =  Path(os.environ["nnUNet_results"]).resolve()           # noqa: SIM112
MEDSAM2_PREPROCESSED = Path(os.environ["MedSAM2_preprocessed"]).resolve()    # noqa: SIM112

