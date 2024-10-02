from __future__ import annotations

import os
from pathlib import Path


AUTONNUNET_OUTPUT = Path("./output").resolve()
AUTONNUNET_PREDCITIONS =  Path("./predictions").resolve()
AUTONNUNET_PLOTS = Path("./plots").resolve()
AUTONNUNET_MSD_SUBMISSIONS = Path("./msd_submissions").resolve()

NNUNET_DATASETS =  Path("./data/nnUNet_datasets").resolve()
NNUNET_RAW =  Path("./data/nnUNet_raw").resolve()
NNUNET_PREPROCESSED =  Path("./data/nnUNet_preprocessed").resolve()
NNUNET_RESULTS =  Path("./data/nnUNet_results").resolve()


os.environ["nnUNet_datasets"] = str(NNUNET_DATASETS)            # noqa: SIM112
os.environ["nnUNet_raw"] = str(NNUNET_RAW)                      # noqa: SIM112
os.environ["nnUNet_preprocessed"] = str(NNUNET_PREPROCESSED)    # noqa: SIM112
os.environ["nnUNet_results"] = str(NNUNET_RESULTS)              # noqa: SIM112
