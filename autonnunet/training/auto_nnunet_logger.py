from __future__ import annotations

import os

import numpy as np
import pandas as pd
from nnunetv2.training.logging.nnunet_logger import nnUNetLogger


class AutoNNUNetLogger(nnUNetLogger):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)

    def plot_progress_png(self, output_folder: str) -> None:
        super().plot_progress_png(output_folder)

        logging_data = self.my_fantastic_logging.copy()
        logging_data["dice_per_class_or_region"] = [
                [float(value) for value in np_array]
                for np_array in logging_data["dice_per_class_or_region"
            ]
        ]

        logging_df = pd.DataFrame(logging_data)
        logging_df["Epoch"] = np.arange(len(logging_df))

        logging_df.to_csv(
            os.path.join(output_folder, "progress.csv"), index=False
        )