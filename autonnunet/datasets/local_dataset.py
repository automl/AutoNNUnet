"""Utilities for local/custom datasets already in nnU-Net raw format."""
from __future__ import annotations

import subprocess

from autonnunet.datasets.base_dataset import Dataset


class LocalDataset(Dataset):
    """Dataset that is already available in nnU-Net raw format on disk.

    Use this for custom datasets that were prepared the same way as for a
    regular nnU-Net (e.g. via `nnUNetv2_convert_dataset` or a custom
    conversion script), instead of one of the Medical Segmentation
    Decathlon (MSD) datasets.
    """
    def __init__(self, name: str) -> None:
        """Initializes a local dataset.

        Parameters
        ----------
        name : str
            The name of the dataset, matching the `DatasetXXX_Name` folder
            in `nnUNet_raw`.
        """
        super().__init__(name)

        if not self.raw_dataset_path.is_dir():
            raise FileNotFoundError(
                f"Expected raw dataset at {self.raw_dataset_path}, but it "
                "does not exist. LocalDataset requires the dataset to "
                "already be in nnU-Net raw format (imagesTr/labelsTr/"
                "dataset.json)."
            )

    def download_and_extract(self) -> None:
        """No-op: the dataset is already available locally."""
        self.logger.info(f"Dataset {self.name} is local, skipping download.")

    def convert(self) -> None:
        """No-op: the dataset is already in nnU-Net raw format."""
        self.logger.info(f"Dataset {self.name} is already in nnU-Net raw format.")

    def preprocess(self) -> None:
        """Preprocesses the dataset using nnU-Net, unless already done.

        Raises:
        ------
        subprocess.CalledProcessError
            If the preprocessing fails.
        """
        if self.preprocessed_dataset_path.is_dir():
            self.logger.info(f"Dataset {self.name} already preprocessed.")
            return

        preprocess_command = [
            "nnUNetv2_plan_and_preprocess",
            "-d", str(self.dataset_id).zfill(3),
            "--verify_dataset_integrity"
        ]
        self.logger.info(f"Executing command: {' '.join(preprocess_command)}")
        subprocess.run(preprocess_command, check=True)  # noqa: S603
