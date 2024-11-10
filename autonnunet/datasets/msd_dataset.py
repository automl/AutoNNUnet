from __future__ import annotations

import argparse
import os
import subprocess

from autonnunet.utils.paths import NNUNET_DATASETS

from .base_dataset import Dataset
from .utils import download_file, untar_file

MSD_URLS = {
    "Dataset001_BrainTumour": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar",
    "Dataset002_Heart": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task02_Heart.tar",
    "Dataset003_Liver": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task03_Liver.tar",
    "Dataset004_Hippocampus": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar",
    "Dataset005_Prostate": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task05_Prostate.tar",
    "Dataset006_Lung": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task06_Lung.tar",
    "Dataset007_Pancreas": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task07_Pancreas.tar",
    "Dataset008_HepaticVessel": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task08_HepaticVessel.tar",
    "Dataset009_Spleen": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar",
    "Dataset010_Colon": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task10_Colon.tar",
}

MSD_DATASETS = list(MSD_URLS.keys())


class MSDDataset(Dataset):
    def __init__(self, name: str, **kwargs) -> None:
        if name not in MSD_URLS:
            raise ValueError(f"Dataset {name} not found in MSD datasets.")
        super().__init__(name, **kwargs)

    def get_url(self) -> str:
        return MSD_URLS[self.name]

    def download_and_extract(self) -> None:
        url = MSD_URLS[self.name]
        tar_filename = os.path.basename(url)
        tar_path = NNUNET_DATASETS / tar_filename

        download_file(url, tar_path)
        untar_file(tar_path, self.dl_dataset_dir)

        tar_directory = tar_filename.split(".")[0]
        extracted_directory = self.dl_dataset_dir / tar_directory

        # Rename extracted directory to dataset name
        self.logger.info(f"Finished dataset extraction to {extracted_directory}")

    def convert(self) -> None:
        if self.raw_dataset_path.is_dir():
            self.logger.info(f"Dataset {self.name} already converted.")
            return

        # Since nnU-Net expects MSD datasets to be stored in "TaskXX_YY" format,
        # we need to convert the extracted dataset to this format
        tar_filename = os.path.basename(MSD_URLS[self.name])
        task_name = tar_filename.replace(".tar", "")
        task_path = self.dl_dataset_dir / task_name

        convert_command = [
            "nnUNetv2_convert_MSD_dataset",
            "-i", task_path,
            "-overwrite_id", str(self.dataset_id).zfill(3)
        ]
        self.logger.info(f"Executing command: {' '.join(convert_command)}")
        subprocess.run(convert_command, check=True)

    def preprocess(self) -> None:
        if self.preprocessed_dataset_path.is_dir():
            self.logger.info(f"Dataset {self.name} already preprocessed.")
            return

        preprocess_command = [
            "nnUNetv2_plan_and_preprocess",
            "-d", str(self.dataset_id).zfill(3),
            "--verify_dataset_integrity"
        ]
        self.logger.info(f"Executing command: {' '.join(preprocess_command)}")
        subprocess.run(preprocess_command, check=True)


if __name__  == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset_name", type=str, required=True)
    args = argparser.parse_args()

    dataset = MSDDataset(name=args.dataset_name)
    dataset.download_and_extract()