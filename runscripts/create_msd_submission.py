"""Create a MSD submission from the predictions of a given approach and configuration."""
from __future__ import annotations

import argparse
import os
import zipfile

from automis.utils import dataset_name_to_msd_task
from automis.utils.paths import AUTONNUNET_MSD_SUBMISSIONS, AUTONNUNET_PREDCITIONS


# According to the MSD, these predictions should not be included in the submission
IGNORE_PREDICTIONS = [
    "liver_141.nii.gz",
    "liver_156.nii.gz",
    "liver_160.nii.gz",
    "liver_161.nii.gz",
    "liver_162.nii.gz",
    "liver_164.nii.gz",
    "liver_167.nii.gz",
    "liver_182.nii.gz",
    "liver_189.nii.gz",
    "liver_190.nii.gz",
    "hepaticvessel_247.nii.gz"
]


def compress_msd_submission(approach: str, configuration: str):
    predictions_dir = AUTONNUNET_PREDCITIONS / approach
    target_path = AUTONNUNET_MSD_SUBMISSIONS / f"{approach}_{configuration}.zip"
    AUTONNUNET_MSD_SUBMISSIONS.mkdir(exist_ok=True)

    with zipfile.ZipFile(target_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for dataset_name in os.listdir(predictions_dir):
            task_name = dataset_name_to_msd_task(dataset_name)
            dataset_dir = predictions_dir / dataset_name / configuration
            for file in os.listdir(dataset_dir):
                if file in IGNORE_PREDICTIONS:
                    continue

                if not file.endswith(".nii.gz"):
                    continue

                file_path = dataset_dir / file

                # Save the file in the zip with the task name as a subdirectory
                zipf.write(file_path, task_name + "/" + file)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--approach", type=str, default="baseline")
    argparser.add_argument("--configuration", type=str, default="3d_fullres")
    args = argparser.parse_args()

    compress_msd_submission(
        approach=args.approach,
        configuration=args.configuration,
    )
