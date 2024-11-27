from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import argparse
from autonnunet.datasets import ALL_DATASETS

from autonnunet.evaluation import run_prediction, extract_incumbent, compress_msd_submission
import logging


if __name__  == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--approach", type=str, default="hpo")
    argparser.add_argument("--configuration", type=str, default="3d_fullres")
    argparser.add_argument("--hpo_seed", type=int, default=0)
    args = argparser.parse_args()

    logger = logging.getLogger("Prediction")
    logger.setLevel(logging.INFO)
    logger.info("Starting MSD evaluation.")

    for dataset_name in ALL_DATASETS:
        extract_incumbent(
            dataset_name=dataset_name,
            approach=args.approach,
            configuration=args.configuration,
            hpo_seed=args.hpo_seed
        )