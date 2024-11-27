from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import argparse
from autonnunet.datasets import ALL_DATASETS

from autonnunet.evaluation import run_prediction, extract_incumbent, compress_msd_submission
import logging


if __name__  == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--approach", type=str, default="baseline")
    argparser.add_argument("--configuration", type=str, default="3d_fullres")
    argparser.add_argument("--hpo_seed", type=int, default=0)
    argparser.add_argument("--use_folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = argparser.parse_args()

    logger = logging.getLogger("Prediction")
    logger.setLevel(logging.INFO)
    logger.info("Starting MSD evaluation.")

    for dataset_name in ALL_DATASETS:
        logger.info(f"Running prediction for {dataset_name}.")
        run_prediction(
            dataset_name=dataset_name,
            approach=args.approach,
            configuration=args.configuration,
            use_folds=args.use_folds
        )
        logger.info(f"Done.")


    logger.info("Compressing MSD submission.")
    compress_msd_submission(
        approach=args.approach,
        configuration=args.configuration
    )
    logger.info(f"Done.")

        
