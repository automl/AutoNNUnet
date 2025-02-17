from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import argparse
import logging

from autonnunet.datasets import ALL_DATASETS
from autonnunet.evaluation import extract_incumbent

if __name__  == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--approach", type=str, default="hpo")
    argparser.add_argument("--configuration", type=str, default="3d_fullres")
    argparser.add_argument("--hpo_seed", type=int, default=0)
    args = argparser.parse_args()

    logger = logging.getLogger("Incumbent Extractor")
    logger.setLevel(logging.INFO)

    for dataset_name in ALL_DATASETS:
        logger.info(f"Extracting incumbent configuration for {args.approach} {dataset_name}.")
        extract_incumbent(
            dataset_name=dataset_name,
            approach=args.approach,
            configuration=args.configuration,
            hpo_seed=args.hpo_seed
        )