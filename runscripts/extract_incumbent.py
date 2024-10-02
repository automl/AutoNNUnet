from __future__ import annotations

import argparse
import os

import pandas as pd
from automis.utils.paths import AUTONNUNET_OUTPUT


def extract_incumbent(dataset_name: str, approach: str, configuration: str, seed: int) -> None:
    output_dir = AUTONNUNET_OUTPUT / approach / dataset_name / configuration / str(seed)
    target_dir = AUTONNUNET_OUTPUT / approach / dataset_name / configuration / "incumbent"

    incumbent_df = pd.read_csv(output_dir / "incumbent.csv")
    incumbent_config_id = int(incumbent_df["config_id"].values[-1])

    for fold in range(5):
        run_id = incumbent_config_id * 5 + fold
        target_fold_dir = target_dir / f"fold_{fold}"
        target_fold_dir.mkdir(exist_ok=True, parents=True)

        # Copy the run to the incumbent directory
        os.system(f"cp -r {run_id}/ {target_fold_dir}/")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset_name", type=str, required=True)
    argparser.add_argument("--approach", type=str, default="smac_mf")
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--configuration", type=str, default="3d_fullres")
    args = argparser.parse_args()

    extract_incumbent(
        dataset_name=args.dataset_name,
        approach=args.approach,
        configuration=args.configuration,
        seed=args.seed
    )