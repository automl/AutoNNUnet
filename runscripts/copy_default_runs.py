from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from autonnunet.datasets import ALL_DATASETS

if __name__  == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--configuration", type=str, default="3d_fullres")
    argparser.add_argument("--approach", type=str, default="hpo")
    args = argparser.parse_args()

    output_dir = Path("./").resolve() / "output"
    for dataset in ALL_DATASETS:
        for fold in range(5):
            source = output_dir / "baseline_ConvolutionalEncoder" / dataset / args.configuration / f"fold_{fold}"
            target = output_dir / args.approach / dataset / args.configuration / "0" / f"{fold}"

            if target.exists():
                continue

            shutil.copytree(source, target)
            print(f"Copied {dataset}, fold {fold}.")
