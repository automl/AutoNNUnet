import shutil
from pathlib import Path
import argparse
from autonnunet.datasets import ALL_DATASETS


if __name__  == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--configuration", type=str, default="3d_fullres")
    args = argparser.parse_args()
    
    output_dir = Path("./").resolve() / "output"
    for dataset in ALL_DATASETS:
        for fold in range(5):
            source = output_dir / "baseline" / dataset / args.configuration / f"fold_{fold}"
            target = output_dir / "prior_band" / dataset / args.configuration / "0" / f"{fold}"

            if target.exists():
                continue

            shutil.copytree(source, target)
            print(f"Copied {dataset}, fold {fold}.")
