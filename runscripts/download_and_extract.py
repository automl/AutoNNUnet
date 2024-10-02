from __future__ import annotations

import argparse

from automis.datasets import MSDDataset

if __name__  == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset_name", type=str, required=True)
    args = argparser.parse_args()

    dataset = MSDDataset(name=args.dataset_name)
    dataset.download_and_extract()
