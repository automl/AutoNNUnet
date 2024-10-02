from __future__ import annotations

import argparse

from automis.utils import Plotter

if __name__  == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset_name", type=str, required=True)
    argparser.add_argument("--configuration", type=str, default="3d_fullres")
    args = argparser.parse_args()

    plotter = Plotter(configuration=args.configuration)
    plotter.plot_baseline(datasets=["Dataset001_BrainTumour"])