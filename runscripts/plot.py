from __future__ import annotations

import os
import pathlib

import argparse
import logging

from autonnunet.utils import Plotter
from autonnunet.datasets import ALL_DATASETS

if __name__  == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--configuration", type=str, default="3d_fullres")
    args = argparser.parse_args()

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    plotter = Plotter(
        datasets=ALL_DATASETS,
        configuration=args.configuration
    )
    plotter.load_data() 
    # plotter.plot_baselines()
    # plotter.plot_nas()
    # plotter.plot_hpo()
    # plotter.plot_hyperband()
    # plotter.compute_emissions() 
    plotter.plot_baseline_runtimes()