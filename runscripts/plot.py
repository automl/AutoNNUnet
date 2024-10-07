from __future__ import annotations

import argparse
import logging

from autonnunet.utils import Plotter

if __name__  == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--configuration", type=str, default="3d_fullres")
    args = argparser.parse_args()

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    plotter = Plotter(configuration=args.configuration)
    plotter.load_data()
    # plotter.plot_baseline(datasets=["Dataset001_BrainTumour"])
    plotter.plot_hpo()
    # plotter.create_table(datasets=["Dataset001_BrainTumour", "Dataset002_Heart", "Dataset003_Liver"])