from __future__ import annotations

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
    plotter.plot_hpo(x_log_scale=False, y_log_scale=False)
    plotter.plot_hyperband()
    # plotter.compute_emissions() 
    # plotter.plot_baselines()