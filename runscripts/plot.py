from __future__ import annotations

import os
import pathlib

import argparse
import logging

from autonnunet.analysis.plotter import Plotter
from autonnunet.datasets import ALL_DATASETS

if __name__  == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--configuration", type=str, default="3d_fullres")
    args = argparser.parse_args()

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    plotter = Plotter(
        # datasets=[ALL_DATASETS[3]],
        datasets=ALL_DATASETS,
        configuration=args.configuration
    )
    # plotter.compare_msd_results()
    # plotter.plot_msd_overview()

    plotter.load_data() 
    # plotter.plot_baselines()
    # plotter.plot_nas(show_configs=False)
    # plotter._plot_nas2_("Dataset004_Hippocampus", show_configs=False)
    # plotter.plot_hpo_combined(show_error=False)
    # plotter.plot_hpo_combined(include_nas=True, show_error=False)
    # plotter.plot_nas_combined()
    # plotter.plot_hyperparameter_importances(budget="combined")
    # plotter.plot_hpo()
    plotter.plot_nas(show_configs=True)
    plotter.plot_nas(show_configs=False)
    # plotter.plot_hpo(include_nas=True)
    # plotter.plot_hpo(include_nas=False)
    # plotter.compute_emissions() 
    # plotter.plot_baseline_runtimes()
    # plotter.plot_baseline_performances_and_runtimes()