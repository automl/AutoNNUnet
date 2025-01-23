from __future__ import annotations

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
        datasets=ALL_DATASETS,
        format="pdf",
    )
    # plotter.create_emissions_table()
    plotter.plot_baselines()
    exit()
    # plotter.compare_msd_results()
    # plotter.plot_msd_overview()

    # plotter.plot_ablation_paths(budget_idx=-1)
    # plotter.plot_pdp(
    #     dataset="Dataset004_Hippocampus",
    #     approach_key="hpo",
    #     hp_name_1="aug_factor",
    #     hp_name_2="weight_decay",
    # )
    # plotter.plot_pdp(
    #     dataset="Dataset004_Hippocampus",
    #     approach_key="hpo",
    #     hp_name_1="aug_factor",
    # )
    # plotter._plot_optimization(
    #     dataset="Dataset005_Prostate",
    # )
    # plotter._plot_nas_budgets(dataset="Dataset004_Hippocampus", approach_key="hpo_nas")
    # plotter._plot_nas_origins(dataset="Dataset004_Hippocampus", approach_key="hpo_nas")
    # plotter.plot_nas_origins(approach_key="hpo_nas")
    # plotter.plot_nas_budgets(approach_key="hpo_nas")
    exit()
    # plotter.plot_hpo_combined(include_nas=True, include_hnas=True, show_error=False)
    # plotter.plot_footprints()
    # plotter.plot_ablation_paths()
    # plotter.plot_hpis()
    # plotter.plot_nas(show_configs=False)
    # plotter._plot_nas2_("Dataset004_Hippocampus", show_configs=False)
    # plotter.plot_hpo_combined(show_error=False)
    plotter.plot_hpo_combined(include_nas=True, include_hnas=True, show_error=False)
    plotter.plot_nas_combined()
    # plotter.plot_hyperparameter_importances(budget="combined")
    # plotter.plot_hpo()
    plotter.plot_nas(approach_key="hpo_hnas", show_configs=True)
    plotter.plot_nas(approach_key="hpo_hnas", show_configs=False)
    # plotter.plot_hpo(include_nas=True)
    # plotter.plot_hpo(include_nas=False)
    # plotter.compute_emissions() 
    # plotter.plot_baseline_runtimes()
    # plotter.plot_baseline_performances_and_runtimes()