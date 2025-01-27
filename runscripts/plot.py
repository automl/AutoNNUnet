from __future__ import annotations

import logging

from autonnunet.analysis.plotter import Plotter
from autonnunet.datasets import ALL_DATASETS

if __name__  == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    plotter = Plotter(
        datasets=ALL_DATASETS,
        format="pdf",
    )

    # --------------------------------------------------------------------------------------------
    # Baselines
    # --------------------------------------------------------------------------------------------
    # TODO add MedSam-2
    plotter.plot_baselines()        # Baselines performance over time

    # --------------------------------------------------------------------------------------------
    # Overview / Comparison
    # --------------------------------------------------------------------------------------------
    # plotter.plot_optimization_combined()    # HPO + NAS + HNAS over time
    # plotter.plot_nas_combined()             # NAS + HNAS Pareto front

    # --------------------------------------------------------------------------------------------
    # Overall Analysis
    # --------------------------------------------------------------------------------------------
    # TODO

    # --------------------------------------------------------------------------------------------
    # HPO 
    # --------------------------------------------------------------------------------------------
    # Results
    plotter.plot_optimization()             # Single datasets HPO over time

    # Analysis
    plotter.plot_footprints(approach_keys=["hpo"])    # Footprints of HPO
    plotter.plot_budget_correlations(approach_keys=["hpo"])  # Budget Correlations of HPO
    plotter.plot_hpis(approach_keys=["hpo"])          # HPIS of HPO
    plotter.plot_ablation_paths(approach_keys=["hpo"])  # Ablation Paths of HPO

    # --------------------------------------------------------------------------------------------
    # HPO + NAS
    # --------------------------------------------------------------------------------------------
    # Results
    plotter.plot_nas_budgets(approach_key="hpo_nas")
    plotter.plot_nas_origins(approach_key="hpo_nas")

    # Analysis
    plotter.plot_footprints(approach_keys=["hpo_nas"])    # Footprints of HPO
    plotter.plot_budget_correlations(approach_keys=["hpo_nas"])  # Budget Correlations of HPO
    plotter.plot_hpis(approach_keys=["hpo_nas"])          # HPIS of HPO
    plotter.plot_ablation_paths(approach_keys=["hpo_nas"])  # Ablation Paths of HPO

    # --------------------------------------------------------------------------------------------
    # HPO + HNAS
    # --------------------------------------------------------------------------------------------
    # # Results
    # plotter.plot_nas_budgets(approach_key="hpo_hnas")
    # plotter.plot_nas_origins(approach_key="hpo_hnas")

    # # Analysis
    # plotter.plot_footprints(approach_keys=["hpo_hnas"])    # Footprints of HPO
    # plotter.plot_budget_correlations(approach_keys=["hpo_hnas"])  # Budget Correlations of HPO
    # plotter.plot_hpis(approach_keys=["hpo_hnas"])          # HPIS of HPO
    # plotter.plot_ablation_paths(approach_keys=["hpo_hnas"])  # Ablation Paths of HPO