from __future__ import annotations

from autonnunet.analysis import Plotter

if __name__  == "__main__":
    plotter = Plotter(
        file_format="pdf",
    )

    # --------------------------------------------------------------------------------------------
    # Tables
    # --------------------------------------------------------------------------------------------
    # plotter.load_all_data()
    # plotter._create_dataset_dsc_table(dataset="Dataset001_BrainTumour")
    # plotter.create_runtime_table()
    # plotter.create_dataset_dsc_tables()

    # --------------------------------------------------------------------------------------------
    # Baselines
    # --------------------------------------------------------------------------------------------
    # plotter.plot_baselines()        # Baselines performance over time

    # --------------------------------------------------------------------------------------------
    # Overview / Comparison
    # --------------------------------------------------------------------------------------------
    # plotter.plot_optimization_combined()    # HPO + NAS + HNAS over time

    # --------------------------------------------------------------------------------------------
    # Dataset Analysis
    # --------------------------------------------------------------------------------------------
    # plotter.plot_joint_dataset_features_heatmap()

    # --------------------------------------------------------------------------------------------
    # Qualitative Analysis
    # --------------------------------------------------------------------------------------------
    # plotter.plot_qualitative_segmentations()

    # --------------------------------------------------------------------------------------------
    # HPO
    # --------------------------------------------------------------------------------------------
    # plotter.plot_optimization()

    # plotter.plot_footprints(approach_keys=["hpo"])
    # plotter.plot_budget_correlations(approach_keys=["hpo"])
    plotter.plot_hpis(approach_keys=["hpo"], plot_pdps=True)
    # plotter.plot_ablation_paths(approach_keys=["hpo"])

    # --------------------------------------------------------------------------------------------
    # HPO + NAS
    # --------------------------------------------------------------------------------------------
    # plotter.plot_nas_combined()
    # plotter.plot_nas_budgets(approach_key="hpo_nas")
    # plotter.plot_nas_origins(approach_key="hpo_nas")

    # Analysis
    # plotter.plot_footprints(approach_keys=["hpo_nas"])
    # plotter.plot_budget_correlations(approach_keys=["hpo_nas"])
    plotter.plot_hpis(approach_keys=["hpo_nas"], plot_pdps=True)
    # plotter.plot_ablation_paths(approach_keys=["hpo_nas"])

    # --------------------------------------------------------------------------------------------
    # HPO + HNAS
    # --------------------------------------------------------------------------------------------
    # Results
    # plotter.plot_nas_budgets(approach_key="hpo_hnas")
    # plotter.plot_nas_origins(approach_key="hpo_hnas")

    # Analysis
    # plotter.plot_footprints(approach_keys=["hpo_hnas"])
    # plotter.plot_budget_correlations(approach_keys=["hpo_hnas"])
    plotter.plot_hpis(approach_keys=["hpo_hnas"], plot_pdps=True)
    # plotter.plot_ablation_paths(approach_keys=["hpo_hnas"])