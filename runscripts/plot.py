from __future__ import annotations

from autonnunet.analysis import Plotter

if __name__  == "__main__":
    plotter = Plotter(
        file_format="pdf",
    )

    # --------------------------------------------------------------------------------------------
    # Tables
    # --------------------------------------------------------------------------------------------
    plotter.load_all_data()
    plotter.create_runtime_table()
    plotter.create_val_dsc_table()
    plotter.create_test_dsc_table()
    plotter.create_dataset_dsc_tables()

    # --------------------------------------------------------------------------------------------
    # Baselines
    # --------------------------------------------------------------------------------------------
    plotter.plot_baselines()        # Baselines performance over time

    # --------------------------------------------------------------------------------------------
    # Overview / Comparison
    # --------------------------------------------------------------------------------------------
    plotter.plot_optimization_combined()    # HPO + NAS + HNAS over time
    plotter.plot_cross_eval_matrix()    

    # --------------------------------------------------------------------------------------------
    # Dataset Analysis
    # --------------------------------------------------------------------------------------------
    plotter.plot_joint_dataset_features_heatmap(
        include="none",
        orientation="left"
    )
    plotter.plot_joint_dataset_features_heatmap(
        include="none",
        orientation="right"
    )
    plotter.plot_joint_dataset_features_heatmap(
        include="incumbents",
        orientation="left"
    )
    plotter.plot_joint_dataset_features_heatmap(
        include="incumbents",
        orientation="right"
    )
    plotter.plot_joint_dataset_features_heatmap(
        include="importances",
        orientation="left"
    )
    plotter.plot_joint_dataset_features_heatmap(
        include="importances",
        orientation="right"
    )

    plotter.create_top_dataset_features_hps_table(
        include="incumbents",
        plot_relationships=True,
        corr_threshold=0.7,
    )
    plotter.create_top_dataset_features_hps_table(
        include="importances",
        plot_relationships=True,
        corr_threshold=0.7
    )
    plotter.plot_joint_dataset_features_combined(
        dataset_feature_1="Volume",
        dataset_feature_2="Class Volume Ratio",
        hp_name="Weight Decay",
        include="incumbents"
    )
    plotter.plot_joint_dataset_features_combined(
        dataset_feature_1="Intensity Min.",
        dataset_feature_2="Volume",
        hp_name="Initial LR",
        include="importances"
    )

    # --------------------------------------------------------------------------------------------
    # Qualitative Analysis
    # --------------------------------------------------------------------------------------------
    plotter.plot_qualitative_segmentations(
        case_where_autonnunet="best"
    )
    plotter.plot_qualitative_segmentations(
        case_where_autonnunet="worst"
    )

    # --------------------------------------------------------------------------------------------
    # HPO
    # --------------------------------------------------------------------------------------------
    plotter.plot_optimization()

    plotter.plot_footprints(approach_keys=["hpo"])
    plotter.plot_budget_correlations(approach_keys=["hpo"])
    plotter.plot_hpis(approach_keys=["hpo"], plot_pdps=True)
    plotter.plot_ablation_paths(approach_keys=["hpo"])
    plotter.compute_all_hp_interaction_tables(approach_key="hpo")

    plotter.plot_pdp(
        dataset="Dataset001_BrainTumour",
        approach_key="hpo",
        hp_name_1="Initial LR",
        hp_name_2="Momentum (SGD)",
    )
    plotter.plot_pdp(
        dataset="Dataset003_Liver",
        approach_key="hpo",
        hp_name_1="Initial LR",
        hp_name_2="Foreground Oversamp.",
    )

    # --------------------------------------------------------------------------------------------
    # HPO + NAS
    # --------------------------------------------------------------------------------------------
    plotter.plot_nas_combined()
    plotter.plot_nas_budgets(approach_key="hpo_nas")
    plotter.plot_nas_origins(approach_key="hpo_nas")

    # Analysis
    plotter.plot_footprints(approach_keys=["hpo_nas"])
    plotter.plot_budget_correlations(approach_keys=["hpo_nas"])
    plotter.plot_hpis(approach_keys=["hpo_nas"], plot_pdps=True)
    plotter.plot_ablation_paths(approach_keys=["hpo_nas"])
    plotter.compute_all_hp_interaction_tables(approach_key="hpo_nas")

    # --------------------------------------------------------------------------------------------
    # HPO + HNAS
    # --------------------------------------------------------------------------------------------
    # Results
    plotter.plot_nas_budgets(approach_key="hpo_hnas")
    plotter.plot_nas_origins(approach_key="hpo_hnas")

    # Analysis
    plotter.plot_footprints(approach_keys=["hpo_hnas"])
    plotter.plot_budget_correlations(approach_keys=["hpo_hnas"])
    plotter.plot_hpis(approach_keys=["hpo_hnas"], plot_pdps=True)
    plotter.plot_ablation_paths(approach_keys=["hpo_hnas"])
    plotter.compute_all_hp_interaction_tables(approach_key="hpo_hnas")