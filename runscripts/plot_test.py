from __future__ import annotations

import logging

from autonnunet.analysis.plotter import Plotter
from autonnunet.datasets import ALL_DATASETS

if __name__  == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    plotter = Plotter(
        datasets=ALL_DATASETS,
        format="png",
    )
    plotter.create_result_table()
    plotter.create_dataset_result_table(dataset="Dataset005_Prostate")
    # plotter._plot_baseline(dataset="Dataset002_Heart")
    # plotter._plot_baseline(dataset="Dataset004_Hippocampus")
    # plotter.plot_baselines(x_metric="Real Runtime Used")    
    # plotter.plot_joint_dataset_features_heatmap()
    # plotter.plot_joint_dataset_features(feature_x="Intensity Mean", feature_y="DSC")