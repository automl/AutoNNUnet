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

    # plotter._plot_baseline(dataset="Dataset001_BrainTumour", x_metric="Real Runtime Used")
    plotter._plot_baseline_vs_dataset_metric(dataset_metric="Test")            