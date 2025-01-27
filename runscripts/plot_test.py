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

    plotter.plot_nas_combined()             # NAS + HNAS Pareto front