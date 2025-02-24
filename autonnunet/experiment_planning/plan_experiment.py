"""Functions to plan an experiment based on a given configuration."""
from __future__ import annotations

from typing import TYPE_CHECKING

from autonnunet.experiment_planning.auto_experiment_planners import (
        AutoExperimentPlanner,
        nnUNetPlannerResEncL,
        nnUNetPlannerResEncM,
        nnUNetPlannerResEncXL,
)

if TYPE_CHECKING:
        from omegaconf import DictConfig

PLANNER_CLASSES = {
    "ConvolutionalEncoder": AutoExperimentPlanner,
    "ResidualEncoderM": nnUNetPlannerResEncM,
    "ResidualEncoderL": nnUNetPlannerResEncL,
    "ResidualEncoderXL": nnUNetPlannerResEncXL
}


def plan_experiment(
        dataset_name: str,
        plans_name: str,
        hp_config: DictConfig
    ) -> dict:
    """Plans an experiment based on the given configuration.

    Parameters
    ----------
    dataset_name : str
        The dataset name.

    plans_name : str
        The plans name.

    hp_config : DictConfig
        The hyperparameter configuration.

    Returns:
    -------
    dict
        The experiment plan.
    """
    if hp_config.encoder_type in PLANNER_CLASSES:
        planner_cls = PLANNER_CLASSES[hp_config.encoder_type]
    else:
        raise ValueError(f"Invalid encoder type: {hp_config.encoder_type}")

    planner = planner_cls(
            hp_config=hp_config,
            dataset_name_or_id=dataset_name,
            plans_name=plans_name
        )

    return planner.plan_experiment()