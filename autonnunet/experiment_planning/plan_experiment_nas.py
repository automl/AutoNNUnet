from __future__ import annotations

from typing import TYPE_CHECKING

from autonnunet.experiment_planning.auto_experiment_planners_nas import (
    AutoExperimentPlanner, nnUNetPlannerResEncL, nnUNetPlannerResEncM,
    nnUNetPlannerResEncXL)

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