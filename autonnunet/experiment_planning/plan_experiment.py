from __future__ import annotations

from typing import TYPE_CHECKING

from nnunetv2.utilities.json_export import recursive_fix_for_json_export

from autonnunet.experiment_planning.auto_experiment_planners import (
    AutoExperimentPlanner, nnUNetPlannerResEncL, nnUNetPlannerResEncM,
    nnUNetPlannerResEncXL)

if TYPE_CHECKING:
        from omegaconf import DictConfig


def plan_experiment(
        dataset_name: str,
        plans_name: str,
        hp_config: DictConfig
    ) -> dict:
        if hp_config.encoder_type == "ConvolutionalEncoder":
            planner = AutoExperimentPlanner(
                dataset_name_or_id=dataset_name,
                plans_name=plans_name
            )

        elif hp_config.encoder_type == "ResidualEncoderM":
            planner = nnUNetPlannerResEncM(
                dataset_name_or_id=dataset_name,
                plans_name=plans_name
            )

        elif hp_config.encoder_type == "ResidualEncoderL":
            planner = nnUNetPlannerResEncL(
                dataset_name_or_id=dataset_name,
                plans_name=plans_name
            )

        elif hp_config.encoder_type == "ResidualEncoderXL":
            planner = nnUNetPlannerResEncXL(
                dataset_name_or_id=dataset_name,
                plans_name=plans_name
            )
        else:
            raise ValueError(f"Invalid encoder type: {hp_config.encoder_type}")

        planner.UNet_base_num_features = hp_config.base_num_features
        planner.UNet_max_features_3d = hp_config.max_features

        # planner.UNet_blocks_per_stage_encoder = (
        #     hp_config.encoder_blocks_stage1,
        #     hp_config.encoder_blocks_stage2,
        #     hp_config.encoder_blocks_stage3,
        #     *[hp_config.encoder_blocks_remaining_stages] * 11)
        # planner.UNet_blocks_per_stage_encoder = (
        #     hp_config.decoder_blocks_stage1,
        #     hp_config.decoder_blocks_stage2,
        #     hp_config.decoder_blocks_stage3,
        #     *[hp_config.decoder_blocks_remaining_stages] * 10)

        plans = planner.plan_experiment()

        # We have to convert everything to pure python data types
        recursive_fix_for_json_export(plans)

        return plans