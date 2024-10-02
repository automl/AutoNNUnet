from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from dynamic_network_architectures.building_blocks.helper import get_matching_batchnorm
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.label_handling.label_handling import (
    determine_num_input_channels,
)
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from torch._dynamo import OptimizedModule

from autonnunet.training.auto_nnunet_trainer import AutoNNUNetTrainer

if TYPE_CHECKING:
    from omegaconf import DictConfig


class CustomNNUNetPredictor(nnUNetPredictor):
    def __init__(self,
                 tile_step_size: float = 0.5,
                 use_gaussian: bool = True,
                 use_mirroring: bool = True,
                 perform_everything_on_device: bool = True,
                 device: torch.device = torch.device("cuda"),
                 verbose: bool = False,
                 verbose_preprocessing: bool = False,
                 allow_tqdm: bool = True):
        super().__init__(
            tile_step_size=tile_step_size,
            use_gaussian=use_gaussian,
            use_mirroring=use_mirroring,
            perform_everything_on_device=perform_everything_on_device,
            device=device,
            verbose=verbose,
            verbose_preprocessing=verbose_preprocessing,
            allow_tqdm=allow_tqdm
        )

    def initialize_from_config(
            self,
            hp_config: DictConfig,
            model_training_output_dir: str,
            use_folds: tuple[int],
            checkpoint_name: str = "checkpoint_final.pth"
        ):
            dataset_json = load_json(join(model_training_output_dir, "fold_0", "dataset.json"))
            plans = load_json(join(model_training_output_dir, "fold_0", "plans.json"))
            plans_manager = PlansManager(plans)

            parameters = []
            for i, f in enumerate(use_folds):
                f = int(f) if f != "all" else f
                checkpoint = torch.load(join(model_training_output_dir, f"fold_{f}", checkpoint_name),
                                        map_location=torch.device("cpu"))
                if i == 0:
                    trainer_name = checkpoint["trainer_name"]
                    configuration_name = checkpoint["init_args"]["configuration"]
                    inference_allowed_mirroring_axes = checkpoint.get("inference_allowed_mirroring_axes", None)

                parameters.append(checkpoint["network_weights"])

            configuration_manager = plans_manager.get_configuration(configuration_name)

            # Update hyperparameter configuration stored in configuration manager
            configuration_manager.configuration["architecture"]["arch_kwargs"]["n_conv_per_stage"] = [hp_config.n_conv_per_stage] * len(configuration_manager.configuration["architecture"]["arch_kwargs"]["n_conv_per_stage"])
            configuration_manager.configuration["architecture"]["arch_kwargs"]["n_conv_per_stage_decoder"] = [hp_config.n_conv_per_stage_decoder] * len(configuration_manager.configuration["architecture"]["arch_kwargs"]["n_conv_per_stage_decoder"])

            # the following part is taken from https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/nnUNetTrainer/variants/network_architecture/nnUNetTrainerBN.py
            if hp_config.normalization == "BatchNorm":
                from pydoc import locate
                conv_op = locate(configuration_manager.configuration["architecture"]["arch_kwargs"]["conv_op"])
                bn_class = get_matching_batchnorm(conv_op)
                configuration_manager.configuration["architecture"]["arch_kwargs"]["norm_op"] = bn_class.__module__ + "." + bn_class.__name__
                configuration_manager.configuration["architecture"]["arch_kwargs"]["norm_op_kwargs"] = {"eps": 1e-5, "affine": True}

            num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)

            network = AutoNNUNetTrainer.build_network_architecture(
                configuration_manager.network_arch_class_name,
                configuration_manager.network_arch_init_kwargs,
                configuration_manager.network_arch_init_kwargs_req_import,
                num_input_channels,
                plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
                enable_deep_supervision=False
            )

            self.plans_manager = plans_manager
            self.configuration_manager = configuration_manager
            self.list_of_parameters = parameters
            self.network = network
            self.dataset_json = dataset_json
            self.trainer_name = trainer_name
            self.allowed_mirroring_axes = inference_allowed_mirroring_axes
            self.label_manager = plans_manager.get_label_manager(dataset_json)
            if ("nnUNet_compile" in os.environ) and (os.environ["NNUNET_COMPILE"].lower() in ("true", "1", "t")) \
                    and not isinstance(self.network, OptimizedModule):
                print("Using torch.compile")
                self.network = torch.compile(self.network)

