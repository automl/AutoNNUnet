"""The AutoNNUNet predictor."""
from __future__ import annotations

import os

import torch
from batchgenerators.utilities.file_and_folder_operations import (join,
                                                                  load_json)
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.label_handling.label_handling import \
    determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from torch._dynamo import OptimizedModule

from autonnunet.training.auto_nnunet_trainer import AutoNNUNetTrainer


class AutoNNUNetPredictor(nnUNetPredictor):
    """Predictor for the AutoNNUNet model."""
    def __init__(
            self,
            tile_step_size: float = 0.5,
            use_gaussian: bool = True,                  # noqa: FBT001, FBT002
            use_mirroring: bool = True,                 # noqa: FBT001, FBT002
            perform_everything_on_device: bool = True,  # noqa: FBT001, FBT002
            device: torch.device | None = None,
            verbose: bool = False,                      # noqa: FBT001, FBT002
            verbose_preprocessing: bool = False,        # noqa: FBT001, FBT002
            allow_tqdm: bool = True                     # noqa: FBT001, FBT002
        ):
        """Initializes the AutoNNUNetPredictor.

        Parameters
        ----------
        tile_step_size : float, optional
            The step size for tiling. Defaults to 0.5.

        use_gaussian : bool, optional
            Whether to use Gaussian smoothing. Defaults to True.

        use_mirroring : bool, optional
            Whether to use mirroring. Defaults to True.

        perform_everything_on_device : bool, optional
            Whether to perform everything on the device. Defaults to True.

        device : torch.device, optional
            The device to use. Defaults to None.
            In this case, the device will be set to "cuda".

        verbose : bool, optional
            Whether to print logging information. Defaults to False.

        verbose_preprocessing : bool, optional
            Whether to print preprocessing information. Defaults to False.

        allow_tqdm : bool, optional
            Whether to use tqdm. Defaults to True.
        """
        if device is None:
            device = torch.device("cuda")
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
            model_training_output_dir: str,
            use_folds: tuple[int],
            checkpoint_name: str,
            trainer: AutoNNUNetTrainer | None = None
        ) -> None:
        """Initializes the predictor from a given configuration.

        Parameters
        ----------
        model_training_output_dir : str
            The directory containing the model training output.

        use_folds : tuple[int]
            The folds to use for the prediction.

        checkpoint_name : str
            The name of the checkpoint file.

        trainer : AutoNNUNetTrainer, optional
            The trainer to use for the prediction. Defaults to None.
            If the trainer is available, it will be used to initialize the predictor.
        """
        parameters = []
        for i, f in enumerate(use_folds):
            f = int(f) if f != "all" else f     # noqa: PLW2901
            checkpoint = torch.load(
                    join(
                    model_training_output_dir,
                    f"fold_{f}",
                    checkpoint_name
                ),
                map_location=torch.device("cpu")
            )
            if i == 0:
                trainer_name = checkpoint["trainer_name"]
                configuration_name = checkpoint["init_args"]["configuration"]
                inference_allowed_mirroring_axes = checkpoint.get(
                    "inference_allowed_mirroring_axes",
                    None
                )

            parameters.append(checkpoint["network_weights"])

        if trainer is None:
            dataset_json = load_json(
                    join(model_training_output_dir, "fold_0", "dataset.json"))
            plans = load_json(
                    join(model_training_output_dir, "fold_0", "plans.json"))
            plans_manager = PlansManager(plans)
            configuration_manager = plans_manager.get_configuration(
                    configuration_name
                )
            num_input_channels = determine_num_input_channels(
                plans_manager,
                configuration_manager,
                dataset_json
            )

            network = AutoNNUNetTrainer.build_network_architecture(
                configuration_manager.network_arch_class_name,
                configuration_manager.network_arch_init_kwargs,
                configuration_manager.network_arch_init_kwargs_req_import,
                num_input_channels,
                plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
                enable_deep_supervision=False
            )
        else:
            trainer.initialize()

            plans_manager = trainer.plans_manager
            configuration_manager = trainer.configuration_manager
            dataset_json = trainer.dataset_json
            network = trainer.network

        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)

        do_compile = ("nnUNet_compile" in os.environ) \
            and (os.environ["nnUNet_compile"].lower() in ("true", "1", "t"))    # noqa: SIM112

        if do_compile and not isinstance(self.network, OptimizedModule):
            print("Using torch.compile")
            self.network = torch.compile(self.network)