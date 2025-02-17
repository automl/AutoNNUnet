"""The AutoNNUNetTrainer class."""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.color_transforms import (
    BrightnessMultiplicativeTransform,
    ContrastAugmentationTransform,
    GammaTransform,
)
from batchgenerators.transforms.noise_transforms import (
    GaussianBlurTransform,
    GaussianNoiseTransform,
)
from batchgenerators.transforms.resample_transforms import (
    SimulateLowResolutionTransform,
)
from batchgenerators.transforms.spatial_transforms import (
    MirrorTransform,
    SpatialTransform,
)
from batchgenerators.transforms.utility_transforms import (
    NumpyToTensor,
    RemoveLabelTransform,
    RenameTransform,
)
from batchgenerators.utilities.file_and_folder_operations import load_json
from nnunetv2.training.data_augmentation\
    .custom_transforms.cascade_transforms import (
    ApplyRandomBinaryOperatorTransform,
    MoveSegAsOneHotToData,
    RemoveRandomConnectedComponentFromOneHotEncodingTransform,
)
from nnunetv2.training.data_augmentation\
    .custom_transforms.deep_supervision_donwsampling import (
    DownsampleSegForDSTransform2,
)
from nnunetv2.training.data_augmentation\
    .custom_transforms.masking import MaskTransform
from nnunetv2.training.data_augmentation\
    .custom_transforms.region_based_training import (
    ConvertSegmentationToRegionsTransform,
)
from nnunetv2.training.data_augmentation\
    .custom_transforms.transforms_for_dummy_2d import (
    Convert2DTo3DTransform,
    Convert3DTo2DTransform,
)
from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import softmax_helper_dim1
from nnunetv2.utilities.label_handling.label_handling import (
    determine_num_input_channels,
)
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: N817
from torch.optim.lr_scheduler import CosineAnnealingLR

from autonnunet.experiment_planning import plan_experiment
from autonnunet.hnas.cfg_unet import CFGUNet
from autonnunet.training.auto_nnunet_logger import AutoNNUNetLogger
from autonnunet.training.dummy_lr_scheduler import DummyLRScheduler
from autonnunet.utils.paths import NNUNET_PREPROCESSED

if TYPE_CHECKING:
    from omegaconf import DictConfig


class AutoNNUNetTrainer(nnUNetTrainer):
    """The AutoNNUNetTrainer class.

    Parameters:
    ----------
    plans : dict
        The nnU-Net plans.

    configuration : str
        The name of the configuration to use.

    fold : int
        The fold number.

    dataset_json : dict
        The dataset JSON.

    unpack_dataset : bool, optional
        Whether to unpack the dataset. Default is True.

    device : torch.device, optional
        The device to use. Default is None. In this case, the device is set to "cuda".
    """
    def __init__(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            unpack_dataset: bool = True,        # noqa: FBT001, FBT002
            device: torch.device | None = None,
        ):
        """Initializes the AutoNNUNetTrainer."""
        if device is None:
            device = torch.device("cuda")
        super().__init__(
            plans,
            configuration,
            fold,
            dataset_json,
            unpack_dataset,
            device
        )

        self.output_folder_base = "."
        self.output_folder = "."

        assert not self.is_cascaded, "Cascaded training is not supported."

        self.log_file = os.path.join(self.output_folder, "training_log.txt")    # noqa: PTH118
        self.logger = AutoNNUNetLogger()

    def initialize(self) -> None:
        """Initializes the trainer.
        This needs to be adapted to generate the CFGUNet if required.
        """
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(
                self.plans_manager,
                self.configuration_manager,
                self.dataset_json
            )

            if "architecture" in self.hp_config:
                self.network = self.build_cfg_unet_architecture().to(self.device)
                logging.getLogger("Trainer").info(
                    f"Using architecture: {self.network.__class__.__name__}")
            else:
                self.network = self.build_network_architecture(
                    self.configuration_manager.network_arch_class_name,
                    self.configuration_manager.network_arch_init_kwargs,
                    self.configuration_manager.network_arch_init_kwargs_req_import,
                    self.num_input_channels,
                    self.label_manager.num_segmentation_heads,
                    self.enable_deep_supervision
                ).to(self.device)
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file("Using torch.compile...")
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                    self.network
                )
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer"
                               " was already initialized. That should not happen.")

    def build_cfg_unet_architecture(self) -> nn.Module:
        """Builds the architecture based on the internal hyperparameters.

        Returns:
        -------
        nn.Module
            The architecture.
        """
        string_tree = str(self.hp_config.architecture).replace("'", "")
        return CFGUNet(
            hp_config=self.hp_config,
            string_tree=string_tree,
            arch_init_kwargs=self.configuration_manager.network_arch_init_kwargs,
            arch_kwargs_req_import=self.configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels=self.num_input_channels,
            num_output_channels=self.label_manager.num_segmentation_heads,
            enable_deep_supervision=self.enable_deep_supervision
        )

    @staticmethod
    def from_config(cfg: DictConfig) -> AutoNNUNetTrainer:
        """Initializes a AutoNNUNetTrainer from a hydra configuration.

        Parameters:
        ----------
        cfg : DictConfig
            The configuration.

        Returns:
        -------
        AutoNNUNetTrainer
            The AutoNNUNetTrainer.
        """
        preprocessed_dataset_folder_base = NNUNET_PREPROCESSED / cfg.dataset.name
        dataset_json = load_json(preprocessed_dataset_folder_base / "dataset.json")

        if "architecture" in cfg.hp_config:
            if "conv_encoder" in cfg.hp_config.architecture:
                cfg.hp_config.encoder_type = "ConvolutionalEncoder"
            else:
                cfg.hp_config.encoder_type = "ResidualEncoderM"

        if "encoder_type" in cfg.hp_config:
            plans = plan_experiment(
                dataset_name=cfg.dataset.name,
                plans_name=cfg.trainer.plans_identifier,
                hp_config=cfg.hp_config,
            )
        else:
            plans = load_json(
                preprocessed_dataset_folder_base /\
                    f"{cfg.trainer.plans_identifier}.json"
            )

        nnunet_trainer = AutoNNUNetTrainer(
            plans=plans,
            configuration=cfg.trainer.configuration,
            fold=cfg.fold,
            dataset_json=dataset_json,
            unpack_dataset=not cfg.trainer.use_compressed_data,
            device=torch.device(cfg.device),
        )
        nnunet_trainer.set_hp_config(cfg.hp_config)

        nnunet_trainer.disable_checkpointing = cfg.trainer.disable_checkpointing

        if cfg.load:
            load_path_best = Path(cfg.load + "_best.pth").resolve()
            load_path_final = Path(cfg.load + "_final.pth").resolve()

            if load_path_final.exists():
                # We copy the best checkpoint to the current directory since the
                # best epoch ever might be in the past and not overriden
                checkpoint_best_path = Path().resolve() / "checkpoint_best.pth"
                shutil.copyfile(load_path_best, checkpoint_best_path)

                nnunet_trainer.load_checkpoint(str(load_path_final))

        # Even if we continue another training run in succesive halving, we want to load
        # the latest checkpoint in the current directory as this is based on the
        # previous run which has been initially loaded from the checkpoint already
        if cfg.pipeline.continue_training and Path("./checkpoint_latest.pth").exists():
            nnunet_trainer.load_checkpoint("checkpoint_latest.pth")

        return nnunet_trainer

    def get_training_transforms(                # noqa: PLR0913
            self,
            patch_size: np.ndarray | tuple[int],
            rotation_for_DA: dict,              # noqa: N803
            deep_supervision_scales: list | tuple | None,
            mirror_axes: tuple[int, ...],
            do_dummy_2d_data_aug: bool,         # noqa: FBT001
            order_resampling_data: int = 3,
            order_resampling_seg: int = 1,
            border_val_seg: int = -1,
            use_mask_for_norm: list[bool] | None= None,
            is_cascaded: bool = False,          # noqa: FBT002, FBT001
            foreground_labels: tuple[int, ...] | list[int] | None = None,
            regions: list[list[int] | tuple[int, ...] | int] | None = None,
            ignore_label: int | None = None,
    ) -> AbstractTransform:
        """Returns the training transforms based on the internal hyperparameters.
        Here, we adapt the data augmentation using the nnU-Net data augmentation
        pipeline.

        Parameters:
        ----------
        patch_size : np.ndarray | tuple[int]
            The patch size.

        rotation_for_DA : dict
            The rotation angles for data augmentation.

        deep_supervision_scales : list | tuple | None
            The deep supervision scales.

        mirror_axes : tuple[int, ...]
            The mirror axes.

        do_dummy_2d_data_aug : bool
            Whether to perform dummy 2D data augmentation.

        order_resampling_data : int, optional
            The order of resampling for data. Default is 3.

        order_resampling_seg : int, optional
            The order of resampling for segmentation. Default is 1.

        border_val_seg : int, optional
            The border value for segmentation. Default is -1.

        use_mask_for_norm : list[bool] | None, optional
            Whether to use mask for normalization. Default is None.

        is_cascaded : bool, optional
            Whether the training is cascaded. Default is False.

        foreground_labels : tuple[int, ...] | list[int] | None, optional
            The foreground labels. Default is None.

        regions : list[list[int] | tuple[int, ...] | int] | None, optional
            The regions. Default is None.

        ignore_label : int | None, optional
            The ignore label. Default is None.

        Returns:
        -------
        AbstractTransform
            The training transforms.
        """
        tr_transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            tr_transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None

        tr_transforms.append(
            SpatialTransform(
                patch_size_spatial,
                patch_center_dist_from_border=None,
                do_elastic_deform=False,
                alpha=(0, 0),
                sigma=(0, 0),
                do_rotation=True,
                angle_x=rotation_for_DA["x"],
                angle_y=rotation_for_DA["y"],
                angle_z=rotation_for_DA["z"],
                p_rot_per_axis=1,
                do_scale=True,
                scale=(0.7, 1.4),
                border_mode_data="constant",
                border_cval_data=0,
                order_data=order_resampling_data,
                border_mode_seg="constant",
                border_cval_seg=border_val_seg,
                order_seg=order_resampling_seg,
                random_crop=False,
                p_el_per_sample=0,
                p_scale_per_sample=self.hp_config.aug_scale_prob *\
                    self.hp_config.aug_factor,
                p_rot_per_sample=self.hp_config.aug_rotate_prob *\
                    self.hp_config.aug_factor,
                independent_scale_for_each_axis=False
            )
        )

        if do_dummy_2d_data_aug:
            tr_transforms.append(Convert2DTo3DTransform())

        tr_transforms.append(
            GaussianNoiseTransform(
                p_per_sample=self.hp_config.aug_gaussian_noise_prob *\
                    self.hp_config.aug_factor
            )
        )
        tr_transforms.append(
            GaussianBlurTransform(
                (0.5, 1.),
                different_sigma_per_channel=True,
                p_per_sample=self.hp_config.aug_gaussian_blur_prob *\
                    self.hp_config.aug_factor,
                p_per_channel=0.5
            )
        )
        tr_transforms.append(
            BrightnessMultiplicativeTransform(
                multiplier_range=(0.75, 1.25),
                p_per_sample=self.hp_config.aug_brightness_prob *\
                    self.hp_config.aug_factor
            )
        )
        tr_transforms.append(
            ContrastAugmentationTransform(
                p_per_sample=self.hp_config.aug_contrast_prob *\
                    self.hp_config.aug_factor
            )
        )
        tr_transforms.append(
            SimulateLowResolutionTransform(
                zoom_range=(0.5, 1),
                per_channel=True,
                p_per_channel=0.5,  # type: ignore
                order_downsample=0,
                order_upsample=3,
                p_per_sample=self.hp_config.aug_lowres_prob * self.hp_config.aug_factor,
                ignore_axes=ignore_axes
            )
        )
        tr_transforms.append(
            GammaTransform(
                gamma_range=(0.7, 1.5),
                invert_image=True,
                per_channel=True,
                retain_stats=True,
                p_per_sample=self.hp_config.aug_gamma_1_prob * self.hp_config.aug_factor
            )
        )
        tr_transforms.append(
            GammaTransform(
                gamma_range=(0.7, 1.5),
                invert_image=False,
                per_channel=True,
                retain_stats=True,
                p_per_sample=self.hp_config.aug_gamma_2_prob * self.hp_config.aug_factor
            )
        )

        if mirror_axes is not None and len(mirror_axes) > 0:
            tr_transforms.append(MirrorTransform(mirror_axes))

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            tr_transforms.append(
                MaskTransform(
                    [i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                    mask_idx_in_seg=0,
                    set_outside_to=0
                )
            )

        tr_transforms.append(RemoveLabelTransform(-1, 0))

        if is_cascaded:
            assert foreground_labels is not None, \
                "We need foreground_labels for cascade augmentations"
            tr_transforms.append(
                MoveSegAsOneHotToData(
                    1,
                    foreground_labels,
                    "seg",
                    "data"
                )
            )
            tr_transforms.append(ApplyRandomBinaryOperatorTransform(
                channel_idx=list(range(-len(foreground_labels), 0)),
                p_per_sample=0.4,
                key="data",
                strel_size=(1, 8),
                p_per_label=1))
            tr_transforms.append(
                RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                    channel_idx=list(range(-len(foreground_labels), 0)),
                    key="data",
                    p_per_sample=0.2,
                    fill_with_other_class_p=0,
                    dont_do_if_covers_more_than_x_percent=0.15))

        tr_transforms.append(RenameTransform("seg", "target", delete_old=True))

        if regions is not None:
            # the ignore label must also be converted
            tr_transforms.append(
                ConvertSegmentationToRegionsTransform(
                    [*list(regions), ignore_label]
                    if ignore_label is not None else regions,
                    "target",
                    "target"
                )
            )

        if deep_supervision_scales is not None:
            tr_transforms.append(DownsampleSegForDSTransform2(
                    deep_supervision_scales,
                    0,
                    input_key="target",
                    output_key="target"
                )
            )
        tr_transforms.append(NumpyToTensor(["data", "target"], "float"))
        return Compose(tr_transforms)

    def print_to_log_file(
            self,
            *args,
            also_print_to_console: bool = True,     # noqa: ARG002
            add_timestamp: bool = True              # noqa: ARG002
        ) -> None:
        """Prints a message to the log file.
        Replaces the nnU-Net default behaviour by using the logging module.

        Parameters:
        ----------
        *args : tuple
            The message to be printed.

        also_print_to_console : bool, optional
            Whether to also print the message to the console. Default is True.
            Unused in this implementation.

        add_timestamp : bool, optional
            Whether to add a timestamp to the message. Default is True.
            Unused in this implementation.
        """
        new_args = []
        for a in args:
            if isinstance(a, tuple | list):
                new_args += [str(s) for s in a]
            else:
                new_args.append(str(a))

        msg = " ".join(new_args)
        logging.getLogger("Trainer").info(msg)

    def set_hp_config(self, hp_config: DictConfig) -> None:
        """Sets the hyperparameter configuration.

        Parameters:
        ----------
        hp_config : DictConfig
            The hyperparameter configuration.
        """
        self.hp_config = hp_config

        self.initial_lr = self.hp_config.initial_lr
        self.weight_decay = self.hp_config.weight_decay
        self.oversample_foreground_percent = (
            self.hp_config.oversample_foreground_percent
        )
        self.num_epochs = round(self.hp_config.num_epochs)
        self.total_epochs = self.hp_config.total_epochs

        self.print_to_log_file(
            "Updated hyperparameter config:",
            also_print_to_console=True,
            add_timestamp=False,
        )
        for key, value in self.hp_config.items():
            self.print_to_log_file(
                f"{key}: {value}",
                also_print_to_console=True,
                add_timestamp=False,
            )

    def _build_loss(self) -> nn.Module:
        """Builds the loss function based on the internal hyperparameters.

        Returns:
        -------
        nn.Module
            The loss function.
        """
        if self.hp_config.loss_function == "DiceLoss":
            loss = self._build_dice_loss()
        elif self.hp_config.loss_function == "DiceAndCrossEntropyLoss":
            loss = self._build_dice_and_ce_loss()
        elif self.hp_config.loss_function == "CrossEntropyLoss":
            loss = self._build_ce_loss()
        elif self.hp_config.loss_function == "TopKLoss":
            loss = self._build_topk10_loss()
        else:
            raise ValueError(f"Invalid loss function: {self.hp_config.loss_function}")

        logging.getLogger("Trainer").info(f"Set loss to {type(loss).__name__}")

        return loss

    def _build_ce_loss(self):
        """Builds the CrossEntropyLoss.

        Returns:
        -------
        nn.Module
            The CrossEntropyLoss.
        """
        # taken from https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/nnUNetTrainer/variants/loss/nnUNetTrainerCELoss.py
        assert not self.label_manager.has_regions, \
            "regions not supported by this trainer"
        if self.label_manager.has_ignore_label:
            ignore_index = self.label_manager.ignore_label
        else:
            ignore_index = -100
        loss = RobustCrossEntropyLoss(
            weight=None,
            ignore_index=ignore_index
        )

        # we give each output a weight which decreases exponentially
        # (division by 2) as the resolution decreases this gives higher
        # resolution outputs more weight in the loss
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array(
                [1 / (2**i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

    def _build_dice_and_ce_loss(self) -> nn.Module:
        """Builds the compound Dice and CrossEntropy loss with a smooth factor of 1e-5.

        Returns:
        -------
        nn.Module
            The compound Dice and CrossEntropy loss with a smooth factor of 1e-5.
        """
        # taken from https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss(
                {},
                {
                    "batch_dice": self.configuration_manager.batch_dice,
                    "do_bg": True,
                    "smooth": 1e-5,
                    "ddp": self.is_ddp,
                },
                use_ignore_label=self.label_manager.ignore_label is not None,
                dice_class=MemoryEfficientSoftDiceLoss,
            )
        else:
            loss = DC_and_CE_loss(
                {
                    "batch_dice": self.configuration_manager.batch_dice,
                    "smooth": 1e-5,
                    "do_bg": False,
                    "ddp": self.is_ddp,
                },
                {},
                weight_ce=1,
                weight_dice=1,
                ignore_label=self.label_manager.ignore_label,
                dice_class=MemoryEfficientSoftDiceLoss,
            )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        # we give each output a weight which decreases exponentially
        # (division by 2) as the resolution decreases this gives higher
        # resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            assert deep_supervision_scales is not None
            weights = np.array(
                [1 / (2**i) for i in range(len(deep_supervision_scales))]
            )
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about
                # unused parameters due to weights[-1] = 0. Interestingly this crash
                # doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

    def _build_dice_loss(self) -> nn.Module:
        """Builds the DiceLoss with a smooth factor of 1e-5.

        Returns:
        -------
        nn.Module
            The DiceLoss with a smooth factor of 1e-5.
        """
        # taken from https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/nnUNetTrainer/variants/loss/nnUNetTrainerDiceLoss.py
        if self.label_manager.has_regions:
            apply_nonlin = torch.sigmoid
        else:
            apply_nonlin = softmax_helper_dim1
        loss = MemoryEfficientSoftDiceLoss(
            batch_dice=self.configuration_manager.batch_dice,
            do_bg=self.label_manager.has_regions,
            smooth=1e-5,
            ddp=self.is_ddp,
            apply_nonlin=apply_nonlin
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # we give each output a weight which decreases exponentially (division by 2)
            # as the resolution decreases this gives higher resolution outputs more
            # weight in the loss
            weights = np.array(
                [1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

    def _build_topk10_loss(self) -> nn.Module:
        """Builds the TopKLoss with k=10.

        Returns:
        -------
        nn.Module
            The TopKLoss with k=10.
        """
        # taken from https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/nnUNetTrainer/variants/loss/nnUNetTrainerTopkLoss.py
        assert not self.label_manager.has_regions, \
            "regions not supported by this trainer"
        if self.label_manager.has_ignore_label:
            ignore_index = self.label_manager.has_ignore_label
        else:
            ignore_index = -100
        loss = TopKLoss(
            ignore_index=ignore_index,
            k=10
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # we give each output a weight which decreases exponentially (division by 2)
            # as the resolution decreases this gives higher resolution outputs more
            # weight in the loss
            weights = np.array(
                [1 / (2**i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

    def _get_optimizer(self) -> torch.optim.Optimizer:  # type: ignore
        """Returns the optimizer based on the internal hyperparameters.

        Returns:
        -------
        torch.optim.Optimizer
            The optimizer.
        """
        assert self.network is not None

        if self.hp_config.optimizer == "SGD":
            return torch.optim.SGD(     # type: ignore
                self.network.parameters(),
                self.initial_lr,
                weight_decay=self.weight_decay,
                momentum=self.hp_config.momentum,
                nesterov=True,
            )
        if self.hp_config.optimizer == "Adam":
            return torch.optim.Adam(    # type: ignore
                self.network.parameters(),
                self.initial_lr,
                weight_decay=self.weight_decay,
            )
        if self.hp_config.optimizer == "AdamW":
            return torch.optim.AdamW(   # type: ignore
                self.network.parameters(),
                self.initial_lr,
                weight_decay=self.weight_decay,
                amsgrad=True
            )
        raise ValueError(f"Invalid optimizer: {self.hp_config.optimizer}")

    def _get_lr_scheduler(
            self,
            optimizer: torch.optim.Optimizer    # type: ignore
        ) -> torch.optim.lr_scheduler.LRScheduler:
        """Returns the learning rate scheduler based on the internal hyperparameters.

        Returns:
        -------
        torch.optim.lr_scheduler.LRScheduler
            The learning rate scheduler.
        """
        if self.hp_config.lr_scheduler == "PolyLRScheduler":
            return PolyLRScheduler(
                optimizer, self.initial_lr, self.total_epochs
            )
        if self.hp_config.lr_scheduler == "CosineAnnealingLR":
            return CosineAnnealingLR(optimizer, T_max=self.total_epochs)
        if self.hp_config.lr_scheduler == "None":
            # We use the dummy here to avoid the overhead of checking the
            # lr_scheduler type in the training loop
            return DummyLRScheduler(optimizer)
        raise ValueError(f"Invalid lr_scheduler: {self.hp_config.lr_scheduler}")

    def configure_optimizers(
            self
        ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:    # type: ignore
        """Configures the optimizer and learning rate scheduler.

        Returns:
        -------
        tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]
            The optimizer and learning rate scheduler.
        """
        optimizer = self._get_optimizer()
        lr_scheduler = self._get_lr_scheduler(optimizer)

        logging.getLogger("Trainer").info(
            f"Set optimizer to {type(optimizer).__name__}")
        logging.getLogger("Trainer").info(
            f"Set lr_scheduler to {type(lr_scheduler).__name__}")

        return optimizer, lr_scheduler

