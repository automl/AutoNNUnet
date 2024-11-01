from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import load_json
from dynamic_network_architectures.building_blocks.helper import \
    get_matching_batchnorm
from nnunetv2.training.loss.compound_losses import (DC_and_BCE_loss,
                                                    DC_and_CE_loss)
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import (RobustCrossEntropyLoss,
                                                   TopKLoss)
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch.optim.lr_scheduler import CosineAnnealingLR

from autonnunet.training.auto_nnunet_logger import AutoNNUNetLogger
from autonnunet.training.dummy_lr_scheduler import DummyLRScheduler
from autonnunet.utils import get_device
from autonnunet.utils.paths import NNUNET_PREPROCESSED

if TYPE_CHECKING:
    from omegaconf import DictConfig


class AutoNNUNetTrainer(nnUNetTrainer):
    def __init__(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            unpack_dataset: bool = True,
            device: torch.device = torch.device("cuda"),
        ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        self.output_folder_base = "."
        self.output_folder = "."

        if self.is_cascaded:
            assert self.configuration_manager.previous_stage_name is not None
            self.folder_with_segs_from_previous_stage = os.path.join(
                ".",
                self.configuration_manager.previous_stage_name,
                "predicted_segmentations",
                self.configuration_name,
            )
        else:
            self.folder_with_segs_from_previous_stage = None

        self.log_file = os.path.join(self.output_folder, "training_log.txt")
        self.logger = AutoNNUNetLogger()

    @staticmethod
    def from_config(cfg: DictConfig) -> AutoNNUNetTrainer:
        preprocessed_dataset_folder_base = NNUNET_PREPROCESSED / cfg.dataset.name
        plans_file = preprocessed_dataset_folder_base / f"{cfg.trainer.plans_identifier}.json"
        plans = load_json(plans_file)
        dataset_json = load_json(preprocessed_dataset_folder_base / "dataset.json")

        nnunet_trainer = AutoNNUNetTrainer(
            plans=plans,
            configuration=cfg.trainer.configuration,
            fold=cfg.fold,
            dataset_json=dataset_json,
            unpack_dataset=not cfg.trainer.use_compressed_data,
            device=get_device(cfg.device),
        )
        nnunet_trainer.set_hp_config(cfg.hp_config)

        nnunet_trainer.disable_checkpointing = cfg.trainer.disable_checkpointing

        if cfg.load:
            load_path_best = Path(cfg.load + f"_fold_{cfg.fold}_best.pth").resolve()
            load_path_final = Path(cfg.load + f"_fold_{cfg.fold}_final.pth").resolve()
            checkpoint_best_path = Path().resolve() / "checkpoint_best.pth"

            shutil.copyfile(load_path_best, checkpoint_best_path)
            nnunet_trainer.load_checkpoint(str(load_path_final))

        if cfg.pipeline.continue_training and Path("./checkpoint_best.pth").exists():
            nnunet_trainer.load_checkpoint("checkpoint_best.pth")
                
        return nnunet_trainer

    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):
        new_args = []
        for a in args:
            if isinstance(a, tuple | list):
                new_args += [str(s) for s in a]
            else:
                new_args.append(str(a))

        msg = " ".join(new_args)
        try:
            logging.getLogger("Trainer").info(msg)
        except Exception as e:
            logging.getLogger("Trainer").error(e)

    def set_hp_config(self, hp_config: DictConfig) -> None:
        self.hp_config = hp_config

        self.initial_lr = self.hp_config.initial_lr
        self.weight_decay = self.hp_config.weight_decay
        self.oversample_foreground_percent = (
            self.hp_config.oversample_foreground_percent
        )
        self.num_epochs = round(self.hp_config.num_epochs)
        self.total_epochs = self.hp_config.total_epochs
        self.enable_deep_supervision = self.hp_config.enable_deep_supervision

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

        # Update hyperparameter configuration stored in configuration manager
        self.configuration_manager.configuration["architecture"]["arch_kwargs"]["n_conv_per_stage"] = [self.hp_config.n_conv_per_stage_encoder] * len(self.configuration_manager.configuration["architecture"]["arch_kwargs"]["n_conv_per_stage"])
        self.configuration_manager.configuration["architecture"]["arch_kwargs"]["n_conv_per_stage_decoder"] = [self.hp_config.n_conv_per_stage_decoder] * len(self.configuration_manager.configuration["architecture"]["arch_kwargs"]["n_conv_per_stage_decoder"])

        self.print_to_log_file(
            "Updated normalization in configuration manager:",
            also_print_to_console=True,
            add_timestamp=False,
        )
        self.print_to_log_file(
            f'self.configuration_manager.configuration["architecture"]["arch_kwargs"]["n_conv_per_stage"] = {self.configuration_manager.configuration["architecture"]["arch_kwargs"]["n_conv_per_stage"]}',
            also_print_to_console=True,
            add_timestamp=False,
        )
        self.print_to_log_file(
            f'self.configuration_manager.configuration["architecture"]["arch_kwargs"]["n_conv_per_stage_decoder"] = {self.configuration_manager.configuration["architecture"]["arch_kwargs"]["n_conv_per_stage_decoder"]}',
            also_print_to_console=True,
            add_timestamp=False,
        )

        # the following part is taken from https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/nnUNetTrainer/variants/network_architecture/nnUNetTrainerBN.py
        if self.hp_config.normalization == "BatchNorm":
            from pydoc import locate
            conv_op = locate(self.configuration_manager.configuration["architecture"]["arch_kwargs"]["conv_op"])
            bn_class = get_matching_batchnorm(conv_op)
            self.configuration_manager.configuration["architecture"]["arch_kwargs"]["norm_op"] = bn_class.__module__ + "." + bn_class.__name__
            self.configuration_manager.configuration["architecture"]["arch_kwargs"]["norm_op_kwargs"] = {"eps": 1e-5, "affine": True}

            self.print_to_log_file(
                f'self.configuration_manager.configuration["architecture"]["arch_kwargs"]["norm_op"] = {self.configuration_manager.configuration["architecture"]["arch_kwargs"]["norm_op"]}',
                also_print_to_console=True,
                add_timestamp=False,
            )
            self.print_to_log_file(
                f'self.configuration_manager.configuration["architecture"]["arch_kwargs"]["norm_op_kwargs"] = {self.configuration_manager.configuration["architecture"]["arch_kwargs"]["norm_op_kwargs"]}',
                also_print_to_console=True,
                add_timestamp=False,
            )

    def _build_loss(self) -> torch.nn.Module:
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
        # taken from https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/nnUNetTrainer/variants/loss/nnUNetTrainerCELoss.py
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        loss = RobustCrossEntropyLoss(
            weight=None, ignore_index=self.label_manager.ignore_label if self.label_manager.has_ignore_label else -100
        )

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

    def _build_dice_and_ce_loss(self) -> torch.nn.Module:
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

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            assert deep_supervision_scales is not None
            weights = np.array(
                [1 / (2**i) for i in range(len(deep_supervision_scales))]
            )
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

    def _build_dice_loss(self) -> torch.nn.Module:
        # taken from https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/nnUNetTrainer/variants/loss/nnUNetTrainerDiceLoss.py
        loss = MemoryEfficientSoftDiceLoss(
            batch_dice=self.configuration_manager.batch_dice, do_bg=self.label_manager.has_regions, smooth=1e-5, ddp=self.is_ddp,
            apply_nonlin=torch.sigmoid if self.label_manager.has_regions else softmax_helper_dim1
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

    def _build_topk10_loss(self) -> torch.nn.Module:
        # taken from https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/nnUNetTrainer/variants/loss/nnUNetTrainerTopkLoss.py
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        loss = TopKLoss(
            ignore_index=self.label_manager.ignore_label if self.label_manager.has_ignore_label else -100, k=10
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

    def _get_optimizer(self) -> torch.optim.Optimizer:
        assert self.network is not None

        if self.hp_config.optimizer == "SGD":
            return torch.optim.SGD(
                self.network.parameters(),
                self.initial_lr,
                weight_decay=self.weight_decay,
                momentum=self.hp_config.momentum,
                nesterov=True,
            )
        if self.hp_config.optimizer == "Adam":
            return torch.optim.Adam(
                self.network.parameters(),
                self.initial_lr,
                weight_decay=self.weight_decay,
            )
        if self.hp_config.optimizer == "AdamW":
            return torch.optim.AdamW(
                self.network.parameters(),
                self.initial_lr,
                weight_decay=self.weight_decay,
                amsgrad=True
            )
        raise ValueError(f"Invalid optimizer: {self.hp_config.optimizer}")

    def _get_lr_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
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
        ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        optimizer = self._get_optimizer()
        lr_scheduler = self._get_lr_scheduler(optimizer)

        logging.getLogger("Trainer").info(
            f"Set optimizer to {type(optimizer).__name__}")
        logging.getLogger("Trainer").info(
            f"Set lr_scheduler to {type(lr_scheduler).__name__}")

        return optimizer, lr_scheduler

