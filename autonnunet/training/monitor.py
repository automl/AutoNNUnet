"""Training monitor: collects metrics from an AutoNNUNetTrainer and writes
them to TensorBoard (`<run dir>/tensorboard`).
Monitoring errors are logged as warnings and never abort a trial.
"""
from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger("Trainer")

_GIB = 1024 ** 3


def _never_raise(method):  # noqa: ANN001, ANN202
    """Turns exceptions in monitoring code into warnings."""

    @functools.wraps(method)
    def wrapper(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        try:
            return method(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Monitoring ({method.__name__}) failed: {e}")
            return None

    return wrapper


class TrainingMonitor:
    """Collects per-epoch and final metrics of one training run (trial) and
    writes them to TensorBoard.
    """

    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = str(log_dir)
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def _write_scalars(self, scalars: dict[str, float], step: int) -> None:
        for tag, value in scalars.items():
            if value is not None:
                self.writer.add_scalar(tag, float(value), global_step=step)

    # ----------------------------------------------------------------------------------
    # Hooks called by the trainer
    # ----------------------------------------------------------------------------------
    @_never_raise
    def on_train_start(self, trainer: Any) -> None:
        """Logs the hyperparameter configuration (incl. HNAS architecture) as text."""
        if trainer.local_rank != 0:
            return
        # Markdown code block so that the YAML indentation is preserved
        self.writer.add_text(
            "config/hp_config",
            f"```\n{OmegaConf.to_yaml(trainer.hp_config)}\n```",
            global_step=trainer.current_epoch,
        )

    @_never_raise
    def on_epoch_start(self, trainer: Any) -> None:
        """Resets the GPU peak memory statistics so they are reported per epoch."""
        if trainer.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(trainer.device)

    @_never_raise
    def on_epoch_end(self, trainer: Any, epoch: int) -> None:
        """Logs the metrics of the finished epoch (call after nnU-Net's on_epoch_end)."""
        if trainer.local_rank != 0:
            return
        metrics = self._epoch_metrics(trainer)
        metrics.update(self._system_metrics(trainer))
        self._write_scalars(metrics, step=epoch)

    @_never_raise
    def on_train_end(self) -> None:
        self.writer.flush()

    @_never_raise
    def log_scalars(self, scalars: dict[str, float], step: int) -> None:
        """Logs additional scalars, e.g. hold-out metrics."""
        self._write_scalars(scalars, step=step)

    @_never_raise
    def log_final(
        self, trainer: Any, objectives: dict, extra_metrics: dict | None = None
    ) -> None:
        """Logs hp_config + final objectives for the HParams tab.

        Parameters
        ----------
        trainer : AutoNNUNetTrainer
            The trainer of this run.
        objectives : dict
            The objectives as returned by `read_objectives`.
        extra_metrics : dict | None
            Additional final metrics, e.g. surface metrics on the validation fold.
        """
        if trainer.local_rank != 0:
            return
        metrics = {"val_dice": 1 - objectives["loss"], "runtime_h": objectives["runtime"]}
        if trainer._best_ema is not None:  # noqa: SLF001
            metrics["best_ema_pseudo_dice"] = float(trainer._best_ema)  # noqa: SLF001
        if extra_metrics:
            metrics.update({k: v for k, v in extra_metrics.items() if np.isfinite(v)})

        hparams = OmegaConf.to_container(trainer.hp_config, resolve=True)
        hparams["fold"] = trainer.fold
        # TensorBoard's HParams tab only accepts int/float/str/bool values
        hparams = {
            k: v if isinstance(v, int | float | str | bool) else str(v)
            for k, v in hparams.items() if v is not None
        }
        # run_name="." keeps the entry in the run's own folder instead of a
        # timestamped sub folder, so each trial shows up as one run
        self.writer.add_hparams(
            hparams, {f"hparams/{k}": float(v) for k, v in metrics.items()}, run_name="."
        )

    def close(self) -> None:
        self.writer.close()

    @staticmethod
    def _epoch_metrics(trainer: Any) -> dict[str, float]:
        """Metrics of the last epoch from the nnU-Net logger."""
        log = trainer.logger.my_fantastic_logging
        metrics: dict[str, float] = {
            "loss/train": log["train_losses"][-1],
            "loss/val": log["val_losses"][-1],
            "dice/pseudo_mean_fg": log["mean_fg_dice"][-1],
            "dice/pseudo_ema_fg": log["ema_fg_dice"][-1],
            "optim/lr": log["lrs"][-1],
            "time/epoch_s": log["epoch_end_timestamps"][-1] - log["epoch_start_timestamps"][-1],
        }

        # Label names for the entries of nnU-Net's `dice_per_class_or_region`
        label_manager = trainer.label_manager
        if label_manager.has_regions:
            names = [f"region_{i}" for i in range(len(label_manager.foreground_regions))]
        else:
            id_to_name = {
                v: k for k, v in trainer.dataset_json["labels"].items() if isinstance(v, int)
            }
            names = [
                id_to_name.get(label, f"label_{label}") for label in label_manager.foreground_labels
            ]

        for i, value in enumerate(log["dice_per_class_or_region"][-1]):
            name = names[i] if i < len(names) else str(i)
            metrics[f"dice_per_class/{name}"] = value
        return metrics

    @staticmethod
    def _system_metrics(trainer: Any) -> dict[str, float]:
        """RAM and GPU memory usage of the training process."""
        metrics: dict[str, float] = {}
        try:
            import psutil

            proc = psutil.Process()
            rss_main = proc.memory_info().rss
            rss_total = rss_main
            # Data augmentation workers. Shared pages (e.g. memmapped data)
            # are counted per process, so this is an upper bound.
            for child in proc.children(recursive=True):
                try:
                    rss_total += child.memory_info().rss
                except psutil.Error:
                    pass
            vm = psutil.virtual_memory()
            metrics["system/ram_rss_main_gb"] = rss_main / _GIB
            metrics["system/ram_rss_incl_workers_gb"] = rss_total / _GIB
            metrics["system/ram_available_gb"] = vm.available / _GIB
            metrics["system/ram_used_percent"] = vm.percent
        except ImportError:
            pass

        if trainer.device.type == "cuda" and torch.cuda.is_available():
            metrics["system/gpu_peak_allocated_gb"] = (
                torch.cuda.max_memory_allocated(trainer.device) / _GIB
            )
            metrics["system/gpu_peak_reserved_gb"] = (
                torch.cuda.max_memory_reserved(trainer.device) / _GIB
            )
        return metrics
