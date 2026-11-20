"""Per-label segmentation metrics based on DeepMind's `surface-distance` package.

Computed on the validation fold after training via `compute_val_surface_metrics`
(called from runscripts/train.py):
Hausdorff (HD95), average symmetric surface distance, surface Dice (NSD) and
surface overlap at a tolerance in mm, volumes and the relative volume error.
Dice is reported as a secondary metric.

All distances are in mm, using the voxel spacing of the exported segmentations
(original image spacing, since nnU-Net resamples predictions back before export).
"""
from __future__ import annotations

import logging
import math
import multiprocessing
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("Trainer")

METRICS = [
    "dice",
    "hd95_mm",
    "assd_mm",
    "nsd",
    "surface_overlap_gt",
    "surface_overlap_pred",
    "volume_gt_ml",
    "volume_pred_ml",
    "rel_volume_error",
]


def label_definitions(dataset_json: dict, label_manager: Any) -> dict[str, list[int]]:
    """Maps label (or region) names to the label values that belong to them.

    Plain labels: {"anterior": [1], ...}; regions: {"tumor_core": [2, 3], ...}.
    Background / ignore labels are skipped.
    """
    labels: dict[str, list[int]] = {}
    for name, value in dataset_json["labels"].items():
        if name in ("background", "ignore"):
            continue
        values = [int(v) for v in value] if isinstance(value, list | tuple) else [int(value)]
        if not label_manager.has_regions and values[0] not in label_manager.foreground_labels:
            continue
        labels[name] = values
    return labels


def compute_label_metrics(
    pred: np.ndarray, gt: np.ndarray, spacing_mm: tuple[float, ...], tolerance_mm: float
) -> dict[str, float]:
    """All metrics for one binary mask pair. NaN where a metric is undefined
    (e.g. distances when one of the masks is empty).
    """
    import surface_distance as sd

    pred, gt = pred.astype(bool), gt.astype(bool)
    voxel_ml = float(np.prod(spacing_mm)) / 1000.0
    vol_gt, vol_pred = float(gt.sum()) * voxel_ml, float(pred.sum()) * voxel_ml

    m: dict[str, float] = dict.fromkeys(METRICS, math.nan)
    m["volume_gt_ml"] = vol_gt
    m["volume_pred_ml"] = vol_pred
    m["rel_volume_error"] = (vol_pred - vol_gt) / vol_gt if vol_gt > 0 else math.nan

    if not gt.any() and not pred.any():
        # Label correctly absent: perfect overlap, distances undefined
        m["dice"] = 1.0
        return m

    m["dice"] = float(sd.compute_dice_coefficient(gt, pred))
    if not gt.any() or not pred.any():
        # One mask empty: overlap-based metrics are 0, distances undefined (inf)
        m["nsd"] = m["surface_overlap_gt"] = m["surface_overlap_pred"] = 0.0
        return m

    dists = sd.compute_surface_distances(gt, pred, spacing_mm)
    m["hd95_mm"] = float(sd.compute_robust_hausdorff(dists, 95))
    gt_to_pred, pred_to_gt = sd.compute_average_surface_distance(dists)
    m["assd_mm"] = float((gt_to_pred + pred_to_gt) / 2)
    m["nsd"] = float(sd.compute_surface_dice_at_tolerance(dists, tolerance_mm))
    overlap_gt, overlap_pred = sd.compute_surface_overlap_at_tolerance(dists, tolerance_mm)
    m["surface_overlap_gt"] = float(overlap_gt)
    m["surface_overlap_pred"] = float(overlap_pred)
    return m


def _evaluate_case(args: tuple) -> list[dict]:
    case, pred_file, gt_file, reader_writer, labels, tolerance_mm = args
    gt_seg, gt_props = reader_writer.read_seg(str(gt_file))
    pred_seg, _ = reader_writer.read_seg(str(pred_file))
    spacing = tuple(float(s) for s in gt_props["spacing"])
    return [
        {"case": case, "label": name, **compute_label_metrics(
            np.isin(pred_seg[0], values), np.isin(gt_seg[0], values), spacing, tolerance_mm
        )}
        for name, values in labels.items()
    ]


def evaluate_predictions(
    trainer: Any, pred_folder: str | Path, tolerance_mm: float = 2.0
) -> pd.DataFrame:
    """Surface metrics for the predictions in `pred_folder` (e.g. `<run>/validation`)
    against nnU-Net's `gt_segmentations`.

    Writes `surface_metrics_per_case.csv` into `pred_folder` and returns the
    per-label means (NaNs ignored) as a DataFrame (index: label, columns: METRICS).
    """
    pred_folder = Path(pred_folder)
    gt_folder = Path(trainer.preprocessed_dataset_folder_base) / "gt_segmentations"
    file_ending = trainer.dataset_json["file_ending"]
    reader_writer = trainer.plans_manager.image_reader_writer_class()
    labels = label_definitions(trainer.dataset_json, trainer.label_manager)

    jobs = [
        (f.name[: -len(file_ending)], f, gt_folder / f.name, reader_writer, labels, tolerance_mm)
        for f in sorted(pred_folder.glob(f"*{file_ending}"))
    ]
    if not jobs:
        raise FileNotFoundError(f"No predictions (*{file_ending}) found in {pred_folder}")

    num_processes = min(8, os.cpu_count() or 1, len(jobs))
    if num_processes > 1:
        with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
            results = pool.map(_evaluate_case, jobs)
    else:
        results = [_evaluate_case(j) for j in jobs]

    per_case = pd.DataFrame([row for rows in results for row in rows])
    per_case.to_csv(pred_folder / "surface_metrics_per_case.csv", index=False)
    label_means = per_case.groupby("label", sort=False)[METRICS].mean()
    logger.info(
        "Surface metrics (%d cases): %s", len(jobs),
        {k: round(v, 4) for k, v in label_means.mean().items()},
    )
    return label_means


def to_scalars(label_means: pd.DataFrame, prefix: str) -> dict[str, float]:
    """TensorBoard scalars `<prefix>/<label>/<metric>` plus the mean over all labels
    as `<prefix>/foreground_mean/<metric>`.
    """
    table = pd.concat([label_means, label_means.mean().to_frame("foreground_mean").T])
    return {
        f"{prefix}/{label}/{metric}": float(value)
        for label, row in table.iterrows()
        for metric, value in row.items()
        if math.isfinite(value)
    }


# Metrics that go into the TensorBoard HParams tab (mean over all foreground labels).
HPARAM_METRICS = ["dice", "hd95_mm", "assd_mm", "nsd", "rel_volume_error"]


def compute_val_surface_metrics(trainer: Any, cfg: Any) -> dict[str, float]:
    """Computes surface metrics on the validation fold and logs them.

    What is computed:
        For every validation case and every foreground label (or region) the
        prediction in `<output_folder>/validation` is compared to nnU-Net's ground
        truth: HD95, ASSD, surface Dice (NSD) and surface overlap at a tolerance of
        `cfg.surface_metrics.nsd_tolerance_mm` (default 2 mm), volumes, relative
        volume error and Dice. All distances in mm.

    When to call:
        After `trainer.perform_actual_validation()` and BEFORE the validation files
        are removed (`cfg.pipeline.remove_validation_files`), because the metrics are
        computed from the exported prediction files (.nii.gz) in that folder.

    Where the results go:
        - `<output_folder>/validation/surface_metrics_per_case.csv`: one row per
          case and label (raw values).
        - TensorBoard scalars `val_surface/<label>/<metric>` and
          `val_surface/foreground_mean/<metric>` (per-label means over all cases).
        - Return value `{"val_surface_<metric>": mean over labels}` for
          HPARAM_METRICS; train.py passes it to `monitor.log_final` so it shows up
          in the TensorBoard HParams tab.

    Errors are caught and logged, and an empty dict is returned: these metrics
    are only additional information and must never cost us the HPO objective
    (the validation Dice) of a run.
    """
    logger.info("Computing surface metrics on the validation fold")
    try:
        tolerance_mm = float(cfg.get("surface_metrics", {}).get("nsd_tolerance_mm", 2.0))
        label_means = evaluate_predictions(
            trainer, Path(trainer.output_folder) / "validation", tolerance_mm
        )
        trainer.monitor.log_scalars(
            to_scalars(label_means, "val_surface"), step=trainer.current_epoch
        )
        return {
            f"val_surface_{metric}": float(value)
            for metric, value in label_means[HPARAM_METRICS].mean().items()
        }
    except Exception as e:  # noqa: BLE001
        logger.error(f"Surface metrics on the validation fold failed: {e}")
        return {}
