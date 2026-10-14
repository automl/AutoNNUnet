"""An nnU-Net dataset that keeps all case data resident in host RAM."""
from __future__ import annotations

import logging

import numpy as np
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset

logger = logging.getLogger("Trainer")


class InMemoryNNUNetDataset(nnUNetDataset):
    """nnUNetDataset variant that eagerly loads every case's image and
    segmentation arrays into RAM once, then serves `load_case()` from that
    cache instead of re-reading from disk on every call.

    The base `nnUNetDataset.load_case()` already supports an "open file"
    cache (`nnUNet_keep_files_open`), but that only keeps a memory-mapped
    file handle around - reads still hit the underlying filesystem on every
    page access. When the preprocessed data lives on a slow/network-mounted
    filesystem, that per-batch, per-epoch re-reading is the actual
    bottleneck this class removes: every array is copied out of the mmap
    view into a real, fully materialized numpy array exactly once, in
    `__init__`.

    This must be constructed in the main training process, before nnU-Net
    spawns its `NonDetMultiThreadedAugmenter` dataloader worker processes
    (which happens right after `get_tr_and_val_datasets()` returns, inside
    `get_dataloaders()`). On Linux, those workers are `fork()`ed rather than
    spawned, so they inherit this cache via copy-on-write without
    duplicating it in RAM, as long as nothing later mutates the cached
    arrays in place.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Builds the case index (see `nnUNetDataset.__init__`), then
        immediately preloads every case's data into RAM.
        """
        super().__init__(*args, **kwargs)

        self._cache: dict[str, tuple[np.ndarray, np.ndarray, dict]] = {}

        n_cases = len(self.dataset)
        logger.info(f"Preloading {n_cases} cases into RAM...")
        for i, key in enumerate(self.dataset):
            self._cache[key] = self._load_case_into_ram(key)
            if (i + 1) % 25 == 0 or i + 1 == n_cases:
                logger.info(f"Preloaded {i + 1}/{n_cases} cases into RAM.")
        logger.info("Finished preloading dataset into RAM.")

    def _load_case_into_ram(self, key: str) -> tuple[np.ndarray, np.ndarray, dict]:
        """Loads a single case via the base class, forcing full materialization.

        `nnUNetDataset.load_case()` returns a memory-mapped array (`np.load(...,
        'r')`) whenever the case has already been unpacked to `.npy` on disk.
        `np.array(...)` forces a real, owned-memory copy rather than another
        view over the same (possibly slow/network-backed) file.
        """
        data, seg, properties = super().load_case(key)
        return np.array(data), np.array(seg), properties

    def load_case(self, key: str) -> tuple[np.ndarray, np.ndarray, dict]:
        """Returns the cached, fully in-RAM case data."""
        return self._cache[key]
