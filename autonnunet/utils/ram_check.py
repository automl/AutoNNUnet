"""Memory-budget guard for `trainer.cache_dataset_in_ram` (issue #11).
TODO: See if this simple Memory-budged guard is sufficient or if I have to implement a more dedicated cache logic.
"""
from __future__ import annotations

import logging
import os
import zipfile
from collections.abc import Iterable

logger = logging.getLogger("Trainer")

_GIB = 1024**3


def available_ram_bytes() -> int:
    """Available host RAM in bytes."""
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except ImportError:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    raise RuntimeError("Could not determine available RAM.")


def case_size_bytes(folder: str, key: str) -> int:
    """Uncompressed size of one case (image + seg) in bytes.

    Parameters
    ----------
    folder : str
        The preprocessed dataset folder.
    key : str
        The case identifier.

    Returns:
    -------
    int
        The size in bytes.
    """
    base = os.path.join(folder, key)
    npy, seg_npy = base + ".npy", base + "_seg.npy"
    if os.path.isfile(npy):
        size = os.path.getsize(npy)
        if os.path.isfile(seg_npy):
            size += os.path.getsize(seg_npy)
        return size
    with zipfile.ZipFile(base + ".npz") as zf:
        return sum(info.file_size for info in zf.infolist())


def check_ram(folder: str, keys: Iterable[str]) -> None:
    """Warns and aborts if the cases need more RAM than available.

    Parameters
    ----------
    folder : str
        The preprocessed dataset folder.
    keys : Iterable[str]
        The case identifiers to check.

    Raises:
    ------
    MemoryError
        If more RAM is needed than available.
    """
    keys = list(keys)
    needed = sum(case_size_bytes(folder, k) for k in keys)
    available = available_ram_bytes()

    logger.info(
        f"RAM check: {len(keys)} cases need ~{needed / _GIB:.2f} GiB, "
        f"available {available / _GIB:.2f} GiB."
    )

    if needed > available:
        msg = (
            f"Not enough RAM: dataset needs ~{needed / _GIB:.2f} GiB, "
            f"but only {available / _GIB:.2f} GiB are available. Aborting training. "
            "Free up RAM or, if trainer.cache_dataset_in_ram=true, set it to false."
        )
        logger.warning(msg)
        raise MemoryError(msg)
