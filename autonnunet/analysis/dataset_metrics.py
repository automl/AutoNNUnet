"""Functions for computing metrics on segmentation masks."""
from __future__ import annotations

import numpy as np


def compute_compactness(class_mask: np.ndarray) -> float:
    """Computes the compactness of a binary segmentation mask.

    Parameters
    ----------
    class_mask : np.ndarray
        The binary segmentation mask.

    Returns:
    -------
    float
        The compactness of the binary segmentation
    """
    volume = np.sum(class_mask)
    if volume == 0:
        return 0.

    # We need the centroid to compute the radius of the enclosing sphere
    coordinates = np.argwhere(class_mask)
    centroid = np.mean(coordinates, axis=0)

    # Now we can compute the radius of the enclosing sphere
    distances = np.linalg.norm(coordinates - centroid, axis=1)
    radius = np.max(distances)

    if radius == 0:
        return float("inf")
    return volume / ((4 / 3) * np.pi * radius**3)



def compute_std_per_axis(class_mask: np.ndarray) -> float:
    """Computes the standard deviation of the binary segmentation mask along each axis.

    Parameters
    ----------
    class_mask : np.ndarray
        The binary segmentation mask.

    Returns:
    -------
    float
        The standard deviation of the binary segmentation mask along each axis.
    """
    if class_mask.ndim == 1:
        return float(np.std(class_mask))
    if class_mask.ndim == 2:
        return float(np.std(class_mask, axis=(0, 1)))
    if class_mask.ndim == 3:
        return float(np.std(class_mask, axis=(0, 1, 2)))
    raise ValueError("Array must have at most 4 dimensions")