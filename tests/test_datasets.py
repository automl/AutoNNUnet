#!/usr/bin/env python
"""Tests for MSD datasets in the `autonnunet` package."""

import pytest
from autonnunet.datasets import MSDDataset
from autonnunet.datasets.msd_dataset import MSD_URLS


@pytest.mark.parametrize("name, id", [
    ("Dataset001_BrainTumour", 1),
    ("Dataset002_Heart", 2),
    ("Dataset003_Liver", 3),
    ("Dataset004_Hippocampus", 4),
    ("Dataset005_Prostate", 5),
    ("Dataset006_Lung", 6),
    ("Dataset007_Pancreas", 7),
    ("Dataset008_HepaticVessel", 8),
    ("Dataset009_Spleen", 9),
    ("Dataset010_Colon", 10),
])
def test_msd_dataset(name: str, id: int):
    """Test if numpy seeding works."""
    ds = MSDDataset(
        name=name
    )

    assert ds.get_url() == MSD_URLS[name]
    assert ds.dataset_id == id


