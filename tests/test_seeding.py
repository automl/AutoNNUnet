#!/usr/bin/env python
"""Tests for seeding of the `autonnunet` package."""

import pytest
import numpy as np
import torch


def test_numpy_seeding():
    """Test if numpy seeding works."""
    np.random.seed(42)
    a = np.random.rand(10)

    np.random.seed(42)
    b = np.random.rand(10)

    assert np.allclose(a, b)


def test_torch_seeding():
    """Test if torch seeding works."""
    torch.manual_seed(42)
    a = torch.rand(10)

    torch.manual_seed(42)
    b = torch.rand(10)

    assert torch.allclose(a, b)


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
