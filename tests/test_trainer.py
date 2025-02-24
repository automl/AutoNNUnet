#!/usr/bin/env python
"""Tests for the trainer in the `autonnunet` package."""

import pytest
from autonnunet.training import AutoNNUNetTrainer
from omegaconf import OmegaConf, DictConfig
import hydra

@pytest.fixture
def cfg():
    """Create a trainer."""
    hydra.initialize(config_path="../runscripts/configs", version_base=None) 
    return hydra.compose(config_name="train")

@pytest.fixture
def trainer(cfg):
    """Create a trainer."""
    trainer = AutoNNUNetTrainer.from_config(cfg)
    return trainer

def test_trainer(cfg, trainer):
    """Test if trainer can be created."""
    assert trainer.hp_config == cfg.hp_config