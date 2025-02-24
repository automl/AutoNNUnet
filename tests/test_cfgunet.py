#!/usr/bin/env python
"""Test for the CFGUNet for `autonnunet` package."""

import pytest
from autonnunet.hnas import get_default_architecture, CFGUNet


@pytest.mark.parametrize("n_stages", [
    (4),
    (5),
    (6),
])
def test_default_arch(n_stages: int):
    """Test the default architecture for a
    given number of stages."""
    string_tree = get_default_architecture(n_stages=n_stages)

    parsed_tree = CFGUNet.parse_nested_brackets(string_tree)
    encoder_cfg, decoder_cfg = CFGUNet.extract_architecture_cfg(parsed_tree)

    assert encoder_cfg["network_type"] == "conv_encoder"
    assert decoder_cfg["network_type"] == "conv_decoder"

    assert len(encoder_cfg["n_blocks_per_stage"]) == n_stages
    assert len(decoder_cfg["n_blocks_per_stage"]) == n_stages - 1

    for b in encoder_cfg["n_blocks_per_stage"]:
        assert b == 2
    
    for b in decoder_cfg["n_blocks_per_stage"]:
        assert b == 2
