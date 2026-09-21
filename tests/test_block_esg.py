"""Tests for Block-ESG (EventSaliencyGate) shape, range, and behaviour."""

import numpy as np
import pytest
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_esg_output_shapes():
    """EventSaliencyGate output shapes match expected (B, M*L, D) and (B, M*L)."""
    from model import EventSaliencyGate
    B, T, D = 2, 120, 96  # micro_dim with 32 ch = 3*32=96
    L, M = 15, 4
    gate = EventSaliencyGate(input_dim=D, block_size=L, top_m=M)
    x = torch.randn(B, T, D)
    # Make first 60 frames valid
    x[:, 60:] = 0.0
    sel_frames, sel_indices = gate(x)
    assert sel_frames.shape == (B, M * L, D), f"Expected ({B}, {M*L}, {D}), got {sel_frames.shape}"
    assert sel_indices.shape == (B, M * L), f"Expected ({B}, {M*L}), got {sel_indices.shape}"


def test_esg_indices_are_valid():
    """Selected frame indices must be within valid range."""
    from model import EventSaliencyGate
    B, T, D = 2, 150, 96
    L, M = 15, 4
    gate = EventSaliencyGate(input_dim=D, block_size=L, top_m=M)
    x = torch.randn(B, T, D)
    x[:, 100:] = 0.0
    _, sel_indices = gate(x)
    # Indices must be >= 0
    assert (sel_indices >= 0).all(), "Negative indices found"


def test_esg_block_scores_shape():
    """get_block_scores() returns (B, N_blocks) after forward pass."""
    from model import EventSaliencyGate
    B, T, D = 2, 90, 96
    L, M = 15, 4
    gate = EventSaliencyGate(input_dim=D, block_size=L, top_m=M)
    x = torch.randn(B, T, D)
    _ = gate(x)
    scores = gate.get_block_scores()
    assert scores is not None
    assert scores.shape[0] == B
    expected_n_blocks = (T + L - 1) // L  # ceil(T/L)
    assert scores.shape[1] == expected_n_blocks, (
        f"Expected {expected_n_blocks} blocks, got {scores.shape[1]}"
    )


def test_esg_masks_padding():
    """All-zero (padded) blocks should not be selected if valid blocks exist."""
    from model import EventSaliencyGate
    B, T, D = 1, 60, 96
    L, M = 15, 2
    gate = EventSaliencyGate(input_dim=D, block_size=L, top_m=M)
    x = torch.zeros(B, T, D)  # mostly padding
    x[:, :15] = torch.randn(B, 15, D)  # only first block is valid
    sel_frames, sel_indices = gate(x)
    # Check that selected indices come from the valid region
    # (at least some should be < 15)
    has_valid = (sel_indices[0] < 15).any()
    assert has_valid, "Valid block should be selected over padding"


def test_esg_top_m_capped_by_available_blocks():
    """When T is small, top_m is capped at available block count."""
    from model import EventSaliencyGate
    B, T, D = 1, 20, 64  # only ~2 blocks at L=15
    L, M = 15, 8  # request more blocks than available
    gate = EventSaliencyGate(input_dim=D, block_size=L, top_m=M)
    x = torch.randn(B, T, D)
    sel_frames, sel_indices = gate(x)
    # Should not crash; output tokens may be < M*L
    assert sel_frames.shape[0] == B
    assert sel_frames.shape[2] == D
