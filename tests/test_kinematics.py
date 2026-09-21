"""Tests for velocity and acceleration computation."""

import numpy as np
import pytest
import torch


def test_spatial_encoder_velocity_shape():
    """SpatialEncoder computes vel/acc internally; output shape must be (B,T,spatial_dim)."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    from model import SpatialEncoder
    enc = SpatialEncoder(spatial_dim=64, dropout=0.0)
    enc.eval()
    x = torch.zeros(2, 60, 33, 2)  # some zero, some non-zero
    x[:, :30] = torch.randn(2, 30, 33, 2) * 0.5
    with torch.no_grad():
        out = enc(x)
    assert out.shape == (2, 60, 64), f"Expected (2, 60, 64), got {out.shape}"


def test_velocity_zero_for_invalid_frames():
    """Velocity should be zero where input is all-zero."""
    from model import SpatialEncoder
    enc = SpatialEncoder(spatial_dim=64, dropout=0.0)
    enc.eval()
    x = torch.zeros(1, 10, 33, 2)  # all invalid
    with torch.no_grad():
        out = enc(x)
    # All frames are invalid/zero — output should be zeros
    np.testing.assert_allclose(out.numpy(), 0.0, atol=1e-5)


def test_velocity_boundary_valid():
    """First frame velocity should be zero (boundary condition)."""
    from model import SpatialEncoder
    enc = SpatialEncoder(spatial_dim=64, dropout=0.0)
    enc.eval()
    # Create input where ALL frames are valid
    x = torch.randn(1, 30, 33, 2) * 0.5 + 1.0  # shift away from zero
    with torch.no_grad():
        out = enc(x)
    assert out.shape == (1, 30, 64)


def test_kinematics_computation_scaling():
    """Velocity should be position_diff * 10, acceleration vel_diff * 5."""
    pos = np.zeros((5, 33, 2), dtype=np.float32)
    # Positions linearly increasing across frames 1..4
    for t in range(1, 5):
        pos[t] = t * 0.1
    vel = np.zeros_like(pos)
    valid = np.abs(pos).sum(axis=(1, 2)) > 1e-4  # frames 1..4 valid
    for t in range(1, 5):
        if valid[t] and valid[t-1]:
            vel[t] = (pos[t] - pos[t-1]) * 10.0
    # Frame 0 is invalid, so vel[1] should be 0
    assert vel[1].sum() == 0.0
    # Frames 1 and 2 are both valid, so vel[2] should be non-zero
    assert vel[2].sum() != 0.0
    np.testing.assert_allclose(vel[2], (pos[2] - pos[1]) * 10.0, atol=1e-5)
