"""Tests for bilateral movement asymmetry utility."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_asymmetry_output_structure():
    """compute_bilateral_asymmetry returns expected keys."""
    from asymmetry import compute_bilateral_asymmetry
    pos = np.random.randn(300, 33, 2).astype(np.float32) * 0.5 + 0.01
    result = compute_bilateral_asymmetry(pos)
    assert "pairs" in result
    assert "mean_asymmetry" in result
    assert "note" in result
    assert "velocity_pairs" in result
    assert "mean_velocity_asymmetry" in result
    assert "valid_frame_count" in result


def test_asymmetry_zero_input():
    """All-zero positions give zero asymmetry."""
    from asymmetry import compute_bilateral_asymmetry
    pos = np.zeros((300, 33, 2), dtype=np.float32)
    result = compute_bilateral_asymmetry(pos)
    assert result["mean_asymmetry"] == 0.0


def test_asymmetry_symmetric_body_is_zero():
    """Perfectly symmetric body (left X = -right X) gives near-zero asymmetry."""
    from asymmetry import compute_bilateral_asymmetry, LR_PAIRS
    pos = np.zeros((300, 33, 2), dtype=np.float32)
    # Make first 100 frames valid and symmetric
    for t in range(100):
        pos[t, :, :] = 0.05  # baseline non-zero
        for _, l_idx, r_idx in LR_PAIRS:
            pos[t, l_idx]  = [ 0.3, 0.2]  # left X = +0.3
            pos[t, r_idx]  = [-0.3, 0.2]  # right X = -0.3 (mirrored)
    result = compute_bilateral_asymmetry(pos)
    for name, val in result["pairs"].items():
        assert val < 0.01, f"{name} asymmetry should be ~0 for symmetric body, got {val}"


def test_asymmetry_note_is_not_biomarker():
    """Asymmetry note must NOT claim it is an ASD biomarker."""
    from asymmetry import compute_bilateral_asymmetry
    pos = np.random.randn(10, 33, 2).astype(np.float32) + 0.1
    result = compute_bilateral_asymmetry(pos)
    note = result["note"].lower()
    assert "validated asd biomarker" not in note or "not" in note


def test_asymmetry_timeseries_shape():
    """asymmetry_timeseries returns (T,) array."""
    from asymmetry import asymmetry_timeseries
    pos = np.random.randn(300, 33, 2).astype(np.float32) + 0.1
    ts = asymmetry_timeseries(pos, joint_pair="wrist")
    assert ts.shape == (300,)
    assert ts.dtype == np.float32


def test_asymmetry_timeseries_invalid_pair():
    """asymmetry_timeseries raises ValueError for unknown joint pair."""
    from asymmetry import asymmetry_timeseries
    pos = np.ones((300, 33, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        asymmetry_timeseries(pos, joint_pair="nonexistent")
