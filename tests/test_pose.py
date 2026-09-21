"""Tests for pose extraction shapes and conventions."""

import numpy as np
import pytest


def test_normalise_shape():
    """normalise() preserves shape."""
    from preprocess import normalise
    kp = np.random.randn(50, 33, 2).astype(np.float32)
    # Make some frames valid
    kp[:30, 23] = np.array([0.5, 0.6])  # left hip
    kp[:30, 24] = np.array([0.4, 0.6])  # right hip
    kp[:30, 11] = np.array([0.3, 0.3])  # left shoulder
    kp[:30, 12] = np.array([0.6, 0.3])  # right shoulder
    out = normalise(kp)
    assert out.shape == (50, 33, 2)
    assert out.dtype == np.float32


def test_normalise_zero_frames_unchanged():
    """normalise() leaves all-zero frames as zero."""
    from preprocess import normalise
    kp = np.zeros((10, 33, 2), dtype=np.float32)
    out = normalise(kp)
    np.testing.assert_array_equal(out, kp)


def test_pad_or_truncate_short():
    """Short sequence padded to T_MAX with zeros."""
    from preprocess import pad_or_truncate, T_MAX
    kp = np.ones((50, 33, 2), dtype=np.float32)
    out = pad_or_truncate(kp)
    assert out.shape == (T_MAX, 33, 2)
    assert np.all(out[:50] == 1.0)
    assert np.all(out[50:] == 0.0)


def test_pad_or_truncate_long():
    """Long sequence truncated to T_MAX."""
    from preprocess import pad_or_truncate, T_MAX
    kp = np.ones((500, 33, 2), dtype=np.float32)
    out = pad_or_truncate(kp)
    assert out.shape == (T_MAX, 33, 2)


def test_pad_or_truncate_exact():
    """Sequence of exactly T_MAX frames unchanged."""
    from preprocess import pad_or_truncate, T_MAX
    kp = np.ones((T_MAX, 33, 2), dtype=np.float32)
    out = pad_or_truncate(kp)
    assert out.shape == (T_MAX, 33, 2)
    np.testing.assert_array_equal(out, kp)
