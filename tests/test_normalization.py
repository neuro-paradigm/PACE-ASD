"""Tests for landmark centering and scale normalization."""

import numpy as np
import pytest


def test_hip_centering():
    """After normalisation, mid-hip should be at origin."""
    from preprocess import normalise
    kp = np.zeros((5, 33, 2), dtype=np.float32)
    kp[:, :, :] = 0.01  # non-zero
    kp[:, 23] = np.array([0.4, 0.5])  # left hip
    kp[:, 24] = np.array([0.6, 0.5])  # right hip
    kp[:, 11] = np.array([0.3, 0.3])  # left shoulder
    kp[:, 12] = np.array([0.7, 0.3])  # right shoulder
    out = normalise(kp)
    # After centering: mid-hip should be ~(0, 0) for each valid frame
    mid_hip = (out[:, 23] + out[:, 24]) / 2.0
    np.testing.assert_allclose(mid_hip, 0.0, atol=1e-5)


def test_scale_normalization():
    """After normalisation, inter-shoulder distance should be ~1.0."""
    from preprocess import normalise
    import numpy.linalg as la
    kp = np.zeros((5, 33, 2), dtype=np.float32)
    kp[:, :, :] = 0.01
    kp[:, 23] = np.array([0.4, 0.5])
    kp[:, 24] = np.array([0.6, 0.5])
    kp[:, 11] = np.array([0.3, 0.3])  # left shoulder
    kp[:, 12] = np.array([0.7, 0.3])  # right shoulder
    out = normalise(kp)
    for t in range(5):
        sd = float(la.norm(out[t, 11] - out[t, 12]))
        np.testing.assert_allclose(sd, 1.0, atol=1e-5)


def test_normalization_dtype():
    """Output dtype must be float32."""
    from preprocess import normalise
    kp = np.random.randn(10, 33, 2).astype(np.float32)
    kp[:, 23] = [0.4, 0.5]; kp[:, 24] = [0.6, 0.5]
    kp[:, 11] = [0.3, 0.3]; kp[:, 12] = [0.7, 0.3]
    out = normalise(kp)
    assert out.dtype == np.float32


def test_pixel_normalisation_is_invariant_to_cropping():
    """The same body in two differently cropped frames gives the same array when
    coordinates are converted to pixels; per-axis normalisation does not."""
    import numpy as np
    from preprocess import normalise
    rng = np.random.default_rng(0)
    px = rng.uniform(300, 700, (5, 33, 2))                  # pixel positions in a 1080 x 1920 frame
    full = px / np.array([1080, 1920])
    crop_px = px - np.array([200, 400])                      # same body, frame cropped to 760 x 1300
    crop = crop_px / np.array([760, 1300])
    a = normalise(full.astype(np.float32), 1080, 1920)
    b = normalise(crop.astype(np.float32), 760, 1300)
    np.testing.assert_allclose(a, b, atol=1e-4)
    assert np.abs(normalise(full.astype(np.float32)) - normalise(crop.astype(np.float32))).max() > 1e-2
