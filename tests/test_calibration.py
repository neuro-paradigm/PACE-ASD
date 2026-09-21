"""Tests for Platt scaling calibration."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_platt_scaler_fit_and_calibrate():
    """PlattScaler fits without error and returns probabilities in [0,1]."""
    from calibration import PlattScaler
    logits = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    labels = np.array([0, 0, 0, 1, 1], dtype=np.float32)
    scaler = PlattScaler()
    scaler.fit(logits, labels)
    probs = scaler.calibrate(logits)
    assert probs.shape == (5,)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_platt_scaler_rank_preservation():
    """Calibration must preserve rank order (AUC invariant)."""
    from calibration import PlattScaler
    import numpy.testing as npt
    logits = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float32)
    labels = np.array([0, 0, 1, 1], dtype=np.float32)
    scaler = PlattScaler()
    scaler.fit(logits, labels)
    probs = scaler.calibrate(logits)
    # Probabilities must be monotonically increasing
    for i in range(len(probs) - 1):
        assert probs[i] <= probs[i + 1], f"Rank not preserved at index {i}"


def test_platt_scaler_temperature_positive():
    """Temperature must remain strictly positive after fitting."""
    import torch
    from calibration import PlattScaler
    logits = np.linspace(-2, 2, 20).astype(np.float32)
    labels = (logits > 0).astype(np.float32)
    scaler = PlattScaler()
    scaler.fit(logits, labels)
    assert float(scaler.temperature.item()) > 0.0


def test_platt_scaler_small_dataset():
    """PlattScaler should not crash on very small datasets (n=4)."""
    from calibration import PlattScaler
    logits = np.array([-1.0, 1.0, -0.5, 0.5], dtype=np.float32)
    labels = np.array([0, 1, 0, 1], dtype=np.float32)
    scaler = PlattScaler()
    scaler.fit(logits, labels, max_iter=10)
    probs = scaler.calibrate(logits)
    assert probs.shape == (4,)
