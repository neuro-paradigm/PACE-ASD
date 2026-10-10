"""Tests for the fifth-phase additions: exact intervals of the calibration
shares, the outcome without a second channel and the projection of the
depth skeletons onto the image plane of the video."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


def test_clopper_pearson_known_values():
    from phase5_analyses import clopper_pearson
    lo, hi = clopper_pearson(0, 20)
    assert lo == 0.0 and abs(hi - 0.1684) < 1e-3
    lo, hi = clopper_pearson(0, 100)
    assert lo == 0.0 and abs(hi - 0.0362) < 1e-3
    lo, hi = clopper_pearson(13, 20)
    assert abs(lo - 0.408) < 2e-3 and abs(hi - 0.846) < 2e-3
    lo, hi = clopper_pearson(20, 20)
    assert hi == 1.0 and abs(lo - (1 - 0.1684)) < 1e-3


def _standing_child(n=31, z=2.5):
    """Kinect camera coordinates of a child facing the sensor: x to the
    sensor's left, y up, z away from the sensor."""
    import depth_skeleton as ds
    X = np.zeros((n, len(ds.JOINTS), 3))
    pos = {"Head": (0, 0.55, 0), "ShoulderLeft": (-0.15, 0.35, 0), "ShoulderRight": (0.15, 0.35, 0),
           "ElbowLeft": (-0.18, 0.12, 0), "ElbowRight": (0.18, 0.12, 0), "WristLeft": (-0.18, -0.1, 0),
           "WristRight": (0.18, -0.1, 0), "HipLeft": (-0.09, -0.1, 0), "HipRight": (0.09, -0.1, 0),
           "KneeLeft": (-0.09, -0.45, 0), "KneeRight": (0.09, -0.45, 0), "AnkleLeft": (-0.09, -0.8, 0),
           "AnkleRight": (0.09, -0.8, 0)}
    for name, p in pos.items():
        X[:, ds.J[name]] = p
    X[..., 2] += z
    return np.arange(n) / 30.0, X


def test_projected_depth_orientation_and_rate():
    from phase5_analyses import projected_depth
    t, X = _standing_child()
    a, L = projected_depth(t, X)
    assert a.shape == (300, 33, 2)
    assert L == 61                                   # 1 s at 60 frames per second
    v = a[:L]
    # the subject's left shoulder appears on the image right, the head above the hips (y downwards)
    assert np.all(v[:, 11, 0] > v[:, 12, 0])
    assert np.all(v[:, 0, 1] < 0) and np.all(v[:, 27, 1] > 0)
    # prepared as Eq. 1: hips centred, shoulder distance 1
    np.testing.assert_allclose((v[:, 23] + v[:, 24]) / 2, 0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(v[:, 11] - v[:, 12], axis=-1), 1, atol=1e-5)
    assert np.all(a[L:] == 0)


def test_projected_depth_descriptors_follow_arm_movement():
    from gait_features import FEATURE_NAMES, feature_matrix
    import depth_skeleton as ds
    from phase5_analyses import projected_depth
    t, X = _standing_child()
    swing = 0.05 * np.sin(2 * np.pi * t)[:, None]
    X1 = X.copy()
    for side in ("Left", "Right"):
        X1[:, ds.J[f"Wrist{side}"], 1] += swing[:, 0]
    still = feature_matrix([projected_depth(t, X)[0]])[0]
    moving = feature_matrix([projected_depth(t, X1)[0]])[0]
    k = FEATURE_NAMES.index("arm_swing")
    assert moving[k] > still[k] + 0.1
