"""Hand-crafted gait descriptors on synthetic walking-like sequences."""

import numpy as np

from gait_features import FEATURE_NAMES, feature_matrix, gait_features


def walker(n=120, stride=60, step_width=0.4, lead=0, T=300):
    """Frontal-view caricature: ankles alternate vertically with the stride,
    wrists swing in counter-phase."""
    t = np.arange(n)
    s = np.zeros((T, 33, 2), np.float32)
    base = np.zeros((33, 2), np.float32)
    base[11], base[12] = (-0.5, -2.0), (0.5, -2.0)          # shoulders
    base[23], base[24] = (-0.25, 0.0), (0.25, 0.0)          # hips
    base[25], base[26] = (-0.25, 1.0), (0.25, 1.0)          # knees
    base[27], base[28] = (-step_width / 2, 2.0), (step_width / 2, 2.0)
    base[13], base[14] = (-0.6, -1.2), (0.6, -1.2)
    base[15], base[16] = (-0.6, -0.5), (0.6, -0.5)
    base[0] = (0.0, -2.6)
    ph = 2 * np.pi * t / stride
    for i in range(n):
        f = base.copy()
        f[27, 1] += 0.2 * np.sin(ph[i]); f[28, 1] -= 0.2 * np.sin(ph[i])
        f[15, 1] -= 0.15 * np.sin(ph[i]); f[16, 1] += 0.15 * np.sin(ph[i])
        s[lead + i] = f
    return s


def test_all_features_present_and_finite():
    f = gait_features(walker())
    assert list(f) == FEATURE_NAMES
    assert all(np.isfinite(v) for v in f.values())


def test_stride_period_recovered():
    for stride in (40, 60):
        assert abs(gait_features(walker(n=180, stride=stride))["stride_lag"] - stride) <= 2


def test_step_width_and_onset_invariance():
    a = gait_features(walker(step_width=0.4))
    b = gait_features(walker(step_width=0.8, lead=30))
    assert b["step_width"] > a["step_width"]
    c = gait_features(walker(step_width=0.4, lead=30))
    for k in FEATURE_NAMES:
        assert np.isclose(a[k], c[k], equal_nan=True), k


def test_feature_matrix_shape():
    X = feature_matrix([walker(), walker(stride=50)])
    assert X.shape == (2, len(FEATURE_NAMES))
