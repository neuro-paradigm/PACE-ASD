"""Tests for the third-phase additions: tracking-quality measures, rotation-
robust scales and the planted arm-swing difference of the positive control."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))


def _walker(T=200, seed=0):
    """Synthetic prepared sequence: fixed skeleton plus a slow 1-Hz arm swing."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, (33, 2)).astype(np.float32)
    base[11], base[12] = [-0.5, -2.0], [0.5, -2.0]          # shoulders one unit apart
    base[23], base[24] = [-0.3, 0.0], [0.3, 0.0]
    seq = np.repeat(base[None], T, 0).astype(np.float32)
    t = np.arange(T) / 60.0
    for j in (13, 15, 17, 19, 21):
        seq[:, j, 1] += 0.3 * np.sin(2 * np.pi * t)
    for j in (14, 16, 18, 20, 22):
        seq[:, j, 1] -= 0.3 * np.sin(2 * np.pi * t)
    return seq


def test_second_difference_noise_estimates_white_noise():
    from recording_features import second_difference_noise
    rng = np.random.default_rng(1)
    s = 0.05
    seq = _walker(2000) + rng.normal(0, s, (2000, 33, 2)).astype(np.float32)
    est = second_difference_noise(seq)["noise2_all"]
    # the norm of 2-D noise is used, so the estimate is sqrt(2) times the per-coordinate SD
    assert abs(est - np.sqrt(2) * s) / (np.sqrt(2) * s) < 0.1


def test_second_difference_noise_ignores_slow_movement():
    from recording_features import second_difference_noise
    assert second_difference_noise(_walker(600))["noise2_all"] < 1e-3


def test_segment_length_variability_zero_for_rigid_segments():
    from recording_features import segment_length_variability
    seq = _walker(300)
    seq[:, 13:17] = seq[0, 13:17]                              # no arm movement
    assert segment_length_variability(seq) < 1e-6


def test_scale_variants_center_and_scale():
    from preprocess import normalise_scale_variant
    kp = np.full((40, 33, 2), 0.5, np.float32)
    kp[:, 11], kp[:, 12] = [0.40, 0.30], [0.60, 0.30]         # shoulders
    kp[:, 23], kp[:, 24] = [0.45, 0.60], [0.55, 0.60]         # hips
    for scale in ("torso", "shoulder_median"):
        out = normalise_scale_variant(kp, 100.0, 200.0, scale)
        np.testing.assert_allclose((out[:, 23] + out[:, 24]) / 2, 0.0, atol=1e-5)
    torso = normalise_scale_variant(kp, 100.0, 200.0, "torso")
    mid_sh = (torso[:, 11] + torso[:, 12]) / 2
    np.testing.assert_allclose(np.linalg.norm(mid_sh, axis=-1), 1.0, atol=1e-5)
    med = normalise_scale_variant(kp, 100.0, 200.0, "shoulder_median")
    np.testing.assert_allclose(np.linalg.norm(med[:, 11] - med[:, 12], axis=-1), 1.0, atol=1e-5)


def test_shoulder_median_scale_resists_single_frame_jumps():
    from preprocess import normalise_scale_variant
    kp = np.full((61, 33, 2), 0.5, np.float32)
    kp[:, 11], kp[:, 12] = [0.40, 0.30], [0.60, 0.30]
    kp[:, 23], kp[:, 24] = [0.45, 0.60], [0.55, 0.60]
    kp[30, 12] = [0.42, 0.30]                                  # shoulders nearly coincide in one frame
    out = normalise_scale_variant(kp, 100.0, 100.0, "shoulder_median")
    assert np.abs(out[30]).max() < 5                           # the per-frame scale would blow this frame up


def test_plant_arm_swing_identity_and_gain():
    from phase3_common import plant_arm_swing
    from gait_features import gait_features
    from sequence import valid_mask
    seq = _walker(300)
    seq[250:] = 0.0                                            # undetected tail
    np.testing.assert_array_equal(plant_arm_swing(seq, 1.0), seq)
    out = plant_arm_swing(seq, 2.0)
    np.testing.assert_array_equal(out[:, 11], seq[:, 11])     # shoulders unchanged
    np.testing.assert_array_equal(out[:, 0:11], seq[:, 0:11])  # face unchanged
    assert not out[~valid_mask(seq)].any()                     # undetected frames stay zero
    a0, a1 = gait_features(seq)["arm_swing"], gait_features(out)["arm_swing"]
    assert 1.8 < a1 / a0 < 2.2


def test_plant_arm_swing_keeps_high_frequency_residual():
    """The residual about the 15-frame moving average is kept exactly, and the
    second-difference noise of the arm landmarks hardly changes."""
    from phase3_common import plant_arm_swing
    from scipy.ndimage import uniform_filter1d
    rng = np.random.default_rng(3)
    seq = _walker(300) + rng.normal(0, 0.02, (300, 33, 2)).astype(np.float32)
    g = 3.0
    out = plant_arm_swing(seq, g)
    rel0 = seq[:, 15] - seq[:, 11]
    low = uniform_filter1d(rel0, 15, axis=0, mode="nearest")
    expected = low.mean(0) + g * (low - low.mean(0)) + (rel0 - low)
    np.testing.assert_allclose(out[:, 15] - out[:, 11], expected, atol=1e-5)
    arms = list(range(13, 23))
    d2 = lambda x: np.sqrt(((x[2:] - 2 * x[1:-1] + x[:-2]) ** 2).sum(-1).mean())
    assert d2(out[:, arms]) < 1.25 * d2(seq[:, arms])


def test_logreg_and_incremental_value_detect_added_information():
    from phase3_common import incremental_value
    rng = np.random.default_rng(4)
    n = 80
    y = np.r_[np.zeros(n // 2), np.ones(n // 2)]
    control = rng.normal(0, 1, (n, 1))
    movement = y * 2.0 + rng.normal(0, 1, n)
    res = incremental_value(y, control, movement, B=50)
    assert res["adds"] and res["loo_auc_with_movement"] > res["loo_auc_controls"]
    res0 = incremental_value(y, control, rng.normal(0, 1, n), B=50)
    assert not res0["adds"]
