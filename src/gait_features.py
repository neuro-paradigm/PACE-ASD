"""
PACE-ASD — hand-crafted gait and upper-body movement descriptors.

Computed from a prepared sequence (mid-hip centred, shoulder-scaled image
coordinates; x to the right, y downwards; the child walks towards the
camera). Every descriptor uses detected frames only; internal detection gaps
are bridged by linear interpolation where a continuous signal is needed.
Durations and counts of frames are deliberately excluded: they are reported
separately as the clip-duration and recording-pattern controls.

Temporal descriptors are expressed per frame; multiplying by the frame rate
converts them to per-second units without changing any classifier.
"""

import numpy as np

from sequence import fill_internal_gaps, valid_mask

NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 0, 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 23, 24, 25, 26, 27, 28


def _angle(a, b, c):
    """Angle at b (radians) between segments b-a and b-c, per frame."""
    u, v = a - b, c - b
    cos = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-8)
    return np.arccos(np.clip(cos, -1, 1))


def _robust_range(x):
    return float(np.percentile(x, 95) - np.percentile(x, 5))


def _asym(a, b):
    return float(abs(a - b) / (abs(a) + abs(b) + 1e-8))


def _smooth(x, k=5):
    if len(x) < k:
        return x
    return np.convolve(x, np.ones(k) / k, mode="same")


def _dominant_lag(x, lo=8, hi=None):
    """Lag and height of the highest local maximum of the autocorrelation in
    [lo, hi] (the stride period and the stride regularity of Moe-Nilssen and
    Helbostad, 2004). NaN when the autocorrelation has no local maximum
    there, which the classifiers impute from the training children."""
    x = x - x.mean()
    n = len(x)
    hi = min(hi or (2 * n) // 3, n - 2)
    if hi <= lo or x.std() < 1e-8:
        return np.nan, np.nan
    ac = np.correlate(x, x, mode="full")[n - 1:] / (x.var() * n)
    peaks = [j for j in range(lo, hi + 1) if ac[j - 1] < ac[j] >= ac[j + 1]]
    if not peaks:
        return np.nan, np.nan
    j = max(peaks, key=lambda q: ac[q])
    return float(j), float(ac[j])


def gait_features(seq: np.ndarray) -> dict:
    vm = valid_mask(seq)
    if vm.sum() < 10:
        return {k: np.nan for k in FEATURE_NAMES}
    idx = np.flatnonzero(vm)
    s = fill_internal_gaps(seq)[idx[0]: idx[-1] + 1]          # continuous span
    v = seq[vm]                                               # detected frames only

    # step signal: vertical separation of the ankles alternates with each step
    step = _smooth(s[:, L_AN, 1] - s[:, R_AN, 1])
    sc = step - np.median(step)
    crossings = int(np.sum(np.diff(np.sign(sc[np.abs(sc) > 0.02])) != 0))
    lag, reg = _dominant_lag(step)

    sh_mid = (v[:, L_SH] + v[:, R_SH]) / 2
    arm_l = v[:, L_WR, 1] - v[:, L_SH, 1]
    arm_r = v[:, R_WR, 1] - v[:, R_SH, 1]
    knee_l = _angle(v[:, L_HIP], v[:, L_KN], v[:, L_AN])
    knee_r = _angle(v[:, R_HIP], v[:, R_KN], v[:, R_AN])
    elb_l = _angle(v[:, L_SH], v[:, L_EL], v[:, L_WR])
    elb_r = _angle(v[:, R_SH], v[:, R_EL], v[:, R_WR])
    tilt = np.arctan2(v[:, R_SH, 1] - v[:, L_SH, 1], v[:, R_SH, 0] - v[:, L_SH, 0] + 1e-8)
    trunk = np.arctan2(sh_mid[:, 0], -sh_mid[:, 1] + 1e-8)

    # frame-to-frame speed and jerk, only between consecutive detected frames
    pair = vm[1:] & vm[:-1]
    d1 = np.linalg.norm(seq[1:] - seq[:-1], axis=-1)[pair]                   # (P, 33)
    trip = vm[3:] & vm[2:-1] & vm[1:-2] & vm[:-3]
    jerk = np.linalg.norm(seq[3:] - 3 * seq[2:-1] + 3 * seq[1:-2] - seq[:-3], axis=-1)[trip]

    def mean_speed(j):
        return float(d1[:, j].mean()) if len(d1) else np.nan

    def mean_jerk(j):
        return float(jerk[:, j].mean()) if len(jerk) else np.nan

    f = {
        "step_rate": crossings / len(s),
        "stride_lag": lag,
        "stride_regularity": reg,
        "step_amplitude": _robust_range(step),
        "step_width": float(np.median(np.abs(v[:, L_AN, 0] - v[:, R_AN, 0]))),
        "step_width_var": float(np.std(np.abs(v[:, L_AN, 0] - v[:, R_AN, 0]))),
        "arm_swing": (_robust_range(arm_l) + _robust_range(arm_r)) / 2,
        "arm_swing_asym": _asym(_robust_range(arm_l), _robust_range(arm_r)),
        "wrist_height": float(np.median(-(arm_l + arm_r) / 2)),
        "wrist_lateral": float((np.std(v[:, L_WR, 0] - v[:, L_SH, 0]) +
                                np.std(v[:, R_WR, 0] - v[:, R_SH, 0])) / 2),
        "elbow_rom": (_robust_range(elb_l) + _robust_range(elb_r)) / 2,
        "elbow_flexion": float(np.pi - np.median(np.concatenate([elb_l, elb_r]))),
        "knee_rom": (_robust_range(knee_l) + _robust_range(knee_r)) / 2,
        "knee_rom_asym": _asym(_robust_range(knee_l), _robust_range(knee_r)),
        "knee_flexion": float(np.pi - np.median(np.concatenate([knee_l, knee_r]))),
        "trunk_sway": float(np.std(sh_mid[:, 0])),
        "trunk_lean": float(np.median(trunk)),
        "shoulder_tilt_var": float(np.std(tilt)),
        "head_sway": float(np.std(v[:, NOSE, 0] - sh_mid[:, 0])),
        "head_bob": float(np.std(v[:, NOSE, 1] - sh_mid[:, 1])),
        "hip_width": float(np.median(np.abs(v[:, L_HIP, 0] - v[:, R_HIP, 0]))),
        "speed_wrists": (mean_speed(L_WR) + mean_speed(R_WR)) / 2,
        "speed_ankles": (mean_speed(L_AN) + mean_speed(R_AN)) / 2,
        "speed_head": mean_speed(NOSE),
        "jerk_wrists": (mean_jerk(L_WR) + mean_jerk(R_WR)) / 2,
        "jerk_ankles": (mean_jerk(L_AN) + mean_jerk(R_AN)) / 2,
    }
    return f


FEATURE_NAMES = [
    "step_rate", "stride_lag", "stride_regularity", "step_amplitude", "step_width",
    "step_width_var", "arm_swing", "arm_swing_asym", "wrist_height", "wrist_lateral",
    "elbow_rom", "elbow_flexion", "knee_rom", "knee_rom_asym", "knee_flexion",
    "trunk_sway", "trunk_lean", "shoulder_tilt_var", "head_sway", "head_bob",
    "hip_width", "speed_wrists", "speed_ankles", "speed_head", "jerk_wrists", "jerk_ankles",
]


def feature_matrix(seqs) -> np.ndarray:
    return np.array([[gait_features(s)[k] for k in FEATURE_NAMES] for s in seqs], float)
