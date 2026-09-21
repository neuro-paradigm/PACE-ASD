"""
PACE-ASD — Bilateral Movement Asymmetry Utility

Provides descriptive bilateral movement asymmetry measures derived from
left/right MediaPipe Pose landmark pairs.

IMPORTANT: These measures are descriptive movement analysis features.
They are NOT validated ASD biomarkers and must not be presented as such.
Terminology: 'bilateral movement asymmetry', 'left-right movement difference'.
"""

import numpy as np


# Left/right landmark pairs (MediaPipe Pose 33-landmark set)
# Format: (name, left_index, right_index)
LR_PAIRS = [
    ("shoulder", 11, 12),
    ("elbow",    13, 14),
    ("wrist",    15, 16),
    ("pinky",    17, 18),
    ("index",    19, 20),
    ("thumb",    21, 22),
    ("hip",      23, 24),
    ("knee",     25, 26),
    ("ankle",    27, 28),
    ("heel",     29, 30),
    ("foot",     31, 32),
]


def compute_bilateral_asymmetry(positions: np.ndarray,
                                 compute_velocity_asymmetry: bool = True) -> dict:
    """
    Compute descriptive bilateral movement asymmetry for each left/right joint pair.

    The asymmetry measure is defined as the mean absolute difference between
    mirrored left and right joint positions over the valid (non-zero) frames.
    Mirroring: right X-coordinate is negated before comparison to account for
    the body-centred coordinate system.

    This is a DESCRIPTIVE MOVEMENT MEASURE, not an ASD biomarker.

    Args:
        positions: (T, 33, 2) float32 — normalised pose sequence
        compute_velocity_asymmetry: if True, also compute velocity asymmetry

    Returns:
        dict with:
            pairs: {joint_name: position_asymmetry_value}
            mean_asymmetry: mean across all pairs
            velocity_pairs: {joint_name: velocity_asymmetry_value}  (if requested)
            mean_velocity_asymmetry: mean velocity asymmetry
            valid_frame_count: int
            note: disclaimer string
    """
    T = positions.shape[0]
    valid_mask = np.abs(positions).sum(axis=(1, 2)) > 1e-4  # (T,)
    valid_pos  = positions[valid_mask]  # (T_valid, 33, 2)
    n_valid    = len(valid_pos)

    result = {
        "valid_frame_count": n_valid,
        "note": (
            "Descriptive bilateral movement asymmetry (left-right movement difference). "
            "Not a validated ASD biomarker."
        ),
    }

    if n_valid < 2:
        result["pairs"] = {name: 0.0 for name, _, _ in LR_PAIRS}
        result["mean_asymmetry"] = 0.0
        if compute_velocity_asymmetry:
            result["velocity_pairs"] = {name: 0.0 for name, _, _ in LR_PAIRS}
            result["mean_velocity_asymmetry"] = 0.0
        return result

    # Position asymmetry
    pos_asym = {}
    for name, l_idx, r_idx in LR_PAIRS:
        left_p  = valid_pos[:, l_idx, :]    # (T_valid, 2)
        right_p = valid_pos[:, r_idx, :]    # (T_valid, 2)
        # Mirror right X to make comparable after hip-centering
        right_mirrored = right_p.copy()
        right_mirrored[:, 0] = -right_mirrored[:, 0]
        pos_asym[name] = round(float(np.abs(left_p - right_mirrored).mean()), 6)

    result["pairs"]         = pos_asym
    result["mean_asymmetry"] = round(float(np.mean(list(pos_asym.values()))), 6)

    # Velocity asymmetry (optional)
    if compute_velocity_asymmetry:
        # Compute velocity from valid positions
        vel_valid = np.zeros_like(valid_pos)
        vel_valid[1:] = (valid_pos[1:] - valid_pos[:-1]) * 10.0

        vel_asym = {}
        for name, l_idx, r_idx in LR_PAIRS:
            left_v  = vel_valid[:, l_idx, :]
            right_v = vel_valid[:, r_idx, :]
            right_v_mirrored = right_v.copy()
            right_v_mirrored[:, 0] = -right_v_mirrored[:, 0]
            vel_asym[name] = round(float(np.abs(left_v - right_v_mirrored).mean()), 6)

        result["velocity_pairs"]          = vel_asym
        result["mean_velocity_asymmetry"] = round(float(np.mean(list(vel_asym.values()))), 6)

    return result


def asymmetry_timeseries(positions: np.ndarray,
                          joint_pair: str = "wrist") -> np.ndarray:
    """
    Compute frame-by-frame bilateral asymmetry for one joint pair.

    Useful for time-series visualization of how asymmetry evolves.

    Args:
        positions:  (T, 33, 2) float32
        joint_pair: name of the pair (must be in LR_PAIRS)

    Returns:
        timeseries: (T,) float32 — per-frame L/R distance; 0.0 for invalid frames
    """
    pair_map = {name: (l, r) for name, l, r in LR_PAIRS}
    if joint_pair not in pair_map:
        raise ValueError(f"Unknown joint pair '{joint_pair}'. "
                         f"Valid: {list(pair_map.keys())}")

    l_idx, r_idx = pair_map[joint_pair]
    T = positions.shape[0]
    valid_mask = np.abs(positions).sum(axis=(1, 2)) > 1e-4

    timeseries = np.zeros(T, dtype=np.float32)
    for t in range(T):
        if not valid_mask[t]:
            continue
        left_p  = positions[t, l_idx, :].copy()
        right_p = positions[t, r_idx, :].copy()
        right_p[0] = -right_p[0]  # mirror X
        timeseries[t] = float(np.linalg.norm(left_p - right_p))

    return timeseries
