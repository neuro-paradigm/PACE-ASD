"""
PACE-ASD — sequence preparation shared by every model, baseline and analysis.

Stored arrays are (T, 33, 2) float32: mid-hip centred, shoulder-scaled image
coordinates, with all-zero frames where no pose was detected or after the end
of the recording. Before any model sees a sequence it is prepared in two steps:

  1. reject_implausible: frames in which any normalised coordinate exceeds
     `max_abs` shoulder widths are set to zero (treated as undetected). Such
     values arise when the two shoulder landmarks nearly coincide in the image
     and the per-frame scale collapses.
  2. align_onset: the sequence is shifted so that its first detected frame is
     frame 0; the frames removed from the front are appended as zeros.

After preparation the content of a sequence is independent of how many
undetected frames preceded the first detection or followed the last one.
"""

import numpy as np

VALID_EPS = 1e-4          # a frame is valid when sum(|coords|) exceeds this
DEFAULT_MAX_ABS = 10.0    # plausibility bound, in shoulder widths


def valid_mask(seq: np.ndarray) -> np.ndarray:
    """(T, J, 2) -> (T,) bool, True where a pose was detected."""
    return np.abs(seq).sum(axis=(1, 2)) > VALID_EPS


def reject_implausible(seq: np.ndarray, max_abs: float = DEFAULT_MAX_ABS) -> np.ndarray:
    """Zero every frame with a normalised coordinate beyond +-max_abs."""
    if max_abs is None:
        return seq
    bad = np.abs(seq).max(axis=(1, 2)) > max_abs
    if not bad.any():
        return seq
    out = seq.copy()
    out[bad] = 0.0
    return out


def align_onset(seq: np.ndarray) -> np.ndarray:
    """Shift so the first valid frame is at index 0 (zeros appended)."""
    vm = valid_mask(seq)
    if not vm.any() or vm[0]:
        return seq
    first = int(np.argmax(vm))
    out = np.zeros_like(seq)
    out[: len(seq) - first] = seq[first:]
    return out


def prepare_sequence(seq: np.ndarray, max_abs: float | None = DEFAULT_MAX_ABS,
                     align: bool = True) -> np.ndarray:
    """Plausibility check followed (optionally) by onset alignment."""
    seq = reject_implausible(seq.astype(np.float32, copy=False), max_abs)
    return align_onset(seq) if align else seq


def fill_internal_gaps(seq: np.ndarray) -> np.ndarray:
    """Linearly interpolate undetected frames that lie between detections.
    Frames before the first or after the last detection are left at zero."""
    vm = valid_mask(seq)
    idx = np.flatnonzero(vm)
    if len(idx) < 2 or len(idx) == idx[-1] - idx[0] + 1:
        return seq
    out = seq.copy()
    span = np.arange(idx[0], idx[-1] + 1)
    flat = seq.reshape(len(seq), -1)
    filled = np.stack([np.interp(span, idx, flat[idx, k]) for k in range(flat.shape[1])], 1)
    out[span] = filled.reshape(len(span), *seq.shape[1:])
    return out


def mask_summary(seq: np.ndarray) -> dict:
    """Recording-pattern descriptors computed from the validity mask only.
    Used for the mask-pattern negative control; none of these describes
    movement."""
    vm = valid_mask(seq)
    T = len(vm)
    idx = np.flatnonzero(vm)
    if len(idx) == 0:
        return {k: 0.0 for k in MASK_FEATURES}
    first, last = int(idx[0]), int(idx[-1])
    span = last - first + 1
    inner = vm[first: last + 1]
    gap_runs, longest, cur = 0, 0, 0
    for v in inner:
        if not v:
            cur += 1
            longest = max(longest, cur)
        else:
            if cur:
                gap_runs += 1
            cur = 0
    gap_pos = np.flatnonzero(~inner)
    return {
        "n_valid": float(len(idx)),
        "first_valid": float(first),
        "last_valid": float(last),
        "span": float(span),
        "trailing_invalid": float(T - 1 - last),
        "gap_frames": float(span - len(idx)),
        "gap_runs": float(gap_runs),
        "longest_gap": float(longest),
        "gap_centroid": float(gap_pos.mean() / span) if len(gap_pos) else 0.5,
    }


MASK_FEATURES = ["n_valid", "first_valid", "last_valid", "span", "trailing_invalid",
                 "gap_frames", "gap_runs", "longest_gap", "gap_centroid"]
