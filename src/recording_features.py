"""
PACE — properties of a recording that are not the participant's movement.

  jitter_features    high-frequency landmark noise of a prepared sequence
                     (the noise floor of the pose estimator)
  smooth_sequence    a prepared sequence low-pass filtered within detected
                     stretches (used to check descriptors against the noise
                     floor)
  scene_features     image properties of the video itself: luminance,
                     contrast, saturation, sharpness, background luminance,
                     sharpness of the (blurred) face region, bit rate and the
                     child's size in the frame

These feed the jitter and scene/camera controls of the validation protocol
(docs/validation_criteria.md, B4c-d); none of them describes movement.

The 9-frame residual of jitter_features keeps movement above about 3 Hz at
60 frames per second, so it measures high-frequency landmark motion rather
than estimator noise. The third phase (docs/validation_criteria_phase3.md,
D5) adds measures of tracking quality:

  second_difference_noise     white-noise estimate from second differences
  segment_length_variability  high-frequency fluctuation of the lengths of
                              rigid limb segments
  (visibility, from the MediaPipe re-extraction of scripts/pose_qc.py)
"""

import os

import numpy as np
from scipy.ndimage import uniform_filter1d

from sequence import valid_mask

REGIONS = {"face": list(range(0, 11)), "arms_hands": list(range(11, 23)),
           "legs_feet": list(range(25, 33))}
JITTER_NAMES = ["jitter_all", "jitter_face", "jitter_arms_hands", "jitter_legs_feet"]
SCENE_NAMES = ["luminance", "contrast", "saturation", "sharpness", "border_luminance",
               "face_sharpness", "bitrate_kbps", "shoulder_px_rel_height"]


def _runs(vm, min_len):
    """(start, stop) of runs of consecutive True values at least min_len long."""
    out, start = [], None
    for t, v in enumerate(np.append(vm, False)):
        if v and start is None:
            start = t
        elif not v and start is not None:
            if t - start >= min_len:
                out.append((start, t))
            start = None
    return out


def smooth_sequence(seq: np.ndarray, window: int = 9) -> np.ndarray:
    """Moving average over `window` frames within every stretch of consecutive
    detected frames; undetected frames stay zero."""
    out = seq.copy()
    for a, b in _runs(valid_mask(seq), 1):
        out[a:b] = uniform_filter1d(seq[a:b], size=min(window, b - a), axis=0, mode="nearest")
    return out


def jitter_features(seq: np.ndarray, window: int = 9) -> dict:
    """Root mean square distance between each landmark trajectory and its
    `window`-frame moving average, over stretches of consecutive detected
    frames at least `window` frames long (units of the sequence: shoulder
    widths for prepared sequences); median over the landmarks of each
    region."""
    vm = valid_mask(seq)
    res = []
    for a, b in _runs(vm, window):
        seg = seq[a:b]
        res.append(np.linalg.norm(seg - uniform_filter1d(seg, size=window, axis=0, mode="nearest"),
                                  axis=-1))           # (n, J)
    if not res:
        return {k: np.nan for k in JITTER_NAMES}
    r = np.sqrt((np.concatenate(res) ** 2).mean(axis=0))   # (J,)
    out = {"jitter_all": float(np.median(r))}
    for name, idx in REGIONS.items():
        out[f"jitter_{name}"] = float(np.median(r[idx]))
    return out


TRACKING_NAMES = ["noise2_all", "noise2_face", "noise2_arms_hands", "noise2_legs_feet", "segment_length_var"]
SEGMENTS = [(11, 13), (12, 14), (13, 15), (14, 16), (23, 25), (24, 26), (25, 27), (26, 28)]


def second_difference_noise(seq: np.ndarray) -> dict:
    """Root mean square of x(t+1) - 2 x(t) + x(t-1) over triplets of
    detected frames, divided by sqrt(6): for white noise of standard
    deviation s on each coordinate this estimates s per coordinate direction
    (the norm of the 2-D vector is used, so a factor sqrt(2) for isotropic
    noise is included). Smooth movement contributes little: a 2 Hz sinusoid
    sampled at 60 frames per second has a second difference of 4% of its
    amplitude. Median over the landmarks of each region."""
    vm = valid_mask(seq)
    trip = vm[2:] & vm[1:-1] & vm[:-2]
    if trip.sum() < 3:
        return {k: np.nan for k in TRACKING_NAMES[:4]}
    d2 = np.linalg.norm(seq[2:] - 2 * seq[1:-1] + seq[:-2], axis=-1)[trip]      # (n, J)
    r = np.sqrt((d2 ** 2).mean(0)) / np.sqrt(6)
    out = {"noise2_all": float(np.median(r))}
    for name, idx in REGIONS.items():
        out[f"noise2_{name}"] = float(np.median(r[idx]))
    return out


def segment_length_variability(seq: np.ndarray, window: int = 9) -> float:
    """High-frequency fluctuation of the lengths of upper arms, forearms,
    thighs and shanks: root mean square difference between each segment's
    length and its `window`-frame moving average within detected stretches,
    divided by the segment's median length; median over the eight segments.
    A rigid segment's length changes only slowly with its orientation, so
    fast changes reflect tracking error."""
    runs = _runs(valid_mask(seq), window)
    if not runs:
        return np.nan
    vals = []
    for a, b in SEGMENTS:
        res, lens = [], []
        for s, e in runs:
            L = np.linalg.norm(seq[s:e, a] - seq[s:e, b], axis=-1)
            res.append(L - uniform_filter1d(L, size=window, mode="nearest"))
            lens.append(L)
        med = np.median(np.concatenate(lens))
        vals.append(np.sqrt((np.concatenate(res) ** 2).mean()) / max(med, 1e-6))
    return float(np.median(vals))


def tracking_features(seq: np.ndarray) -> dict:
    return {**second_difference_noise(seq), "segment_length_var": segment_length_variability(seq)}


def scene_features(video_path: str, keypoints: np.ndarray, fps: float, step: int = 10) -> dict:
    """Image statistics of every `step`-th frame of a video. `keypoints` are the
    raw image-normalized landmarks (T, 33, 2) of the same video, used to place
    the face box (centred on the nose, half-width half the shoulder width)."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    lum, con, sat, sharp, border, face, shoulder = [], [], [], [], [], [], []
    t = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if t % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lum.append(gray.mean()); con.append(gray.std()); sat.append(hsv[..., 1].mean())
            sharp.append(cv2.Laplacian(gray, cv2.CV_32F).var())
            bw, bh = max(1, int(0.05 * W)), max(1, int(0.05 * H))
            band = np.concatenate([gray[:bh].ravel(), gray[-bh:].ravel(),
                                   gray[:, :bw].ravel(), gray[:, -bw:].ravel()])
            border.append(band.mean())
            if t < len(keypoints) and np.any(keypoints[t] != 0):
                kp = keypoints[t] * np.array([W, H])
                sw = np.linalg.norm(kp[11] - kp[12])
                shoulder.append(sw / H)
                r = max(4, int(0.5 * sw))
                cx, cy = int(kp[0, 0]), int(kp[0, 1])
                box = gray[max(0, cy - r):min(H, cy + r), max(0, cx - r):min(W, cx + r)]
                if box.size > 16:
                    face.append(cv2.Laplacian(box, cv2.CV_32F).var())
        t += 1
    cap.release()
    seconds = t / fps if fps else np.nan
    return {"luminance": float(np.mean(lum)), "contrast": float(np.mean(con)),
            "saturation": float(np.mean(sat)), "sharpness": float(np.mean(sharp)),
            "border_luminance": float(np.mean(border)),
            "face_sharpness": float(np.median(face)) if face else np.nan,
            "bitrate_kbps": float(os.path.getsize(video_path) * 8 / 1000 / seconds),
            "shoulder_px_rel_height": float(np.median(shoulder)) if shoulder else np.nan}
