"""
PACE — depth-camera skeletons of the demonstration dataset.

Besides the color videos, the Dryad deposit contains for every child a
skeleton recorded by the depth camera (`<child>/video/8.xlsx`): one row per
frame with a time stamp (H:M:S:ms) and the 3-D camera-space position (metres;
x lateral, y up, z away from the camera) of 25 joints. The same device,
distance and processing were used for both groups, so the frame-format
difference of the color videos cannot occur in these data.

  read_skeleton        time stamps (s) and joints (T, 25, 3) of one file
  skeleton_features    gait descriptors fixed in docs/validation_criteria.md (B5)
  recording_features   properties of the recording that are not movement

The workbook is read with the standard library only.
"""

import re
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
from scipy.ndimage import uniform_filter1d

JOINTS = ["Midspain", "AnkleLeft", "AnkleRight", "ElbowLeft", "ElbowRight", "FootLeft",
          "FootRight", "HandLeft", "HandRight", "HandTipLeft", "HandTipRight", "Head",
          "HipLeft", "HipRight", "KneeLeft", "KneeRight", "Neck", "ShoulderLeft",
          "ShoulderRight", "SpineBase", "SpineShoulder", "ThumbLeft", "ThumbRight",
          "WristLeft", "WristRight"]
J = {n: i for i, n in enumerate(JOINTS)}
FEATURE_NAMES = ["step_width", "speed_per_leg", "arm_swing", "wrist_height", "knee_rom",
                 "elbow_rom", "trunk_sway", "head_bob", "jitter"]
RECORDING_NAMES = ["n_frames", "mean_distance", "initial_distance", "interval_irregularity",
                   "time_of_day"]
_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _col(ref):
    letters = re.match(r"[A-Z]+", ref).group(0)
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch) - 64
    return n - 1


def read_rows(path):
    """Rows of the first worksheet as lists (None for empty cells)."""
    z = zipfile.ZipFile(path)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", _NS):
            shared.append("".join(t.text or "" for t in si.iter(f"{{{_NS['m']}}}t")))
    rows = []
    for row in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter(f"{{{_NS['m']}}}row"):
        cells = {}
        for c in row.findall("m:c", _NS):
            v = c.find("m:v", _NS)
            if v is None:
                continue
            val = shared[int(v.text)] if c.get("t") == "s" else float(v.text)
            cells[_col(c.get("r"))] = val
        rows.append([cells.get(i) for i in range(max(cells) + 1)] if cells else [])
    return rows


def _seconds(stamp):
    h, m, s, ms = (int(p) for p in str(stamp).split(":"))
    return h * 3600 + m * 60 + s + ms / 1000


def read_skeleton(path):
    rows = read_rows(path)
    header = rows[0]
    cols = {}
    for i, name in enumerate(header):
        if isinstance(name, str) and name in J:
            cols[name] = i
    missing = set(JOINTS) - set(cols)
    if missing:
        raise ValueError(f"{path}: joints missing {sorted(missing)}")
    times, frames = [], []
    for r in rows[1:]:
        if not r or not isinstance(r[0], str) or not re.match(r"\d+:\d+:\d+:\d+", r[0]):
            continue
        try:
            xyz = [[float(r[cols[n] + k]) for k in range(3)] for n in JOINTS]
        except (TypeError, IndexError):
            continue
        times.append(_seconds(r[0]))
        frames.append(xyz)
    return np.array(times), np.array(frames)


def _angle(a, b, c):
    u, v = a - b, c - b
    cos = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-8)
    return np.arccos(np.clip(cos, -1, 1))


def _rrange(x):
    return float(np.percentile(x, 95) - np.percentile(x, 5))


def skeleton_features(times, X, jitter_window=5):
    """Descriptors of B5 (lengths divided by the shoulder width). The jitter
    window of 5 frames at about 30 frames per second matches in time the
    9-frame window used for the 60-frames-per-second videos."""
    g = lambda n: X[:, J[n]]
    sw = float(np.median(np.linalg.norm(g("ShoulderLeft") - g("ShoulderRight"), axis=-1)))
    leg = float(np.median((np.linalg.norm(g("HipLeft") - g("AnkleLeft"), axis=-1) +
                           np.linalg.norm(g("HipRight") - g("AnkleRight"), axis=-1)) / 2))
    t = times - times[0]
    slope = np.polyfit(t, g("SpineBase")[:, 2], 1)[0] if len(t) > 2 and t[-1] > 0 else np.nan
    knee = [_angle(g(f"Hip{s}"), g(f"Knee{s}"), g(f"Ankle{s}")) for s in ("Left", "Right")]
    elb = [_angle(g(f"Shoulder{s}"), g(f"Elbow{s}"), g(f"Wrist{s}")) for s in ("Left", "Right")]
    resid = X - uniform_filter1d(X, size=min(jitter_window, len(X)), axis=0, mode="nearest")
    return {
        "step_width": float(np.median(np.abs(g("AnkleLeft")[:, 0] - g("AnkleRight")[:, 0]))) / sw,
        "speed_per_leg": float(-slope / leg),
        "arm_swing": np.mean([_rrange(g(f"Wrist{s}")[:, 2] - g(f"Shoulder{s}")[:, 2])
                              for s in ("Left", "Right")]) / sw,
        "wrist_height": float(np.median((g("WristLeft")[:, 1] - g("ShoulderLeft")[:, 1] +
                                         g("WristRight")[:, 1] - g("ShoulderRight")[:, 1]) / 2)) / sw,
        "knee_rom": float(np.mean([_rrange(k) for k in knee])),
        "elbow_rom": float(np.mean([_rrange(e) for e in elb])),
        "trunk_sway": float(np.std(g("SpineShoulder")[:, 0] - g("SpineBase")[:, 0])) / sw,
        "head_bob": float(np.std(g("Head")[:, 1] - g("SpineShoulder")[:, 1])) / sw,
        "jitter": float(np.median(np.sqrt((np.linalg.norm(resid, axis=-1) ** 2).mean(0)))) / sw,
    }


def recording_features(times, X):
    dt = np.diff(times)
    z = X[:, J["SpineBase"], 2]
    return {"n_frames": float(len(times)), "mean_distance": float(z.mean()),
            "initial_distance": float(z[0]),
            "interval_irregularity": float(dt.std() / dt.mean()) if len(dt) > 1 else 0.0,
            "time_of_day": float(times[0] / 3600)}
