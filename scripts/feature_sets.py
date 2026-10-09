"""
PACE — feature sets of the feature classifiers and controls, and one nested
cross-validation run of one classifier under any partition.

Used by run_feature_models.py (the third-phase rows), permutation_null.py,
positive_control.py and phase3_analyses.py, so that every run uses the same
definitions as the stored results. Feature kinds:

  gait, gait_demo, gait_torso, gait_median   gait descriptors (pixel units;
                         with age and sex for confound regression; with the
                         torso-length and median-shoulder scales of D7)
  mean_pose              mean prepared landmark positions (66 values; D9)
  duration, mask         detected frames; validity-pattern descriptors
  demo, format           age and sex; frame width, height, aspect, uncropped
  scene                  image properties of the video
  jitter                 high-frequency landmark motion (9-frame residual)
  tracking               second-difference noise, segment-length variability
                         and MediaPipe visibility (D5)
  depth, depth_recording depth-camera descriptors; recording properties
  time_of_day            time of day of the depth recording (D4)
  covariates             format, demographics and validity pattern
  all_controls           everything that is not movement (D4)
"""

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from gait_features import feature_matrix                    # noqa: E402
from sequence import MASK_FEATURES, mask_summary, prepare_sequence, valid_mask  # noqa: E402

CACHE = os.path.join(ROOT, "results", "phase3", "cache")
VIS_REGIONS = {"face": list(range(0, 11)), "arms_hands": list(range(11, 23)), "legs_feet": list(range(25, 33))}


def stored(s):
    return np.load(os.path.join(ROOT, "processed", "features", f"{s}.npy")).astype(np.float32)


def depth_path(raw_dir, s):
    group, n = s.split("_")
    folder = "Autism/children with ASD" if group == "asd" else "Typical"
    return os.path.join(raw_dir, folder, n, "video", "8.xlsx")


def raw_skeletons(subjects, raw_dir=None):
    """Time stamps and joints of the depth-camera skeletons, cached."""
    import depth_skeleton as ds
    path = os.path.join(CACHE, "depth_raw.npz")
    tab = dict(np.load(path, allow_pickle=True)) if os.path.isfile(path) else {}
    missing = [s for s in subjects if f"{s}_t" not in tab]
    if missing:
        raw_dir = raw_dir or os.environ.get("PACE_RAW_DIR", os.path.join(ROOT, "Dataset-2", "Dataset"))
        for s in missing:
            t, X = ds.read_skeleton(depth_path(raw_dir, s))
            tab[f"{s}_t"], tab[f"{s}_X"] = t, X
        os.makedirs(CACHE, exist_ok=True)
        np.savez(path, **tab)
    return {s: (tab[f"{s}_t"], tab[f"{s}_X"]) for s in subjects}


def scale_variant(s, scale):
    from preprocess import normalise_scale_variant, pad_or_truncate
    vid = pd.read_csv(os.path.join(ROOT, "processed", "video_metadata.csv")).set_index("clip_id")
    kp = np.load(os.path.join(ROOT, "processed", "keypoints", f"{s}.npy"))
    arr = pad_or_truncate(normalise_scale_variant(kp, float(vid.loc[s, "width"]), float(vid.loc[s, "height"]), scale))
    return prepare_sequence(arr)


class FeatureSets:
    """Lazily computed feature vectors {subject: vector} of each kind.

    prep: optional {subject: prepared sequence} replacing the stored arrays
    (the positive control plants its difference there); depth_gain: arm-swing
    gain applied to the depth skeletons of the subjects in `depth_planted`."""

    def __init__(self, subjects, prep=None, depth_gain=1.0, depth_planted=()):
        self.S = list(subjects)
        self._prep = prep
        self.depth_gain = depth_gain
        self.depth_planted = set(depth_planted)
        self._cache = {}
        self.meta = pd.read_csv(os.path.join(ROOT, "processed", "participant_metadata.csv")).set_index("subject_id")
        self.vid = pd.read_csv(os.path.join(ROOT, "processed", "video_metadata.csv")).set_index("clip_id")

    def prep(self):
        if self._prep is None:
            self._prep = {s: prepare_sequence(stored(s)) for s in self.S}
        return self._prep

    def get(self, kind):
        if kind not in self._cache:
            self._cache[kind] = getattr(self, "_" + kind)()
        return self._cache[kind]

    def matrix(self, kind, subjects=None):
        d = self.get(kind)
        return np.stack([d[s] for s in (subjects or self.S)])

    # ── movement ──────────────────────────────────────────────────────────────
    def _gait(self):
        P = self.prep()
        return dict(zip(self.S, feature_matrix([P[s] for s in self.S])))

    def _gait_demo(self):
        g, d = self.get("gait"), self.get("demo")
        return {s: np.concatenate([g[s], d[s]]) for s in self.S}

    def _gait_torso(self):
        return dict(zip(self.S, feature_matrix([scale_variant(s, "torso") for s in self.S])))

    def _gait_median(self):
        return dict(zip(self.S, feature_matrix([scale_variant(s, "shoulder_median") for s in self.S])))

    def _mean_pose(self):
        P = self.prep()
        return {s: P[s][valid_mask(P[s])].reshape(-1, 66).mean(0) for s in self.S}

    def _depth(self):
        import depth_skeleton as ds
        if self.depth_gain == 1.0 or not self.depth_planted:
            tab = json.load(open(os.path.join(ROOT, "results", "cv", "depth_features.json")))
            return {s: np.array([tab[s]["features"][k] for k in ds.FEATURE_NAMES], float) for s in self.S}
        from phase3_common import plant_arm_swing_depth
        sk = raw_skeletons(self.S)
        out = {}
        for s in self.S:
            t, X = sk[s]
            if s in self.depth_planted:
                X = plant_arm_swing_depth(X, self.depth_gain)
            f = ds.skeleton_features(t, X)
            out[s] = np.array([f[k] for k in ds.FEATURE_NAMES], float)
        return out

    # ── not movement ──────────────────────────────────────────────────────────
    def _duration(self):
        return {s: np.array([valid_mask(stored(s)).sum()], float) for s in self.S}

    def _mask(self):
        return {s: np.array([mask_summary(stored(s))[k] for k in MASK_FEATURES]) for s in self.S}

    def _demo(self):
        return {s: np.array([self.meta.loc[s, "age_years"], float(self.meta.loc[s, "sex"] == "M")]) for s in self.S}

    def _format(self):
        v = self.vid
        return {s: np.array([v.loc[s, "width"], v.loc[s, "height"], v.loc[s, "width"] / v.loc[s, "height"],
                             float(v.loc[s, "width"] == 1080 and v.loc[s, "height"] == 1920)]) for s in self.S}

    def _scene(self):
        from recording_features import SCENE_NAMES
        sc = json.load(open(os.path.join(ROOT, "results", "cv", "scene_features.json")))
        return {s: np.array([sc[s][k] for k in SCENE_NAMES], float) for s in self.S}

    def _jitter(self):
        from recording_features import JITTER_NAMES, jitter_features
        P = self.prep()
        return {s: np.array([jitter_features(P[s])[k] for k in JITTER_NAMES]) for s in self.S}

    def _tracking(self):
        from recording_features import TRACKING_NAMES, tracking_features
        P = self.prep()
        out = {}
        for s in self.S:
            f = tracking_features(P[s])
            vis = np.load(os.path.join(CACHE, "keypoints_vis", f"{s}.npy"))
            det = np.abs(vis[..., :2]).sum((1, 2)) > 0
            v = [float(vis[det][:, j, 2].mean()) for j in VIS_REGIONS.values()]
            out[s] = np.array([f[k] for k in TRACKING_NAMES] + v, float)
        return out

    def _depth_recording(self):
        import depth_skeleton as ds
        tab = json.load(open(os.path.join(ROOT, "results", "cv", "depth_features.json")))
        return {s: np.array([tab[s]["recording"][k] for k in ds.RECORDING_NAMES], float) for s in self.S}

    def _time_of_day(self):
        import depth_skeleton as ds
        r = self.get("depth_recording")
        i = ds.RECORDING_NAMES.index("time_of_day")
        return {s: r[s][i:i + 1] for s in self.S}

    def _covariates(self):
        f, d, m = self.get("format"), self.get("demo"), self.get("mask")
        return {s: np.concatenate([f[s], d[s], m[s]]) for s in self.S}

    def _all_controls(self):
        parts = [self.get(k) for k in ("format", "demo", "mask", "duration", "scene", "tracking", "depth_recording")]
        return {s: np.concatenate([p[s] for p in parts]) for s in self.S}


def fit_predict(name, F, part, r=0, subjects=None):
    """Out-of-fold probabilities {subject: p} of classifier `name` (as defined
    in run_feature_models.models) for repetition r of `part`."""
    from sklearn.model_selection import GridSearchCV
    from run_feature_models import model_seed, models
    S = subjects or part["subjects"]
    lab = part["labels"]
    out = {}
    for k in range(part["n_outer"]):
        fold = part["repeats"][r][k]
        test = fold["test"]
        train = sorted(set(S) - set(test))
        pos = {s: i for i, s in enumerate(train)}
        inner = [([pos[s] for s in f["train"]], [pos[s] for s in f["val"]]) for f in fold["inner"]]
        est, grid, kind = models(model_seed(r, k, 0))[name]
        Xtr, Xte = F.matrix(kind, train), F.matrix(kind, test)
        ytr = np.array([lab[s] for s in train])
        gs = GridSearchCV(est, grid, scoring="roc_auc", cv=inner, refit=True).fit(Xtr, ytr)
        out.update(zip(test, map(float, gs.predict_proba(Xte)[:, 1])))
    return out
