"""
PACE — feature-based classifiers and negative controls under the same
repeated nested cross-validation partition as the neural networks.

  Gait-LR, Gait-SVM, Gait-RF, Gait-XGB
      hand-crafted gait and upper-body descriptors (src/gait_features.py);
      hyperparameters chosen by mean AUC over the three predefined inner folds
      of each outer fold, then refitted on all outer-training children.
  Duration
      number of frames with a detected pose (logistic regression).
  Mask-pattern
      nine descriptors of the recording's validity pattern (number of detected
      frames, first and last detected frame, span, trailing undetected frames,
      internal gap frames, gap runs, longest gap, gap position), computed from
      the stored arrays before any preparation (logistic regression).
  Demographics
      age and sex from processed/participant_metadata.csv (logistic regression).
  Frame-format
      frame width, height, aspect ratio and whether the frame is the
      uncropped 1080 x 1920 phone format (logistic regression).
  Covariates
      everything above that is not movement: frame format, age, sex and the
      validity-pattern descriptors (logistic regression).
  Gait-LR-imagenorm
      Gait-LR computed from image-normalised coordinates (each axis divided by
      its own frame dimension) instead of pixel units; measures how much that
      normalisation changes the result.

Fourth phase (docs/validation_criteria_phase4.md, E1):
  Time-of-day, Depth-recording, All-controls
      the time of day (last column) enters as a cubic spline (SplineLast);
      the third-phase linear versions are kept as Time-of-day-linear,
      Depth-recording-linear and All-controls-linear.

Third phase (settings fixed in docs/validation_criteria_phase3.md):
  Time-of-day, All-controls, Tracking-quality, Mean-pose
      time of day of the depth recording alone; everything that is not
      movement together; second-difference noise, segment-length variability
      and MediaPipe visibility; the mean prepared pose (posture only).
  Gait-LR-torso, Gait-RF-torso, Gait-LR-median, Gait-RF-median
      the gait descriptors with torso length or a 1-s running median of the
      shoulder distance as the scale (rotation-robust scales).

Second revision (settings fixed in docs/validation_criteria.md, B4-B5):
  Jitter
      landmark jitter (src/recording_features.py), overall and by region.
  Scene
      image properties of the videos (luminance, contrast, saturation,
      sharpness, border luminance, face-region sharpness, bit rate, the
      child's shoulder width relative to the frame height).
  Gait-LR-adjusted
      the gait descriptors residualized on age and sex by a linear regression
      fitted on the training children of each fold (confound regression),
      then the Gait-LR pipeline.
  Depth-LR, Depth-RF, Depth-recording
      descriptors of the depth-camera skeletons (src/depth_skeleton.py) and
      the properties of those recordings (frames, distance, frame-interval
      irregularity, time of day).

Usage:
    python scripts/run_feature_models.py --raw_dir <deposit folder>
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)

from gait_features import FEATURE_NAMES, feature_matrix      # noqa: E402
from run_cv import make_partition, model_seed, stored_array  # noqa: E402
from sequence import MASK_FEATURES, mask_summary, prepare_sequence, valid_mask  # noqa: E402
import depth_skeleton                                        # noqa: E402
from recording_features import JITTER_NAMES, SCENE_NAMES, jitter_features, scene_features  # noqa: E402

from sklearn.base import BaseEstimator, TransformerMixin    # noqa: E402

warnings.filterwarnings("ignore")


class ConfoundRegression(BaseEstimator, TransformerMixin):
    """Residualize every feature on the last `n_confounds` columns with a
    linear regression fitted on the training data only, and drop those
    columns (confound regression; Snoek et al., 2019)."""

    def __init__(self, n_confounds=2):
        self.n_confounds = n_confounds

    def _split(self, X):
        k = self.n_confounds
        return X[:, :-k], np.c_[np.ones(len(X)), X[:, -k:]]

    def fit(self, X, y=None):
        F, C = self._split(X)
        self.beta_ = np.linalg.lstsq(C, F, rcond=None)[0]
        return self

    def transform(self, X):
        F, C = self._split(X)
        return F - C @ self.beta_


class SplineLast(BaseEstimator, TransformerMixin):
    """Replace the last column (the time of day of the depth recording) by a
    cubic B-spline basis with knots spread evenly over the range of the
    training data and constant extrapolation, so that a control can represent
    a group recorded both earlier and later in the day than the other
    (docs/validation_criteria_phase4.md, E1)."""

    def __init__(self, n_knots=5, degree=3):
        self.n_knots = n_knots
        self.degree = degree

    def fit(self, X, y=None):
        from sklearn.preprocessing import SplineTransformer
        self.spline_ = SplineTransformer(n_knots=self.n_knots, degree=self.degree, knots="uniform",
                                         extrapolation="constant").fit(X[:, -1:])
        return self

    def transform(self, X):
        return np.c_[X[:, :-1], self.spline_.transform(X[:, -1:])]


def pipeline(clf, confounds=0, spline_last=False):
    steps = [("impute", SimpleImputer(strategy="median"))]
    if confounds:
        steps.append(("confounds", ConfoundRegression(confounds)))
    if spline_last:
        steps.append(("spline", SplineLast()))
    return Pipeline(steps + [("scale", StandardScaler()), ("clf", clf)])


def scene_table(S, vid, raw_dir, cache):
    """Scene/camera features of every video, cached in `cache`."""
    if os.path.isfile(cache):
        tab = json.load(open(cache))
        if set(S) <= set(tab):
            return tab
    if not raw_dir or not os.path.isdir(raw_dir):
        raise SystemExit("--raw_dir (the deposit's Dataset folder) is needed for the scene features")
    tab = {}
    for s in S:
        kp = np.load(os.path.join("processed", "keypoints", f"{s}.npy"))
        tab[s] = scene_features(os.path.join(raw_dir, vid.loc[s, "video"]), kp, float(vid.loc[s, "fps"]))
        print(f"[scene] {s} {tab[s]}", flush=True)
    json.dump(tab, open(cache, "w"), indent=1)
    return tab


def depth_path(raw_dir, s):
    group, n = s.split("_")
    folder = "Autism/children with ASD" if group == "asd" else "Typical"
    return os.path.join(raw_dir, folder, n, "video", "8.xlsx")


def depth_table(S, raw_dir, cache):
    """Descriptors and recording properties of the depth-camera skeletons."""
    if os.path.isfile(cache):
        tab = json.load(open(cache))
        if set(S) <= set(tab):
            return tab
    if not raw_dir or not os.path.isdir(raw_dir):
        raise SystemExit("--raw_dir (the deposit's Dataset folder) is needed for the depth skeletons")
    tab = {}
    for s in S:
        t, X = depth_skeleton.read_skeleton(depth_path(raw_dir, s))
        tab[s] = {"features": depth_skeleton.skeleton_features(t, X),
                  "recording": depth_skeleton.recording_features(t, X)}
    json.dump(tab, open(cache, "w"), indent=1)
    return tab


def models(seed):
    # threads per random forest; the fitted forests do not depend on it
    nj = int(os.environ.get("PACE_RF_JOBS", "4"))
    lr_grid = {"clf__C": [0.001, 0.01, 0.1, 1.0, 10.0]}
    return {
        "Gait-LR": (pipeline(LogisticRegression(max_iter=5000)), lr_grid, "gait"),
        "Gait-SVM": (pipeline(SVC(kernel="rbf", probability=True, random_state=seed)),
                     {"clf__C": [0.1, 1.0, 10.0], "clf__gamma": ["scale", 0.01, 0.1]}, "gait"),
        "Gait-RF": (pipeline(RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=nj)),
                    {"clf__max_depth": [2, 4, None], "clf__min_samples_leaf": [1, 3, 5]}, "gait"),
        "Gait-XGB": (pipeline(XGBClassifier(learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                            eval_metric="logloss", random_state=seed, n_jobs=nj)),
                     {"clf__n_estimators": [100, 300], "clf__max_depth": [2, 3]}, "gait"),
        "Duration": (pipeline(LogisticRegression(max_iter=5000)), {"clf__C": [1.0]}, "duration"),
        "Mask-pattern": (pipeline(LogisticRegression(max_iter=5000)), lr_grid, "mask"),
        "Demographics": (pipeline(LogisticRegression(max_iter=5000)), lr_grid, "demo"),
        "Frame-format": (pipeline(LogisticRegression(max_iter=5000)), lr_grid, "format"),
        "Gait-LR-imagenorm": (pipeline(LogisticRegression(max_iter=5000)), lr_grid, "gait_image"),
        "Covariates": (pipeline(LogisticRegression(max_iter=5000)), lr_grid, "covariates"),
        # second revision
        "Jitter": (pipeline(LogisticRegression(max_iter=5000)), lr_grid, "jitter"),
        "Scene": (pipeline(LogisticRegression(max_iter=5000)), lr_grid, "scene"),
        "Gait-LR-adjusted": (pipeline(LogisticRegression(max_iter=5000), confounds=2), lr_grid,
                             "gait_demo"),
        "Depth-LR": (pipeline(LogisticRegression(max_iter=5000)), lr_grid, "depth"),
        "Depth-RF": (pipeline(RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=nj)),
                     {"clf__max_depth": [2, 4, None], "clf__min_samples_leaf": [1, 3, 5]}, "depth"),
        # fourth phase (docs/validation_criteria_phase4.md, E1): the time of day
        # (last column) enters as a cubic spline; the "-linear" rows are the
        # second- and third-phase definitions, kept for comparison
        "Depth-recording": (pipeline(LogisticRegression(max_iter=5000), spline_last=True), lr_grid,
                            "depth_recording"),
        "Depth-recording-linear": (pipeline(LogisticRegression(max_iter=5000)), lr_grid, "depth_recording"),
        # third phase (docs/validation_criteria_phase3.md, D4, D5, D7, D9)
        "Time-of-day": (pipeline(LogisticRegression(max_iter=5000), spline_last=True), lr_grid, "time_of_day"),
        "Time-of-day-linear": (pipeline(LogisticRegression(max_iter=5000)), lr_grid, "time_of_day"),
        "All-controls": (pipeline(LogisticRegression(max_iter=5000), spline_last=True), lr_grid, "all_controls"),
        "All-controls-linear": (pipeline(LogisticRegression(max_iter=5000)), lr_grid, "all_controls"),
        "Tracking-quality": (pipeline(LogisticRegression(max_iter=5000)), lr_grid, "tracking"),
        "Mean-pose": (pipeline(LogisticRegression(max_iter=5000)), lr_grid, "mean_pose"),
        "Gait-LR-torso": (pipeline(LogisticRegression(max_iter=5000)), lr_grid, "gait_torso"),
        "Gait-RF-torso": (pipeline(RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=nj)),
                          {"clf__max_depth": [2, 4, None], "clf__min_samples_leaf": [1, 3, 5]}, "gait_torso"),
        "Gait-LR-median": (pipeline(LogisticRegression(max_iter=5000)), lr_grid, "gait_median"),
        "Gait-RF-median": (pipeline(RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=nj)),
                           {"clf__max_depth": [2, 4, None], "clf__min_samples_leaf": [1, 3, 5]}, "gait_median"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--raw_dir", default=os.environ.get("PACE_RAW_DIR", os.environ.get("PACE_ASD_RAW_DIR", os.path.join("Dataset-2", "Dataset"))))
    ap.add_argument("--models", nargs="*", default=None, help="subset of models to run")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    part = make_partition(cfg)
    R = args.repeats or cfg["cv"]["n_repeats"]
    S = part["subjects"]
    lab = part["labels"]
    fdir = os.path.join(cfg["data"]["processed_dir"], "features")
    raw = {s: np.load(os.path.join(fdir, f"{s}.npy")).astype(np.float32) for s in S}
    prep = {s: prepare_sequence(raw[s], cfg["data"]["max_abs_coord"], True) for s in S}
    feats = {
        "gait": dict(zip(S, feature_matrix([prep[s] for s in S]))),
        "duration": {s: np.array([valid_mask(raw[s]).sum()], float) for s in S},
        "mask": {s: np.array([mask_summary(raw[s])[k] for k in MASK_FEATURES]) for s in S},
    }
    meta = pd.read_csv(os.path.join(cfg["data"]["processed_dir"], "participant_metadata.csv"))
    meta = meta.set_index("subject_id")
    feats["demo"] = {s: np.array([meta.loc[s, "age_years"], float(meta.loc[s, "sex"] == "M")]) for s in S}
    vid = pd.read_csv(os.path.join(cfg["data"]["processed_dir"], "video_metadata.csv")).set_index("clip_id")
    feats["format"] = {s: np.array([vid.loc[s, "width"], vid.loc[s, "height"],
                                    vid.loc[s, "width"] / vid.loc[s, "height"],
                                    float(vid.loc[s, "width"] == 1080 and vid.loc[s, "height"] == 1920)])
                       for s in S}
    feats["covariates"] = {s: np.concatenate([feats["format"][s], feats["demo"][s], feats["mask"][s]])
                           for s in S}
    image = {s: prepare_sequence(stored_array({**cfg, "data": {**cfg["data"], "coordinates": "image"}}, s),
                                 cfg["data"]["max_abs_coord"], True) for s in S}
    feats["gait_image"] = dict(zip(S, feature_matrix([image[s] for s in S])))
    os.makedirs(cfg["output"]["cv_dir"], exist_ok=True)
    wanted = set(args.models) if args.models else None
    need = lambda kind: wanted is None or any(models(0)[m][2] == kind for m in wanted)
    feats["jitter"] = {s: np.array([jitter_features(prep[s])[k] for k in JITTER_NAMES]) for s in S}
    feats["gait_demo"] = {s: np.concatenate([feats["gait"][s], feats["demo"][s]]) for s in S}
    if need("scene"):
        scene = scene_table(S, vid, args.raw_dir, os.path.join(cfg["output"]["cv_dir"], "scene_features.json"))
        feats["scene"] = {s: np.array([scene[s][k] for k in SCENE_NAMES]) for s in S}
    if need("depth") or need("depth_recording"):
        depth = depth_table(S, args.raw_dir, os.path.join(cfg["output"]["cv_dir"], "depth_features.json"))
        feats["depth"] = {s: np.array([depth[s]["features"][k] for k in depth_skeleton.FEATURE_NAMES])
                          for s in S}
        feats["depth_recording"] = {s: np.array([depth[s]["recording"][k]
                                                 for k in depth_skeleton.RECORDING_NAMES]) for s in S}
    from feature_sets import FeatureSets
    FS = FeatureSets(S, prep=prep)
    for kind in ("time_of_day", "all_controls", "tracking", "mean_pose", "gait_torso", "gait_median"):
        if need(kind):
            feats[kind] = FS.get(kind)
    with open(os.path.join(cfg["output"]["cv_dir"], "gait_features.json"), "w") as f:
        json.dump({"names": FEATURE_NAMES,
                   "values": {s: [None if np.isnan(v) else float(v) for v in feats["gait"][s]]
                              for s in S}}, f)

    for r in range(R):
        for k in range(part["n_outer"]):
            fold = part["repeats"][r][k]
            test = fold["test"]
            train = sorted(set(S) - set(test))
            pos = {s: i for i, s in enumerate(train)}
            inner = [([pos[s] for s in f["train"]], [pos[s] for s in f["val"]]) for f in fold["inner"]]
            ytr = np.array([lab[s] for s in train]); yte = np.array([lab[s] for s in test])
            for name, (est, grid, kind) in models(model_seed(r, k, 0)).items():
                if wanted is not None and name not in wanted:
                    continue
                out_dir = os.path.join(cfg["output"]["cv_dir"], "runs", name)
                os.makedirs(out_dir, exist_ok=True)
                path = os.path.join(out_dir, f"r{r}_k{k}.json")
                if os.path.isfile(path):
                    continue
                Xtr = np.stack([feats[kind][s] for s in train])
                Xte = np.stack([feats[kind][s] for s in test])
                gs = GridSearchCV(est, grid, scoring="roc_auc", cv=inner, refit=True).fit(Xtr, ytr)
                p = gs.predict_proba(Xte)[:, 1]
                rec = {"arm": name, "repeat": r, "outer_fold": k, "test_subject_ids": test,
                       "test_labels": yte.tolist(), "probs": p.tolist(),
                       "best_params": {kk: (vv if isinstance(vv, (int, float, str)) or vv is None
                                            else str(vv)) for kk, vv in gs.best_params_.items()},
                       "inner_auc": float(gs.best_score_)}
                if name == "Gait-LR":
                    clf = gs.best_estimator_.named_steps["clf"]
                    rec["coef"] = dict(zip(FEATURE_NAMES, clf.coef_[0].tolist()))
                json.dump(rec, open(path, "w"))
                print(f"[{name}] r{r} k{k} auc={roc_auc_score(yte, p):.3f} {gs.best_params_}", flush=True)


if __name__ == "__main__":
    main()
