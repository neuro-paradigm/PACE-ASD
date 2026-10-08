"""
PACE-ASD — feature-based classifiers and negative controls under the same
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

Usage:
    python scripts/run_feature_models.py
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

warnings.filterwarnings("ignore")


def pipeline(clf):
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()), ("clf", clf)])


def models(seed):
    lr_grid = {"clf__C": [0.001, 0.01, 0.1, 1.0, 10.0]}
    return {
        "Gait-LR": (pipeline(LogisticRegression(max_iter=5000)), lr_grid, "gait"),
        "Gait-SVM": (pipeline(SVC(kernel="rbf", probability=True, random_state=seed)),
                     {"clf__C": [0.1, 1.0, 10.0], "clf__gamma": ["scale", 0.01, 0.1]}, "gait"),
        "Gait-RF": (pipeline(RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=4)),
                    {"clf__max_depth": [2, 4, None], "clf__min_samples_leaf": [1, 3, 5]}, "gait"),
        "Gait-XGB": (pipeline(XGBClassifier(learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                            eval_metric="logloss", random_state=seed, n_jobs=4)),
                     {"clf__n_estimators": [100, 300], "clf__max_depth": [2, 3]}, "gait"),
        "Duration": (pipeline(LogisticRegression(max_iter=5000)), {"clf__C": [1.0]}, "duration"),
        "Mask-pattern": (pipeline(LogisticRegression(max_iter=5000)), lr_grid, "mask"),
        "Demographics": (pipeline(LogisticRegression(max_iter=5000)), lr_grid, "demo"),
        "Frame-format": (pipeline(LogisticRegression(max_iter=5000)), lr_grid, "format"),
        "Gait-LR-imagenorm": (pipeline(LogisticRegression(max_iter=5000)), lr_grid, "gait_image"),
        "Covariates": (pipeline(LogisticRegression(max_iter=5000)), lr_grid, "covariates"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--repeats", type=int, default=None)
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
