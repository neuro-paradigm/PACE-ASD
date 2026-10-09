"""
PACE — independent checks of the analysis code (docs/validation_criteria_phase3.md, D16).

Each check re-derives a reported quantity with code that does not call the
implementation it checks, and records whether the two agree:

  auc            AUC of every stored model by direct counting of
                 (positive, negative) pairs against fast_auc and
                 sklearn.metrics.roc_auc_score
  bootstrap      the 95% interval of the test-case network's AUC re-derived
                 with a separately written stratified bootstrap (same seed
                 convention, different code) against summary.json
  partition      outer test sets disjoint and covering all children in every
                 repetition; inner folds disjoint, inside the outer training
                 children and covering them; labels as in labels.csv; the
                 partition file regenerated from the configuration seed is
                 identical
  duplicates     MD5 of every color video recomputed from the deposit; the
                 five removed recordings are byte-identical to a retained one
                 and no other pair is
  arrays         every stored (300, 33, 2) array recomputed from the stored
                 keypoints and the frame size
  runs           every stored out-of-fold record has the test children of its
                 partition fold and probabilities in [0, 1]

Writes results/phase3/verification.json.

Usage:
    python scripts/verify_analysis.py --raw_dir Dataset-2/Dataset
"""

import argparse
import glob
import hashlib
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)

OUT = os.path.join("results", "phase3", "verification.json")


def auc_by_pairs(y, p):
    """Probability that a random positive scores above a random negative
    (ties count one half), by enumerating all pairs."""
    y = np.asarray(y); p = np.asarray(p, float)
    pos, neg = p[y == 1], p[y == 0]
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def load_oof(name):
    by_r = defaultdict(dict)
    for f in glob.glob(os.path.join("results", "cv", "runs", name, "r*_k*.json")):
        rec = json.load(open(f))
        by_r[rec["repeat"]].update(zip(rec["test_subject_ids"], rec["probs"]))
    return by_r


def check_auc():
    from sklearn.metrics import roc_auc_score
    from analyze_mechanisms import fast_auc
    part = json.load(open("splits/cv_partition.json"))
    S = part["subjects"]
    y = np.array([part["labels"][s] for s in S])
    worst, n = 0.0, 0
    summary = json.load(open("results/cv/summary.json"))
    rep_mismatch = 0.0
    for name in sorted(os.listdir(os.path.join("results", "cv", "runs"))):
        if name == "pilot":
            continue
        by_r = load_oof(name)
        aucs = []
        for r in sorted(by_r):
            if len(by_r[r]) != len(S):
                continue
            p = np.array([by_r[r][s] for s in S])
            a, b, c = auc_by_pairs(y, p), fast_auc(y, p), roc_auc_score(y, p)
            worst = max(worst, abs(a - b), abs(a - c))
            aucs.append(a)
            n += 1
        if name in summary["performance"] and len(aucs) >= summary["n_repeats"]:
            rep_mismatch = max(rep_mismatch, abs(np.mean(aucs[:summary["n_repeats"]])
                                                 - summary["performance"][name]["mean"]["auc"]))
    return {"n_auc_values": n, "max_abs_difference_pairs_vs_implementations": worst,
            "max_abs_difference_vs_summary": rep_mismatch, "agree": bool(worst < 1e-12 and rep_mismatch < 1e-9)}


def check_bootstrap(B=2000, seed=20261010):
    """Re-derive the interval of the test-case network's repetition-averaged
    AUC. analyze_cv.py draws the positives and then the negatives with one
    generator; this re-implementation draws index arrays the same way but
    computes the AUC by pair counting."""
    part = json.load(open("splits/cv_partition.json"))
    S = part["subjects"]
    y = np.array([part["labels"][s] for s in S])
    by_r = load_oof("PACE")
    R = json.load(open("results/cv/summary.json"))["n_repeats"]
    P = np.array([[by_r[r][s] for s in S] for r in sorted(by_r)[:R]])
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    draws = []
    names = sorted(n for n in os.listdir(os.path.join("results", "cv", "runs"))
                   if n != "pilot" and os.path.isdir(os.path.join("results", "cv", "runs", n)))
    # analyze_cv.bootstrap draws once per resample for all models together, so
    # the resample sequence does not depend on the model; only the indices matter
    for _ in range(B):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        draws.append(np.mean([auc_by_pairs(y[idx], P[r, idx]) for r in range(len(P))]))
    ci = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
    rep = json.load(open("results/cv/summary.json"))["performance"]["PACE"]["ci95"]["auc"]
    return {"reimplemented": ci, "reported": rep, "max_abs_difference": float(max(abs(ci[0] - rep[0]), abs(ci[1] - rep[1]))),
            "agree": bool(max(abs(ci[0] - rep[0]), abs(ci[1] - rep[1])) < 1e-9), "n_models_in_summary": len(names)}


def check_partition():
    import yaml
    from run_cv import make_partition
    part = json.load(open("splits/cv_partition.json"))
    S = set(part["subjects"])
    lab = pd.read_csv("processed/labels.csv").set_index("subject_id")
    problems = []
    for s in S:
        if int(lab.loc[s, "label"]) != part["labels"][s]:
            problems.append(f"label {s}")
    for r, folds in enumerate(part["repeats"]):
        tests = [set(f["test"]) for f in folds]
        if set().union(*tests) != S or sum(len(t) for t in tests) != len(S):
            problems.append(f"repeat {r}: outer test sets do not partition the cohort")
        for k, f in enumerate(folds):
            train = S - set(f["test"])
            vals = [set(i["val"]) for i in f["inner"]]
            if set().union(*vals) != train or sum(len(v) for v in vals) != len(train):
                problems.append(f"repeat {r} fold {k}: inner validation sets do not partition the training children")
            for i in f["inner"]:
                if set(i["train"]) | set(i["val"]) != train or set(i["train"]) & set(i["val"]):
                    problems.append(f"repeat {r} fold {k}: inner split")
                if set(i["train"]) & set(f["test"]) or set(i["val"]) & set(f["test"]):
                    problems.append(f"repeat {r} fold {k}: test child in inner fold")
    cfg = yaml.safe_load(open("configs/config.yaml"))
    tmp = os.path.join("results", "phase3", "cache", "partition_regenerated.json")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    if os.path.isfile(tmp):
        os.remove(tmp)
    regen = make_partition({**cfg, "cv": {**cfg["cv"], "partition_file": tmp}})
    same = regen["repeats"] == part["repeats"] and regen["subjects"] == part["subjects"]
    return {"n_children": len(S), "n_repeats_in_file": len(part["repeats"]), "problems": problems,
            "regenerated_identical": bool(same), "agree": bool(not problems and same)}


def check_duplicates(raw_dir):
    from preprocess import build_video_catalogue
    cat = build_video_catalogue(os.path.abspath(raw_dir))
    md5 = {}
    for e in cat:
        h = hashlib.md5(open(e["video_path"], "rb").read()).hexdigest()
        md5.setdefault(h, []).append(e["clip_id"])
    groups = [sorted(v) for v in md5.values() if len(v) > 1]
    removed = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join("processed", "removed_duplicates", "*.npy")))
    cohort = set(json.load(open("splits/cv_partition.json"))["subjects"])
    ok = all(any(r in g for g in groups) for r in removed) and all(
        len([c for c in g if c in cohort]) == 1 for g in groups)
    return {"n_videos": len(cat), "identical_groups": groups, "removed": removed,
            "each_group_keeps_one": bool(ok), "agree": bool(ok and len(groups) == len(removed))}


def check_arrays():
    from preprocess import normalise, pad_or_truncate
    vid = pd.read_csv("processed/video_metadata.csv").set_index("clip_id")
    S = json.load(open("splits/cv_partition.json"))["subjects"]
    worst = 0.0
    for s in S:
        kp = np.load(os.path.join("processed", "keypoints", f"{s}.npy"))
        arr = pad_or_truncate(normalise(kp, float(vid.loc[s, "width"]), float(vid.loc[s, "height"])))
        stored = np.load(os.path.join("processed", "features", f"{s}.npy"))
        worst = max(worst, float(np.abs(arr - stored).max()))
    return {"n": len(S), "max_abs_difference": worst, "agree": bool(worst < 1e-5)}


def check_runs():
    part = json.load(open("splits/cv_partition.json"))
    problems = []
    n = 0
    for f in glob.glob(os.path.join("results", "cv", "runs", "*", "r*_k*.json")):
        if os.sep + "pilot" + os.sep in f:
            continue
        rec = json.load(open(f))
        fold = part["repeats"][rec["repeat"]][rec["outer_fold"]]
        if sorted(rec["test_subject_ids"]) != sorted(fold["test"]):
            problems.append(f)
        p = np.asarray(rec["probs"], float)
        if not (np.all(p >= 0) and np.all(p <= 1) and len(p) == len(rec["test_subject_ids"])):
            problems.append(f + " probs")
        if [part["labels"][s] for s in rec["test_subject_ids"]] != rec["test_labels"]:
            problems.append(f + " labels")
        n += 1
    return {"n_records": n, "problems": problems[:20], "agree": not problems}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default=os.environ.get("PACE_RAW_DIR", os.path.join("Dataset-2", "Dataset")))
    a = ap.parse_args()
    res = {}
    for name, fn in (("auc", check_auc), ("bootstrap", check_bootstrap), ("partition", check_partition),
                     ("duplicates", lambda: check_duplicates(a.raw_dir)), ("arrays", check_arrays),
                     ("runs", check_runs)):
        res[name] = fn()
        print(name, {k: v for k, v in res[name].items() if k not in ("identical_groups",)}, flush=True)
    rep = "results/cv/reproducibility.json"
    if os.path.isfile(rep):
        res["rerun_one_fold"] = json.load(open(rep))
    res["all_agree"] = bool(all(v.get("agree", True) for v in res.values() if isinstance(v, dict)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
