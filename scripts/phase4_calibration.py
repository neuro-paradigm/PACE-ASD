"""
PACE — operating characteristics of the protocol and planted recording
differences (docs/validation_criteria_phase4.md, E3-E5).

Pseudo-groups of the 45 typically developing children, as in the third-phase
positive control (D1), with:

  pc_g<g>   the arm-movement gain g in pseudo-group 1 (video landmarks and
            depth skeletons); g = 1 is the null. Sample sizes n = 20, 30, 45.
  n1        a planted image difference: pseudo-group 1's videos darker, lower
            in contrast and more saturated (landmarks re-extracted from the
            transformed frames; scripts/negative_control_videos.py)
  n2        a planted framing difference: pseudo-group 1's videos cropped to
            their central 80% (landmarks re-extracted)
  n3        a movement-like difference in the video only: the arm-movement
            gain g = 4 in pseudo-group 1's video landmarks, depth skeletons
            unchanged

Every split runs the fast Tier-1 pipeline (12 classifiers under one
repetition of nested cross-validation, with the inner-fold predictions that
the nested incremental-value test needs) and applies criteria (i)-(v) and the
graded outcome.

  --variants        per-child features of every variant, cached
  --run             the splits (--scenarios, --sizes, --workers/--worker_id)
  --summarize       results/phase4/calibration.json

Usage:
    python scripts/phase4_calibration.py --variants
    python scripts/phase4_calibration.py --run --workers 5 --worker_id 0
    python scripts/phase4_calibration.py --summarize
"""

import argparse
import glob
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)

from phase3_common import (nested_partition, paired_auc_bootstrap, plant_arm_swing,  # noqa: E402
                           plant_arm_swing_depth, pseudo_partition)
from phase4_common import OUT4, STACK_CONTROLS, fast_auc, fit_predict_nested, nested_incremental  # noqa: E402

CAL = os.path.join(OUT4, "calibration")
CACHE = os.path.join(OUT4, "cache")
GAINS = [1.0, 2.0, 3.0, 4.0, 6.0]
SIZES = [20, 30, 45]
N_SPLITS = 20
G_STAR = 4.0
MOVEMENT = ["Gait-LR", "Gait-RF", "Mean-pose"]
CONTROLS = ["Frame-format", "Scene", "Time-of-day", "Demographics", "Tracking-quality", "All-controls"]
DEPTH = ["Depth-LR", "Depth-RF", "Depth-recording"]
MODELS = MOVEMENT + CONTROLS + DEPTH
VIDEO_KINDS = ["gait", "mean_pose", "tracking", "gait_smooth", "gait_first90", "span"]


def td_children():
    p = json.load(open("splits/cv_partition.json"))
    return sorted(s for s in p["subjects"] if p["labels"][s] == 0)


def scenarios():
    out = [f"pc_g{g:g}" for g in GAINS]
    return out + ["n1", "n2"]


# ── per-child features of each variant ────────────────────────────────────────

def _video_kinds(prep, vis_dir=None):
    """Video-derived features of prepared sequences {child: (300, 33, 2)}."""
    from gait_features import feature_matrix
    from protocol import crop_valid
    from recording_features import TRACKING_NAMES, smooth_sequence, tracking_features
    from feature_sets import CACHE as VIS_CACHE, VIS_REGIONS
    from sequence import mask_summary, valid_mask
    S = sorted(prep)
    vis_dir = vis_dir or os.path.join(VIS_CACHE, "keypoints_vis")
    out = {"gait": dict(zip(S, feature_matrix([prep[s] for s in S]))),
           "mean_pose": {s: prep[s][valid_mask(prep[s])].reshape(-1, 66).mean(0) for s in S},
           "gait_smooth": dict(zip(S, feature_matrix([smooth_sequence(prep[s]) for s in S]))),
           "gait_first90": dict(zip(S, feature_matrix([crop_valid(prep[s], 90) for s in S]))),
           "span": {s: float(mask_summary(prep[s])["span"]) for s in S}}
    tr = {}
    for s in S:
        f = tracking_features(prep[s])
        vis = np.load(os.path.join(vis_dir, f"{s}.npy"))
        det = np.abs(vis[..., :2]).sum((1, 2)) > 0
        v = [float(vis[det][:, j, 2].mean()) for j in VIS_REGIONS.values()]
        tr[s] = np.array([f[k] for k in TRACKING_NAMES] + v, float)
    out["tracking"] = tr
    return out


def build_variants():
    """Cache {variant: {kind: {child: value}}} for base, video_g<g>, depth_g<g>,
    n1 and n2 (n1/n2 need scripts/negative_control_videos.py first)."""
    from feature_sets import FeatureSets, stored
    from sequence import prepare_sequence
    os.makedirs(CACHE, exist_ok=True)
    td = td_children()
    base_prep = {s: prepare_sequence(stored(s)) for s in td}
    F = FeatureSets(td, prep=base_prep)

    def save(name, d):
        pickle.dump(d, open(os.path.join(CACHE, f"variant_{name}.pkl"), "wb"))
        print(f"[variants] {name}", flush=True)

    if not os.path.isfile(os.path.join(CACHE, "variant_base.pkl")):
        d = {k: F.get(k) for k in ("format", "scene", "demo", "mask", "duration", "depth", "depth_recording")}
        d.update(_video_kinds(base_prep))
        save("base", d)
    for g in GAINS[1:]:
        if not os.path.isfile(os.path.join(CACHE, f"variant_video_g{g:g}.pkl")):
            save(f"video_g{g:g}", _video_kinds({s: plant_arm_swing(base_prep[s], g) for s in td}))
        if not os.path.isfile(os.path.join(CACHE, f"variant_depth_g{g:g}.pkl")):
            Fd = FeatureSets(td, prep=base_prep, depth_gain=g, depth_planted=td)
            save(f"depth_g{g:g}", {"depth": Fd.get("depth")})
    for v in ("n1", "n2"):
        kd = os.path.join(CACHE, f"keypoints_{v}")
        if os.path.isfile(os.path.join(CACHE, f"variant_{v}.pkl")) or not os.path.isdir(kd):
            continue
        from preprocess import normalise, pad_or_truncate
        from recording_features import SCENE_NAMES
        from sequence import MASK_FEATURES, mask_summary, valid_mask
        meta = json.load(open(os.path.join(CACHE, f"videos_{v}.json")))
        raw = {s: pad_or_truncate(normalise(np.load(os.path.join(kd, f"{s}.npy"))[..., :2],
                                            meta[s]["width"], meta[s]["height"])) for s in td}
        prep = {s: prepare_sequence(raw[s]) for s in td}
        d = _video_kinds(prep, vis_dir=kd)
        d["scene"] = {s: np.array([meta[s]["scene"][k] for k in SCENE_NAMES], float) for s in td}
        d["mask"] = {s: np.array([mask_summary(raw[s])[k] for k in MASK_FEATURES]) for s in td}
        d["duration"] = {s: np.array([valid_mask(raw[s]).sum()], float) for s in td}
        W = {s: float(meta[s]["width"]) for s in td}
        H = {s: float(meta[s]["height"]) for s in td}
        d["format"] = {s: np.array([W[s], H[s], W[s] / H[s], float(W[s] == 1080 and H[s] == 1920)]) for s in td}
        save(v, d)


def load_variant(name):
    return pickle.load(open(os.path.join(CACHE, f"variant_{name}.pkl"), "rb"))


def scenario_sources(scn):
    """For pseudo-group 1: {kind: variant}; kinds not listed come from base."""
    if scn.startswith("pc_g"):
        g = float(scn[4:])
        if g == 1.0:
            return {}
        return {**{k: f"video_g{g:g}" for k in VIDEO_KINDS}, "depth": f"depth_g{g:g}"}
    if scn == "n3":
        return {k: f"video_g{G_STAR:g}" for k in VIDEO_KINDS}
    if scn in ("n1", "n2"):
        src = {k: scn for k in VIDEO_KINDS + ["scene", "mask", "duration"]}
        if scn == "n2":
            src["format"] = "n2"
        return src
    raise ValueError(scn)


class Assembled:
    """Feature sets of one split: each child's features from the variant its
    pseudo-group calls for (the interface of feature_sets.FeatureSets used by
    fit_predict)."""

    KINDS = ["gait", "mean_pose", "tracking", "format", "scene", "demo", "mask", "duration", "depth",
             "depth_recording"]

    def __init__(self, subjects, labels, scn, variants):
        src = scenario_sources(scn)
        self.S = list(subjects)
        self._d = {}
        for kind in self.KINDS + ["gait_smooth", "gait_first90", "span"]:
            self._d[kind] = {s: variants[src.get(kind, "base") if labels[s] == 1 else "base"][kind][s]
                             for s in self.S}
        import depth_skeleton as ds
        i = ds.RECORDING_NAMES.index("time_of_day")
        self._d["time_of_day"] = {s: self._d["depth_recording"][s][i:i + 1] for s in self.S}
        parts = ("format", "demo", "mask", "duration", "scene", "tracking", "depth_recording")
        self._d["all_controls"] = {s: np.concatenate([self._d[k][s] for k in parts]) for s in self.S}

    def get(self, kind):
        return self._d[kind]

    def matrix(self, kind, subjects=None):
        return np.stack([np.atleast_1d(self._d[kind][s]) for s in (subjects or self.S)])


# ── splits ────────────────────────────────────────────────────────────────────

def split_partition(n, j, td):
    """n = 45: the third-phase pseudo-groups (D1); n < 45: n/2 children per
    pseudo-group drawn at random (seed 20261301 + 1000 n + j)."""
    if n == 45:
        return pseudo_partition(j, td)
    seed = 20261301 + 1000 * n + j
    perm = np.random.default_rng(seed).permutation(len(td))
    h = n // 2
    subj = [td[i] for i in perm[:n]]
    lab = [1] * h + [0] * (n - h)
    order = np.argsort(subj)
    return nested_partition([subj[i] for i in order], [lab[i] for i in order], seed)


def jobs(scns, sizes):
    """Largest sample size first, so that the n = 45 cells finish first."""
    out = []
    for n in sorted(sizes, reverse=True):
        for scn in scns:
            if scn.startswith("pc_") or n == 45:
                out += [(scn, n, j) for j in range(N_SPLITS)]
    return out


def run(scns, sizes, workers, worker_id):
    td = td_children()
    variants = {}
    for scn, n, j in jobs(scns, sizes)[worker_id::workers]:
        path = os.path.join(CAL, scn, f"n{n}", f"split{j:02d}.json")
        if os.path.isfile(path):
            continue
        t0 = time.time()
        part = split_partition(n, j, td)
        need = {"base"} | set(scenario_sources(scn).values())
        for v in need - set(variants):
            variants[v] = load_variant(v)
        F = Assembled(part["subjects"], part["labels"], scn, variants)
        rec = {"scenario": scn, "n": n, "split": j, "labels": part["labels"], "probs": {}, "inner": {},
               "params": {}}
        # n1 and n2 leave the depth skeletons as they are, so their depth-side classifiers are
        # identical to those of pc_g1 on the same split (same partition, features and seeds)
        null = os.path.join(CAL, "pc_g1", f"n{n}", f"split{j:02d}.json")
        reuse = json.load(open(null)) if scn in ("n1", "n2") and os.path.isfile(null) else None
        for m in MODELS:
            if reuse is not None and m in DEPTH:
                rec["probs"][m], rec["inner"][m], rec["params"][m] = reuse["probs"][m], reuse["inner"][m], reuse["params"][m]
                continue
            o, i, p = fit_predict_nested(m, F, part)
            rec["probs"][m], rec["inner"][m], rec["params"][m] = o, i, p
        rec["depth_from_pc_g1"] = reuse is not None
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(rec, open(path, "w"))
        y = [part["labels"][s] for s in part["subjects"]]
        print(f"[{scn} n={n} split {j}] {time.time() - t0:.0f}s " +
              " ".join(f"{m}={fast_auc(y, [rec['probs'][m][s] for s in part['subjects']]):.2f}" for m in MODELS),
              flush=True)


# ── criteria ──────────────────────────────────────────────────────────────────

def robust_descriptors(F, S, y, B=500):
    """Criterion (iv): descriptors whose group effect (adjusted for age and
    sex) has q < 0.05 over the 26 descriptors and keeps its sign with a 95%
    interval excluding 0 for smoothed trajectories and for the first 90
    frames (recordings with at least 90 frames)."""
    from analyze_mechanisms import _bh, _ols_group_effect
    from gait_features import FEATURE_NAMES
    meta = pd.read_csv("processed/participant_metadata.csv").set_index("subject_id")
    age = np.array([meta.loc[s, "age_years"] for s in S], float)
    male = np.array([meta.loc[s, "sex"] == "M" for s in S])
    X, Xs, X9 = F.matrix("gait", S), F.matrix("gait_smooth", S), F.matrix("gait_first90", S)
    span = np.array([F.get("span")[s] for s in S])
    ok9 = span >= 90
    eff = [_ols_group_effect(X[:, j], y, age, male, B=B) for j in range(X.shape[1])]
    q = _bh([e["p"] for e in eff])
    robust = []
    for j in np.flatnonzero(q < 0.05):
        st = eff[j]
        same = lambda e: np.sign(e["beta"]) == np.sign(st["beta"]) and (e["ci95"][0] > 0 or e["ci95"][1] < 0)
        e_sm = _ols_group_effect(Xs[:, j], y, age, male, B=B)
        e_9 = _ols_group_effect(X9[ok9, j], y[ok9], age[ok9], male[ok9], B=B)
        if same(e_sm) and same(e_9):
            robust.append(FEATURE_NAMES[j])
    a = FEATURE_NAMES.index("arm_swing")
    return {"robust": robust, "n_q05": int((q < 0.05).sum()), "arm_swing_q": float(q[a]),
            "arm_swing_robust": "arm_swing" in robust}


def grade(c):
    if c["i"] and c["ii"] and c["iii"] and c["iv"] and c["v"]:
        return "A"
    if c["i"] and c["ii"] and c["v"]:
        return "B"
    return "C"


def evaluate(rec, variants, td):
    part = split_partition(rec["n"], rec["split"], td)
    S = part["subjects"]
    assert {s: part["labels"][s] for s in S} == rec["labels"]
    y = np.array([rec["labels"][s] for s in S])
    P = {m: np.array([rec["probs"][m][s] for s in S]) for m in MODELS}
    auc = {m: fast_auc(y, P[m]) for m in MODELS}
    out = {"auc": auc}
    c = {"i": bool(all(auc[m] < 0.70 for m in CONTROLS))}
    best = max(MOVEMENT, key=lambda m: auc[m])
    pb = paired_auc_bootstrap(y, P[best], P["All-controls"])
    c["ii"] = bool(pb["ci_a"][0] > 0.5 and pb["diff_ci95"][0] > 0)
    outer = {m: {0: rec["probs"][m]} for m in STACK_CONTROLS + [best]}
    inner = {m: {0: {int(k): v for k, v in rec["inner"][m].items()}} for m in STACK_CONTROLS + [best]}
    iv = nested_incremental(part, outer, inner, STACK_CONTROLS, best, R=1, B=2000)
    c["iii"] = iv["adds"]
    F = Assembled(S, rec["labels"], rec["scenario"], variants)
    rb = robust_descriptors(F, S, y)
    c["iv"] = bool(len(rb["robust"]) > 0)
    c["iv_arm_swing"] = rb["arm_swing_robust"]
    dbest = max(["Depth-LR", "Depth-RF"], key=lambda m: auc[m])
    dpb = paired_auc_bootstrap(y, P[dbest], P["Depth-recording"])
    c["v"] = bool(dpb["ci_a"][0] > 0.5 and dpb["auc_a"] > dpb["auc_b"])
    out.update({"best_movement_model": best, "best_vs_all_controls": pb, "incremental_value": iv,
                "descriptors": rb, "depth": {"model": dbest, **dpb}, "criteria": c, "grade": grade(c),
                "movement_concluded": grade(c) == "A",
                "movement_concluded_arm_swing_rule": bool(c["i"] and c["ii"] and c["iii"] and c["iv_arm_swing"]
                                                          and c["v"])})
    return out


def _n3_record(n45_g4, n45_g1):
    """Scenario n3: video-side classifiers of pc_g4 and depth-side classifiers
    of pc_g1 (same pseudo-groups, partitions and seeds)."""
    rec = json.loads(json.dumps(n45_g4))
    for m in ("Depth-LR", "Depth-RF", "Depth-recording"):
        rec["probs"][m] = n45_g1["probs"][m]
        rec["inner"][m] = n45_g1["inner"][m]
        rec["params"][m] = n45_g1["params"][m]
    rec["scenario"] = "n3"
    return rec


def summarize():
    td = td_children()
    variants = {v: load_variant(v) for v in
                ["base"] + [f"video_g{g:g}" for g in GAINS[1:]] + [f"depth_g{g:g}" for g in GAINS[1:]] +
                [v for v in ("n1", "n2") if os.path.isfile(os.path.join(CACHE, f"variant_{v}.pkl"))]}
    res = {"cells": {}}
    keys = ["i", "ii", "iii", "iv", "iv_arm_swing", "v"]
    cells = [(scn, n) for scn in scenarios() for n in (SIZES if scn.startswith("pc_") else [45])] + [("n3", 45)]
    for scn, n in cells:
        rows = []
        for j in range(N_SPLITS):
            if scn == "n3":
                a = os.path.join(CAL, "pc_g4", "n45", f"split{j:02d}.json")
                b = os.path.join(CAL, "pc_g1", "n45", f"split{j:02d}.json")
                if not (os.path.isfile(a) and os.path.isfile(b)):
                    continue
                rec = _n3_record(json.load(open(a)), json.load(open(b)))
            else:
                f = os.path.join(CAL, scn, f"n{n}", f"split{j:02d}.json")
                if not os.path.isfile(f):
                    continue
                rec = json.load(open(f))
            if any(v not in variants for v in set(scenario_sources(rec["scenario"]).values())):
                continue
            # evaluations are cached next to the summary; recomputed if the split record is newer
            ecache = os.path.join(OUT4, "calibration_eval", scn, f"n{n}", f"split{j:02d}.json")
            srcs = [os.path.join(CAL, s, f"n{n}", f"split{j:02d}.json") for s in
                    (["pc_g4", "pc_g1"] if scn == "n3" else [scn])]
            if os.path.isfile(ecache) and os.path.getmtime(ecache) > max(os.path.getmtime(f) for f in srcs):
                ev = json.load(open(ecache))
            else:
                ev = evaluate(rec, variants, td)
                os.makedirs(os.path.dirname(ecache), exist_ok=True)
                json.dump(ev, open(ecache, "w"))
            rows.append({"split": j, **ev})
            print(f"[summary] {scn} n={n} split {j}: grade {ev['grade']} {ev['criteria']}", flush=True)
        if not rows:
            continue
        share = {k: float(np.mean([r["criteria"][k] for r in rows])) for k in keys}
        share["movement_concluded"] = float(np.mean([r["movement_concluded"] for r in rows]))
        share["movement_concluded_arm_swing_rule"] = float(np.mean([r["movement_concluded_arm_swing_rule"] for r in rows]))
        share["grade_A"] = float(np.mean([r["grade"] == "A" for r in rows]))
        share["grade_AB"] = float(np.mean([r["grade"] in ("A", "B") for r in rows]))
        aucs = {m: {"mean": float(np.mean([r["auc"][m] for r in rows])),
                    "min": float(np.min([r["auc"][m] for r in rows])),
                    "max": float(np.max([r["auc"][m] for r in rows]))} for m in MODELS}
        ctrl = [max(r["auc"][m] for m in CONTROLS) for r in rows]
        res["cells"][f"{scn}|n{n}"] = {"scenario": scn, "n": n, "n_splits": len(rows), "share": share, "auc": aucs,
                                        "largest_control_auc": {"mean": float(np.mean(ctrl)), "max": float(np.max(ctrl))},
                                        "splits": rows}
    json.dump(res, open(os.path.join(OUT4, "calibration.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--sizes", type=int, nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--worker_id", type=int, default=0)
    a = ap.parse_args()
    if a.variants:
        build_variants()
    if a.run:
        run(a.scenarios or scenarios(), a.sizes or SIZES, a.workers, a.worker_id)
    if a.summarize:
        summarize()
