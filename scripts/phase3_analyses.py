"""
PACE — third-phase analyses (docs/validation_criteria_phase3.md) that work on
the stored data, predictions and checkpoints. Writes
results/phase3/analyses.json; each section can be run alone with --sections.

  metadata      D14  AVI header fields and file times in the deposit, by group
  tracking      D5   tracking-quality measures by group
  incremental   D3   movement scores beyond the control scores; correlations
  reliability   D6   split-half reliability; reliability-bounded agreement
  scale         D7   descriptors with rotation-robust scales
  viewpoint     D10  adjusted descriptors with a viewpoint and size covariate
  age_subset    D11  children whose ages agree with the original article
  within_cropped D12 nested cross-validation within the cropped videos
  shuffle       D9   frame-order shuffle of the stored checkpoints
  ig            D13  integrated gradients and temporal occlusion, real data

Usage:
    python scripts/phase3_analyses.py [--sections ...]
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)

from analyze_mechanisms import _auc_ci, _bh, _ols_group_effect, fast_auc, mat, oof, part_info, summ  # noqa: E402
from phase3_common import OUT, incremental_value, paired_auc_bootstrap   # noqa: E402
from sequence import mask_summary, prepare_sequence, valid_mask           # noqa: E402

RES = os.path.join(OUT, "analyses")
RAW_DIR = os.environ.get("PACE_RAW_DIR", os.path.join("Dataset-2", "Dataset"))
DEPTH_MATCH = {"step_width": "step_width", "arm_swing": "arm_swing", "wrist_height": "wrist_height",
               "knee_rom": "knee_rom", "elbow_rom": "elbow_rom", "trunk_sway": "trunk_sway",
               "head_bob": "head_bob"}


def cohort_info():
    p, S, y = part_info()
    meta = pd.read_csv("processed/participant_metadata.csv").set_index("subject_id")
    vid = pd.read_csv("processed/video_metadata.csv").set_index("clip_id")
    age = np.array([meta.loc[s, "age_years"] for s in S], float)
    male = np.array([meta.loc[s, "sex"] == "M" for s in S])
    crop = np.array([not (vid.loc[s, "width"] == 1080 and vid.loc[s, "height"] == 1920) for s in S])
    return S, y, age, male, crop, vid


def raw(s):
    return np.load(f"processed/features/{s}.npy").astype(np.float32)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# ── D14 metadata ──────────────────────────────────────────────────────────────

def _avi_header(path, maxbytes=200000):
    import struct
    data = open(path, "rb").read(maxbytes)
    out = {}

    def walk(off, end):
        while off + 8 <= min(end, len(data)):
            cid = data[off:off + 4]
            size = struct.unpack("<I", data[off + 4:off + 8])[0]
            if cid in (b"RIFF", b"LIST"):
                walk(off + 12, min(off + 8 + size, len(data)))
            else:
                body = data[off + 8:off + 8 + min(size, 256)]
                k = cid.decode("latin1")
                if k == "strh" and body[:4] == b"vids":
                    out["codec"] = body[4:8].decode("latin1").upper()
                elif k in ("ISFT", "IDIT", "ICRD"):
                    out[k] = body.rstrip(b"\0").decode("latin1", errors="replace")
                elif k == "avih":
                    out["us_per_frame"] = struct.unpack("<I", body[0:4])[0]
            off += 8 + size + (size & 1)
    walk(0, len(data))
    return out


def metadata():
    import datetime
    S, y, *_ , vid = cohort_info()
    rows = {}
    for s in S:
        g, n = s.split("_")
        folder = os.path.join(RAW_DIR, "Autism/children with ASD" if g == "asd" else "Typical", n, "video")
        h = _avi_header(os.path.join(RAW_DIR, vid.loc[s, "video"]))
        r = {"codec": h.get("codec"), "writing_library": h.get("ISFT"),
             "date_fields": {k: h[k] for k in ("IDIT", "ICRD") if k in h}}
        for f in ("Tvideo.avi", "Svideo.avi", "8.xlsx", os.path.basename(vid.loc[s, "video"])):
            p = os.path.join(folder, f)
            if os.path.isfile(p):
                key = "color_video" if f.startswith("video") else f
                r[f"mtime_{key}"] = datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d")
        rows[s] = r
    out = {"per_child": rows, "by_group": {}}
    for g, lab in (("autistic", 1), ("td", 0)):
        R = [rows[s] for s, yy in zip(S, y) if yy == lab]
        d = {"codecs": sorted({r["codec"] for r in R}), "writing_libraries": sorted({str(r["writing_library"]) for r in R}),
             "n_with_date_field": int(sum(bool(r["date_fields"]) for r in R))}
        for key in ("Tvideo.avi", "Svideo.avi", "8.xlsx", "color_video"):
            v = sorted(r[f"mtime_{key}"] for r in R if f"mtime_{key}" in r)
            if v:
                d[f"{key}_range"] = [v[0], v[-1]]
                d[f"{key}_months"] = sorted({x[:7] for x in v})
        out["by_group"][g] = d
    ta = [rows[s]["mtime_Tvideo.avi"] for s, yy in zip(S, y) if yy == 1]
    tt = [rows[s]["mtime_Tvideo.avi"] for s, yy in zip(S, y) if yy == 0]
    out["tvideo_ranges_overlap"] = bool(max(ta) >= min(tt) and max(tt) >= min(ta))
    return out


# ── D5 tracking quality by group ──────────────────────────────────────────────

def tracking():
    from feature_sets import FeatureSets
    from recording_features import TRACKING_NAMES
    S, y, age, male, crop, vid = cohort_info()
    F = FeatureSets(S)
    T = F.matrix("tracking")
    names = TRACKING_NAMES + ["visibility_face", "visibility_arms_hands", "visibility_legs_feet"]
    J = F.matrix("jitter")
    out = {}
    for i, k in enumerate(names):
        out[k] = {"median_autistic": float(np.nanmedian(T[y == 1, i])), "median_td": float(np.nanmedian(T[y == 0, i])),
                  "median_autistic_cropped": float(np.nanmedian(T[(y == 1) & crop, i])),
                  "all": _auc_ci(y, T[:, i]), "cropped": _auc_ci(y[crop], T[crop, i])}
    # how far the 9-frame residual is above the second-difference noise
    out["jitter_over_noise2_median"] = float(np.nanmedian(J[:, 0] / T[:, 0]))
    return out


# ── D3 incremental value ──────────────────────────────────────────────────────

CONTROL_SCORES = ["Scene", "Time-of-day", "Frame-format", "Demographics"]
MOVEMENT_SCORES = ["PACE", "Gait-RF", "Depth-RF"]


def incremental():
    S, y, age, male, crop, vid = cohort_info()
    sc = {m: logit(mat(oof(m), S, 3).mean(0)) for m in CONTROL_SCORES + MOVEMENT_SCORES + ["All-controls", "Gait-LR"]}
    out = {"controls": CONTROL_SCORES, "lambda": 0.1}
    for name, idx in (("all", np.arange(len(S))), ("cropped", np.flatnonzero(crop))):
        C = np.c_[[sc[c][idx] for c in CONTROL_SCORES]].T
        r = {}
        for m in MOVEMENT_SCORES:
            r[m] = incremental_value(y[idx], C, sc[m][idx])
            print(f"[incremental] {name} {m}: {r[m]['lr_p']:.3g} {r[m]['loo_auc_controls']:.3f} -> "
                  f"{r[m]['loo_auc_with_movement']:.3f} {r[m]['loo_auc_diff_ci95']}", flush=True)
        out[name] = r
    # correlations of movement and control scores, overall and within each group
    cor = {}
    for m in MOVEMENT_SCORES + ["Gait-LR"]:
        for c in CONTROL_SCORES + ["All-controls"]:
            d = {"all": float(stats.spearmanr(sc[m], sc[c]).correlation),
                 "cropped": float(stats.spearmanr(sc[m][crop], sc[c][crop]).correlation)}
            for g, lab in (("autistic", 1), ("td", 0)):
                k = y == lab
                d[g] = float(stats.spearmanr(sc[m][k], sc[c][k]).correlation)
                kc = k & crop
                d[f"{g}_cropped"] = float(stats.spearmanr(sc[m][kc], sc[c][kc]).correlation)
            cor[f"{m}|{c}"] = d
    out["spearman"] = cor
    return out


# ── D6 split-half reliability ─────────────────────────────────────────────────

def _video_half(seq, which):
    vm = valid_mask(seq)
    idx = np.flatnonzero(vm)
    a, b = idx[0], idx[-1] + 1
    mid = a + (b - a) // 2
    out = np.zeros_like(seq)
    lo, hi = (a, mid) if which == 0 else (mid, b)
    out[lo:hi] = seq[lo:hi]
    return prepare_sequence(out)


def _sb(rho):
    r = max(rho, 0.0)
    return 2 * r / (1 + r)


def reliability():
    import depth_skeleton as ds
    from feature_sets import raw_skeletons
    from gait_features import FEATURE_NAMES, feature_matrix
    S, y, age, male, crop, vid = cohort_info()
    prep = [prepare_sequence(raw(s)) for s in S]
    H = [feature_matrix([_video_half(a, w) for a in prep]) for w in (0, 1)]
    out = {"video": {}, "depth": {}, "pairs": {}}
    rv = {}
    for j, n in enumerate(FEATURE_NAMES):
        ok = np.isfinite(H[0][:, j]) & np.isfinite(H[1][:, j])
        rho = stats.spearmanr(H[0][ok, j], H[1][ok, j]).correlation
        rv[n] = _sb(rho)
        out["video"][n] = {"rho_halves": float(rho), "spearman_brown": rv[n], "n": int(ok.sum())}
    sk = raw_skeletons(S)
    D = {0: [], 1: []}
    for s in S:
        t, X = sk[s]
        m = len(t) // 2
        for w, sl in ((0, slice(0, m)), (1, slice(m, None))):
            f = ds.skeleton_features(t[sl], X[sl])
            D[w].append([f[k] for k in ds.FEATURE_NAMES])
    D = {w: np.array(v, float) for w, v in D.items()}
    rd = {}
    for j, n in enumerate(ds.FEATURE_NAMES):
        ok = np.isfinite(D[0][:, j]) & np.isfinite(D[1][:, j])
        rho = stats.spearmanr(D[0][ok, j], D[1][ok, j]).correlation
        rd[n] = _sb(rho)
        out["depth"][n] = {"rho_halves": float(rho), "spearman_brown": rd[n], "n": int(ok.sum()),
                           "frames_per_half_median": float(np.median([len(sk[s][0]) // 2 for s in S]))}
    M = json.load(open("results/cv/mechanisms.json"))
    conv = M["depth"]["video_depth_spearman"]
    robust = M["movement"]["robust_descriptors"]
    for dk, vk in DEPTH_MATCH.items():
        ceil = float(np.sqrt(rv[vk] * rd[dk]))
        rho = conv[dk]["rho"]
        out["pairs"][dk] = {"rho": rho, "r_video": rv[vk], "r_depth": rd[dk], "ceiling": ceil,
                            "disattenuated": float(rho / ceil) if ceil >= 0.2 else None,
                            "interpretable": bool(ceil >= 0.5), "robust_in_video": vk in robust}
    with_depth = [d for d in robust if d in DEPTH_MATCH.values()]
    agree = [d for d in with_depth if conv[[k for k, v in DEPTH_MATCH.items() if v == d][0]]["rho"] >= 0.3]
    out["denominator"] = {"robust_video": robust, "n_robust": len(robust),
                          "robust_with_depth_counterpart": with_depth, "n_with_depth": len(with_depth),
                          "agree_rho_ge_0.3": agree, "n_agree": len(agree),
                          "n_pairs_measured_in_both": len(DEPTH_MATCH)}
    return out


# ── D7 scale variants, D10 viewpoint ──────────────────────────────────────────

def _adjusted_set(X, names, y, covs, sub, B=2000):
    """Adjusted group effect (OLS with intercept, group and covariates) of
    every descriptor within `sub`; BH q over descriptors."""
    rows = []
    for j, n in enumerate(names):
        rows.append({"name": n, "adjusted": _ols_cov(X[sub, j], y[sub], covs[sub], B=B)})
    for r, q in zip(rows, _bh([r["adjusted"]["p"] for r in rows])):
        r["adjusted"]["q"] = float(q)
    return rows


def _ols_cov(x, group, covs, B=2000, seed=20261011):
    ok = np.isfinite(x) & np.all(np.isfinite(covs), 1)
    x, g, C = x[ok], group[ok].astype(float), covs[ok]
    z = (x - x.mean()) / (x.std(ddof=1) + 1e-12)
    X = np.c_[np.ones(len(z)), g, C]
    fit = lambda ix: np.linalg.lstsq(X[ix], z[ix], rcond=None)[0]
    beta = fit(np.arange(len(z)))
    resid = z - X @ beta
    dof = len(z) - X.shape[1]
    cov = np.linalg.pinv(X.T @ X) * (resid @ resid) / dof
    t = beta[1] / np.sqrt(cov[1, 1])
    p = float(2 * stats.t.sf(abs(t), dof))
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(g == 1), np.flatnonzero(g == 0)
    bs = [fit(np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]))[1] for _ in range(B)]
    return {"beta": float(beta[1]), "p": p, "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "n": int(len(z))}


def scale():
    from feature_sets import FeatureSets
    from gait_features import FEATURE_NAMES
    S, y, age, male, crop, vid = cohort_info()
    F = FeatureSets(S)
    M = json.load(open("results/cv/mechanisms.json"))
    robust = M["movement"]["robust_descriptors"]
    std = {r["name"]: r["adjusted"] for r in M["movement"]["descriptors_cropped"]["standard"]}
    covs = np.c_[age, male.astype(float)]
    out = {"robust_standard": robust, "variants": {}}
    for kind in ("gait_torso", "gait_median"):
        X = F.matrix(kind)
        rows = _adjusted_set(X, FEATURE_NAMES, y, covs, crop)
        eff = {r["name"]: r["adjusted"] for r in rows}
        out["variants"][kind] = {"descriptors": rows,
                                 "significant_q05": [n for n in FEATURE_NAMES if eff[n]["q"] < 0.05]}
    keep = []
    for n in robust:
        ok = True
        for kind in ("gait_torso", "gait_median"):
            e = {r["name"]: r["adjusted"] for r in out["variants"][kind]["descriptors"]}[n]
            ok &= bool(np.sign(e["beta"]) == np.sign(std[n]["beta"]) and (e["ci95"][0] > 0 or e["ci95"][1] < 0))
        if ok:
            keep.append(n)
    out["scale_robust"] = keep
    return out


def scale_noise():
    """Descriptive (not pre-specified): second-difference noise and the
    9-frame residual of the prepared coordinates under the per-frame
    shoulder scale and the two alternative scales of D7, and the share of
    consecutive detected frames whose shoulder distance changes by more than
    30% (from the pose quality control)."""
    from feature_sets import scale_variant, stored
    from recording_features import jitter_features, second_difference_noise
    S, y, *_ = cohort_info()
    out = {}
    for name, fn in (("shoulder_per_frame", lambda s: prepare_sequence(stored(s))),
                     ("shoulder_median_1s", lambda s: scale_variant(s, "shoulder_median")),
                     ("torso", lambda s: scale_variant(s, "torso"))):
        n2 = np.array([second_difference_noise(fn(s))["noise2_all"] for s in S])
        jt = np.array([jitter_features(fn(s))["jitter_all"] for s in S])
        out[name] = {"noise2": summ(n2), "jitter9": summ(jt),
                     "noise2_autistic": float(np.nanmedian(n2[y == 1])), "noise2_td": float(np.nanmedian(n2[y == 0]))}
    q = json.load(open(os.path.join(OUT, "pose_qc.json")))["per_child"]
    out["scale_jump_share"] = summ([q[s]["scale_jump"] / max(q[s]["n_detected"] - 1, 1) for s in S])
    return out


def viewpoint():
    from feature_sets import FeatureSets
    from gait_features import FEATURE_NAMES
    S, y, age, male, crop, vid = cohort_info()
    F = FeatureSets(S)
    from recording_features import SCENE_NAMES
    size = F.matrix("scene")[:, SCENE_NAMES.index("shoulder_px_rel_height")]
    X = F.matrix("gait")
    covs = np.c_[age, male.astype(float), size]
    rows = _adjusted_set(X, FEATURE_NAMES, y, covs, crop)
    M = json.load(open("results/cv/mechanisms.json"))
    robust = M["movement"]["robust_descriptors"]
    std = {r["name"]: r["adjusted"] for r in M["movement"]["descriptors_cropped"]["standard"]}
    eff = {r["name"]: r["adjusted"] for r in rows}
    still = [n for n in robust if eff[n]["q"] < 0.05 and np.sign(eff[n]["beta"]) == np.sign(std[n]["beta"])]
    return {"descriptors": rows, "robust_standard": robust, "robust_after_size_adjustment": still,
            "size_auc_cropped": _auc_ci(y[crop], size[crop]),
            "size_median": {"autistic_cropped": float(np.median(size[crop & (y == 1)])),
                            "td": float(np.median(size[y == 0]))}}


# ── D11 age-consistent subset ─────────────────────────────────────────────────

ARTICLE_AGES = {1: (4, 12), 0: (6, 11)}


def age_subset():
    from feature_sets import FeatureSets
    from gait_features import FEATURE_NAMES
    S, y, age, male, crop, vid = cohort_info()
    ok = np.array([ARTICLE_AGES[l][0] <= a <= ARTICLE_AGES[l][1] for a, l in zip(age, y)])
    out = {"article_ranges": {"autistic": ARTICLE_AGES[1], "td": ARTICLE_AGES[0]},
           "n": {"autistic": int((ok & (y == 1)).sum()), "td": int((ok & (y == 0)).sum()),
                 "autistic_cropped": int((ok & crop & (y == 1)).sum()), "td_cropped": int((ok & crop & (y == 0)).sum())},
           "models": {}}
    for name in sorted(os.listdir("results/cv/runs")):
        if name == "pilot":
            continue
        try:
            P = mat(oof(name), S, 3)
        except (KeyError, IndexError):
            continue
        r = {}
        for k, m in (("subset", ok), ("subset_cropped", ok & crop)):
            idx = np.flatnonzero(m)
            r[k] = float(np.mean([fast_auc(y[idx], P[q, idx]) for q in range(len(P))]))
        out["models"][name] = r
    F = FeatureSets(S)
    X = F.matrix("gait")
    sub = ok & crop
    out["descriptors_sex_only"] = _adjusted_set(X, FEATURE_NAMES, y, male.astype(float)[:, None], sub)
    out["descriptors_age_sex"] = _adjusted_set(X, FEATURE_NAMES, y, np.c_[age, male.astype(float)], sub)
    return out


# ── D12 training within the cropped videos ────────────────────────────────────

WITHIN = ["Gait-LR", "Gait-RF", "Gait-LR-adjusted", "Mean-pose", "Depth-LR", "Depth-RF",
          "Scene", "Time-of-day", "Demographics", "Tracking-quality", "All-controls-noformat"]


def within_cropped():
    from feature_sets import FeatureSets, fit_predict
    from phase3_common import nested_partition
    import run_feature_models as rfm
    S, y, age, male, crop, vid = cohort_info()
    Sc = [s for s, c in zip(S, crop) if c]
    yc = y[crop]
    F = FeatureSets(Sc)
    # All-controls without frame format (all cropped), as a separate model
    F._cache["all_controls_noformat"] = {s: np.concatenate([F.get(k)[s] for k in
                                         ("demo", "mask", "duration", "scene", "tracking", "depth_recording")])
                                         for s in Sc}
    _models = rfm.models

    def models(seed):
        m = _models(seed)
        m["All-controls-noformat"] = (m["All-controls"][0], m["All-controls"][1], "all_controls_noformat")
        return m
    rfm.models = models
    probs = {m: [] for m in WITHIN}
    for r in range(3):
        part = nested_partition(Sc, yc, 20261104 + r)
        for m in WITHIN:
            probs[m].append(fit_predict(m, F, part))
        print(f"[within cropped] repetition {r}", flush=True)
    rfm.models = _models
    out = {"n": {"autistic": int(yc.sum()), "td": int((1 - yc).sum())}, "models": {}}
    P = {m: np.array([[d[s] for s in Sc] for d in v]) for m, v in probs.items()}
    rng = np.random.default_rng(20261107)
    pos, neg = np.flatnonzero(yc == 1), np.flatnonzero(yc == 0)
    draws = {m: [] for m in P}
    for _ in range(2000):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        for m in P:
            draws[m].append(np.mean([fast_auc(yc[i], P[m][q, i]) for q in range(3)]))
    for m in P:
        est = float(np.mean([fast_auc(yc, P[m][q]) for q in range(3)]))
        out["models"][m] = {"auc": est, "ci95": [float(np.percentile(draws[m], 2.5)), float(np.percentile(draws[m], 97.5))]}
    for m in ("Gait-LR", "Gait-RF", "Mean-pose", "Depth-RF"):
        d = np.array(draws[m]) - np.array(draws["All-controls-noformat"])
        out["models"][m]["minus_all_controls"] = {"diff": out["models"][m]["auc"] - out["models"]["All-controls-noformat"]["auc"],
                                                  "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    out["probs"] = {m: [[float(v) for v in row] for row in P[m]] for m in P}
    out["subjects"] = Sc
    return out


# ── D9 frame-order shuffle ────────────────────────────────────────────────────

def _shuffle_valid(seq, rng):
    vm = valid_mask(seq)
    idx = np.flatnonzero(vm)
    out = seq.copy()
    out[idx] = seq[rng.permutation(idx)]
    return out


def shuffle():
    import torch
    from analyze_mechanisms import load_ckpt
    S, y, age, male, crop, vid = cohort_info()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos = {s: i for i, s in enumerate(S)}
    out = {}
    for arm in ("PACE", "PACE-nogate"):
        files = sorted(glob.glob(os.path.join("models", "cv", arm, "*.pt")))
        if not files:
            continue
        # probs[variant][repeat] -> list over inner models of {subject: p}
        acc = {v: np.zeros((3, len(S))) for v in ["original"] + [f"shuffle{k}" for k in range(5)]}
        cnt = np.zeros((3, len(S)))
        for f in files:
            m, ck = load_ckpt(f, dev)
            T, b = ck["scaler"]["temperature"], ck["scaler"]["bias"]
            test = ck["test"]
            xs = [prepare_sequence(raw(s)) for s in test]
            for v in acc:
                if v == "original":
                    X = xs
                else:
                    k = int(v[-1])
                    X = [_shuffle_valid(x, np.random.default_rng(20261103 + k + 1000 * pos[s])) for x, s in zip(xs, test)]
                with torch.no_grad():
                    lg = m(torch.from_numpy(np.stack(X)).to(dev))[1].float().cpu().numpy()
                p = 1 / (1 + np.exp(-(T * lg + b)))
                for s, pp in zip(test, p):
                    acc[v][ck["repeat"], pos[s]] += pp
            for s in test:
                cnt[ck["repeat"], pos[s]] += 1
        P = {v: a / cnt for v, a in acc.items()}
        res = {}
        base = P["original"]
        stored_P = mat(oof(arm), S, 3)
        res["max_abs_diff_vs_stored"] = float(np.abs(base - stored_P).max())
        sh = np.mean([P[f"shuffle{k}"] for k in range(5)], 0)
        rng = np.random.default_rng(20261108)
        pos1, neg1 = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        for name, idx in (("all", np.arange(len(S))), ("cropped", np.flatnonzero(crop))):
            yy = y[idx]
            a0 = float(np.mean([fast_auc(yy, base[q, idx]) for q in range(3)]))
            a1 = float(np.mean([fast_auc(yy, sh[q, idx]) for q in range(3)]))
            pp, nn = np.flatnonzero(yy == 1), np.flatnonzero(yy == 0)
            d = []
            for _ in range(2000):
                i = np.concatenate([rng.choice(pp, len(pp)), rng.choice(nn, len(nn))])
                d.append(np.mean([fast_auc(yy[i], base[q, idx][i]) - fast_auc(yy[i], sh[q, idx][i]) for q in range(3)]))
            res[name] = {"auc_original": a0, "auc_shuffled": a1, "drop": a0 - a1,
                         "drop_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                         "per_shuffle": [float(np.mean([fast_auc(yy, P[f"shuffle{k}"][q, idx]) for q in range(3)]))
                                         for k in range(5)]}
            res[name]["uses_order"] = bool(res[name]["drop"] > 0.05 and res[name]["drop_ci95"][0] > 0)
        out[arm] = res
        print(f"[shuffle] {arm}: {res['all']['auc_original']:.3f} -> {res['all']['auc_shuffled']:.3f}", flush=True)
    return out


# ── D13 integrated gradients and temporal occlusion (real data) ───────────────

STREAMS = {"position": slice(0, 66), "velocity": slice(66, 132), "acceleration": slice(132, 198)}
REGIONS = {"head": list(range(0, 11)), "arms_hands": list(range(11, 23)), "hips": [23, 24],
           "legs_feet": list(range(25, 33))}


def integrated_gradients(model, x, steps=32):
    """|IG| of the logit with respect to the 198-d descriptor at the encoder
    input, zero baseline, midpoint rule; shape (B, T, 198)."""
    import torch
    model.eval()
    layer = model.spatial_encoder.input_proj[0]
    store = {}

    def capture(mod, args):
        store["f"] = args[0].detach().clone()
        return None
    h = layer.register_forward_pre_hook(capture)
    with torch.no_grad():
        model(x)
    h.remove()
    f = store["f"]
    total = torch.zeros_like(f)
    for k in range(steps):
        a = (k + 0.5) / steps

        def scaled(mod, args, a=a):
            z = (a * f).requires_grad_(True)
            store["z"] = z
            return (z,)
        h = layer.register_forward_pre_hook(scaled)
        try:
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                _, lg = model(x)
                lg.sum().backward()
            total += store["z"].grad.detach()
        finally:
            h.remove()
    ig = (f * total / steps).abs()
    B, T = x.shape[:2]
    return ig.reshape(B, T, -1).cpu().numpy()


def temporal_occlusion(model, x_np, block=15):
    """|logit change| when each 15-frame block of an aligned sequence is set
    to undetected; shares over the valid blocks."""
    import torch
    vm = valid_mask(x_np)
    nb = int(np.ceil(len(vm) / block))
    blocks = [b for b in range(nb) if vm[b * block:(b + 1) * block].any()]
    variants = [x_np]
    for b in blocks:
        z = x_np.copy(); z[b * block:(b + 1) * block] = 0.0
        variants.append(z)
    dev = next(model.parameters()).device
    with torch.no_grad():
        lg = model(torch.from_numpy(np.stack(variants)).to(dev))[1].float().cpu().numpy()
    d = np.abs(lg[1:] - lg[0])
    return blocks, d / (d.sum() + 1e-12)


def ig():
    import torch
    from analyze_mechanisms import load_ckpt
    from protocol import build_model, set_seed
    S, y, *_ = cohort_info()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = sorted(glob.glob(os.path.join("models", "cv", "PACE", "*.pt")))
    sh = {"trained": {k: [] for k in STREAMS}, "random": {k: [] for k in STREAMS}}
    rg = {"trained": {k: [] for k in REGIONS}, "random": {k: [] for k in REGIONS}}
    prof_corr, occl_corr, occl_motion = [], [], {"trained": [], "random": []}
    for n, f in enumerate(files):
        m, ck = load_ckpt(f, dev)
        set_seed(20261011 + n)
        rnd = build_model(ck["arm"], ck["config"]).to(dev).eval()
        test = ck["test"]
        xs_np = [prepare_sequence(raw(s)) for s in test]
        xs = torch.from_numpy(np.stack(xs_np)).to(dev)
        vm = np.stack([valid_mask(a) for a in xs_np])
        profs = {}
        for kind, model in (("trained", m), ("random", rnd)):
            g = integrated_gradients(model, xs)
            pr = []
            for i in range(len(test)):
                gi = g[i][vm[i]]
                tot = gi.sum() + 1e-12
                for k, sl in STREAMS.items():
                    sh[kind][k].append(gi[:, sl].sum() / tot)
                lm = gi.reshape(-1, 3, 33, 2).sum(axis=(0, 3))[1:].sum(0)
                for k, j in REGIONS.items():
                    rg[kind][k].append(lm[j].sum() / (lm.sum() + 1e-12))
                pr.append(lm / (lm.sum() + 1e-12))
            profs[kind] = np.mean(pr, 0)
        prof_corr.append(stats.spearmanr(profs["trained"], profs["random"]).correlation)
        # temporal occlusion: per child, profile of trained vs random and vs motion energy
        for i, s in enumerate(test):
            a = xs_np[i]
            bl, ot = temporal_occlusion(m, a)
            _, orr = temporal_occlusion(rnd, a)
            if len(bl) >= 4:
                occl_corr.append(stats.spearmanr(ot, orr).correlation)
                e = np.r_[0, np.linalg.norm(a[1:] - a[:-1], axis=-1).sum(-1)]
                me = [e[b * 15:(b + 1) * 15].mean() for b in bl]
                occl_motion["trained"].append(stats.spearmanr(ot, me).correlation)
                occl_motion["random"].append(stats.spearmanr(orr, me).correlation)
        print(f"[ig] {n + 1}/{len(files)}", flush=True)
    out = {"n_models": len(files), "steps": 32,
           "stream_share": {k: summ(v) for k, v in sh["trained"].items()},
           "random_init_stream_share": {k: summ(v) for k, v in sh["random"].items()},
           "region_share_velocity_acceleration": {k: summ(v) for k, v in rg["trained"].items()},
           "random_init_region_share_velocity_acceleration": {k: summ(v) for k, v in rg["random"].items()},
           "trained_vs_random_landmark_profile_spearman": summ(prof_corr),
           "occlusion_trained_vs_random_spearman": summ([c for c in occl_corr if np.isfinite(c)]),
           "occlusion_vs_motion_energy_spearman": {k: summ([c for c in v if np.isfinite(c)])
                                                   for k, v in occl_motion.items()}}
    diffs = [abs(out["stream_share"][k]["mean"] - out["random_init_stream_share"][k]["mean"]) for k in STREAMS]
    diffs += [abs(out["region_share_velocity_acceleration"][k]["mean"]
                  - out["random_init_region_share_velocity_acceleration"][k]["mean"]) for k in REGIONS]
    out["criterion"] = {"max_share_difference": float(max(diffs)),
                        "profile_rho": out["trained_vs_random_landmark_profile_spearman"]["mean"],
                        "passes": bool(max(diffs) > 0.05 and out["trained_vs_random_landmark_profile_spearman"]["mean"] < 0.5)}
    return out


# ── exploratory: the visual audit of every recording ─────────────────────────

def visual():
    """Summary of results/phase3/qc/visual_audit.csv (setting, whether the
    child walks towards the camera, other people visible, gross tracking
    errors; annotated from overlays of the first, middle and last detected
    frame) by group and frame format, and the AUC of every stored model
    within the children who walk towards the camera. Not pre-specified."""
    S, y, age, male, crop, vid = cohort_info()
    # fourth phase (E7): the blinded two-annotator consensus once it exists
    from visual_audit_phase4 import annotation_table
    a, source = annotation_table()
    walk = np.array([a.loc[s, "walks_toward_camera"] == "yes" for s in S])
    other = np.array([a.loc[s, "other_person_visible"] == "yes" for s in S])
    gross = np.array([a.loc[s, "gross_tracking_error"] == "yes" for s in S])
    sett = np.array([a.loc[s, "setting"] for s in S])
    out = {"source": source, "by_group": {}}
    for name, m in (("autistic", y == 1), ("td", y == 0), ("autistic_cropped", (y == 1) & crop),
                    ("autistic_uncropped", (y == 1) & ~crop)):
        out["by_group"][name] = {"n": int(m.sum()), "n_settings": int(len(set(sett[m]))),
                                 "settings": sorted(set(sett[m])), "walks_toward_camera": int(walk[m].sum()),
                                 "other_person_visible": int(other[m].sum()), "gross_tracking_error": int(gross[m].sum())}
    out["shared_settings"] = sorted(set(sett[y == 1]) & set(sett[y == 0]))
    out["n_walk"] = {"autistic": int((walk & (y == 1)).sum()), "td": int((walk & (y == 0)).sum()),
                     "autistic_cropped": int((walk & crop & (y == 1)).sum())}
    models = {}
    for name in sorted(os.listdir("results/cv/runs")):
        if name == "pilot":
            continue
        try:
            P = mat(oof(name), S, 3)
        except (KeyError, IndexError):
            continue
        idx = np.flatnonzero(walk)
        models[name] = float(np.mean([fast_auc(y[idx], P[q, idx]) for q in range(len(P))]))
    out["auc_walking_subset"] = models
    return out


SECTIONS = {"metadata": metadata, "tracking": tracking, "incremental": incremental,
            "reliability": reliability, "scale": scale, "viewpoint": viewpoint, "age_subset": age_subset,
            "within_cropped": within_cropped, "shuffle": shuffle, "ig": ig, "visual": visual,
            "scale_noise": scale_noise}


def main():
    """Each section is written to results/phase3/analyses/<section>.json, so
    that sections can run in parallel processes."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", nargs="*", default=None)
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    for name, fn in SECTIONS.items():
        if a.sections is None or name in a.sections:
            t0 = time.time()
            json.dump(fn(), open(os.path.join(RES, f"{name}.json"), "w"), indent=1)
            print(f"{name} done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
