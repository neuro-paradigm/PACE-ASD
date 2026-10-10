"""
PACE — fifth-phase analyses (docs/validation_criteria_phase5.md).

  calibration   F1, F2: every calibration cell with exact (Clopper-Pearson)
                95% intervals, 100 splits in the extended cells, and the
                single-channel outcome (criterion (v) unavailable)
                -> results/phase5/calibration.json
  scale         F3: the test-case network (run_cv.py --tag PACE-median), the
                mean-pose classifier, frame order and channel zeroing on the
                corrected body scale (1-s running median of the shoulder
                distance) -> results/phase5/scale.json
  depth         F4: video and depth descriptors matched for projection, frame
                rate and duration -> results/phase5/depth_matched.json

The splits of F1 are run by scripts/phase4_calibration.py --run --phase5 and
PACE-median by
    python scripts/run_cv.py --arm PACE --tag PACE-median --set data.scale=shoulder_median --save_checkpoints

Usage:
    python scripts/phase5_analyses.py --sections calibration scale depth
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)

from analyze_mechanisms import boot_auc, fast_auc, mat, oof, part_info  # noqa: E402
from sequence import prepare_sequence, valid_mask                      # noqa: E402

OUT5 = os.path.join(ROOT, "results", "phase5")
R = 3


def clopper_pearson(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return [lo, hi]


# ── F1, F2: calibration with exact intervals; single-channel outcome ──────────

def calibration():
    import phase4_calibration as pc
    td = pc.td_children()
    variants = pc.all_variants()
    cells = [(scn, n) for scn in pc.scenarios() for n in (pc.SIZES if scn.startswith("pc_") else [45])]
    cells.append(("n3", 45))
    null_cells = {("pc_g1", 20), ("pc_g1", 30), ("pc_g1", 45), ("n1", 45), ("n2", 45), ("n3", 45)}
    keys = ["i", "ii", "iii", "iv", "v"]
    out = {"cells": {}}
    for scn, n in cells:
        rows = []
        for j in range(pc.n_splits(scn, n)):
            ev = pc.evaluated(scn, n, j, variants, td)
            if ev is not None:
                c = ev["criteria"]
                rows.append({"split": j, "grade": ev["grade"], "criteria": {k: bool(c[k]) for k in keys},
                             "single_channel": "S" if (c["i"] and c["ii"] and c["iii"] and c["iv"]) else "C",
                             "auc": ev["auc"]})
        if not rows:
            continue
        N = len(rows)

        def count(fn, sub=rows):
            k = int(sum(bool(fn(r)) for r in sub))
            return {"k": k, "n": len(sub), "share": k / len(sub), "ci95": clopper_pearson(k, len(sub))}

        cell = {"scenario": scn, "n": n, "n_splits": N,
                "A": count(lambda r: r["grade"] == "A"),
                "AB": count(lambda r: r["grade"] in ("A", "B")),
                "S": count(lambda r: r["single_channel"] == "S"),
                "criteria": {k: count(lambda r, k=k: r["criteria"][k]) for k in keys},
                "auc_mean": {m: float(np.mean([r["auc"][m] for r in rows])) for m in pc.MODELS}}
        if N > pc.N_SPLITS:
            first, later = rows[:pc.N_SPLITS], rows[pc.N_SPLITS:]
            cell["first20"] = {g: count(fn, first) for g, fn in
                               (("A", lambda r: r["grade"] == "A"), ("AB", lambda r: r["grade"] in ("A", "B")))}
            cell["later"] = {g: count(fn, later) for g, fn in
                             (("A", lambda r: r["grade"] == "A"), ("AB", lambda r: r["grade"] in ("A", "B")))}
        if (scn, n) in null_cells:
            for g in ("A", "AB"):
                cell[f"spec_{g}"] = {"met_point": cell[g]["share"] <= 0.10,
                                     "met_interval": cell[g]["ci95"][1] <= 0.10}
        if (scn, n) == ("pc_g4", 45):
            cell["sens_A"] = {"met_point": cell["A"]["share"] >= 0.80, "met_interval": cell["A"]["ci95"][0] >= 0.80}
        out["cells"][f"{scn}|n{n}"] = cell
        print(f"[calibration] {scn} n={n}: {N} splits, A {cell['A']['k']}, AB {cell['AB']['k']}, "
              f"S {cell['S']['k']}", flush=True)
    nulls = [out["cells"][f"{s}|n{n}"] for s, n in sorted(null_cells) if f"{s}|n{n}" in out["cells"]]
    out["specificity"] = {g: {"met_point": all(c[f"spec_{g}"]["met_point"] for c in nulls),
                              "met_interval": all(c[f"spec_{g}"]["met_interval"] for c in nulls)}
                          for g in ("A", "AB")}
    a, b = out["cells"].get("pc_g4|n45"), out["cells"].get("n3|n45")
    if a and b:
        out["single_channel_n3_equals_g4"] = bool(a["S"]["k"] == b["S"]["k"] and a["S"]["n"] == b["S"]["n"])
    return out


# ── F3: corrected body scale ──────────────────────────────────────────────────

def _median_arrays(S):
    from feature_sets import scale_variant
    return {s: scale_variant(s, "shoulder_median") for s in S}


def _shuffle_valid(seq, rng):
    vm = valid_mask(seq)
    idx = np.flatnonzero(vm)
    out = seq.copy()
    out[idx] = seq[rng.permutation(idx)]
    return out


def _network_checks(arm, arrays, S, y, crop):
    """Frame-order test (as phase3_analyses.shuffle) and channel zeroing (as
    analyze_mechanisms.attribution) for the checkpoints of `arm`."""
    import torch
    from analyze_mechanisms import STREAMS, load_ckpt
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = sorted(glob.glob(os.path.join("models", "cv", arm, "*.pt")))
    pos = {s: i for i, s in enumerate(S)}
    acc = {v: np.zeros((R, len(S))) for v in ["original"] + [f"shuffle{k}" for k in range(5)]}
    cnt = np.zeros((R, len(S)))
    occl = defaultdict(lambda: defaultdict(dict))
    for f in files:
        m, ck = load_ckpt(f, dev)
        T, b = ck["scaler"]["temperature"], ck["scaler"]["bias"]
        test = ck["test"]
        xs = [arrays[s] for s in test]
        for v in acc:
            X = xs if v == "original" else [
                _shuffle_valid(x, np.random.default_rng(20261103 + int(v[-1]) + 1000 * pos[s])) for x, s in zip(xs, test)]
            with torch.no_grad():
                lg = m(torch.from_numpy(np.stack(X)).to(dev))[1].float().cpu().numpy()
            for s, pp in zip(test, 1 / (1 + np.exp(-(T * lg + b)))):
                acc[v][ck["repeat"], pos[s]] += pp
        for s in test:
            cnt[ck["repeat"], pos[s]] += 1
        xt = torch.from_numpy(np.stack(xs)).to(dev)
        layer = m.spatial_encoder.input_proj[0]
        for k, sl in [("none", None), *STREAMS.items()]:
            def hook(mod, args, sl=sl):
                if sl is None:
                    return None
                f_ = args[0].clone(); f_[:, sl] = 0.0
                return (f_,)
            h = layer.register_forward_pre_hook(hook)
            with torch.no_grad():
                lg = m(xt)[1].float().cpu().numpy()
            h.remove()
            occl[k][(ck["repeat"], ck["inner_fold"])].update(zip(test, lg))
    P = {v: a / cnt for v, a in acc.items()}
    stored = mat(oof(arm), S, R)
    res = {"n_checkpoints": len(files), "max_abs_diff_vs_stored": float(np.abs(P["original"] - stored).max())}
    sh = np.mean([P[f"shuffle{k}"] for k in range(5)], 0)
    rng = np.random.default_rng(20261108)
    for name, idx in (("all", np.arange(len(S))), ("cropped", np.flatnonzero(crop))):
        yy = y[idx]
        a0 = float(np.mean([fast_auc(yy, P["original"][q, idx]) for q in range(R)]))
        a1 = float(np.mean([fast_auc(yy, sh[q, idx]) for q in range(R)]))
        pp, nn = np.flatnonzero(yy == 1), np.flatnonzero(yy == 0)
        d = []
        for _ in range(2000):
            i = np.concatenate([rng.choice(pp, len(pp)), rng.choice(nn, len(nn))])
            d.append(np.mean([fast_auc(yy[i], P["original"][q, idx][i]) - fast_auc(yy[i], sh[q, idx][i])
                              for q in range(R)]))
        ci = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
        res[f"shuffle_{name}"] = {"auc_original": a0, "auc_shuffled": a1, "drop": a0 - a1, "drop_ci95": ci,
                                  "uses_order": bool(a0 - a1 > 0.05 and ci[0] > 0)}
    occ = {}
    for k, d in occl.items():
        aucs = [fast_auc(y, [vals[s] for s in S]) for vals in d.values() if len(vals) == len(S)]
        occ[k] = {"mean": float(np.mean(aucs)), "n": len(aucs)}
    res["zeroing"] = occ
    res["zeroing_drop"] = {k: occ["none"]["mean"] - occ[k]["mean"] for k in STREAMS}
    return res


def scale():
    import pandas as pd
    from feature_sets import FeatureSets
    from phase4_common import fit_predict_nested
    p, S, y = part_info()
    vid = pd.read_csv("processed/video_metadata.csv").set_index("clip_id")
    crop = np.array([not (vid.loc[s, "width"] == 1080 and vid.loc[s, "height"] == 1920) for s in S])
    arrays = _median_arrays(S)
    # Mean-pose-median under the frozen partition (three repetitions)
    F = FeatureSets(S, prep=arrays)
    mp = {}
    for r in range(R):
        o, _, prm = fit_predict_nested("Mean-pose", F, p, r=r)
        mp[r] = o
    os.makedirs(OUT5, exist_ok=True)
    json.dump({str(r): v for r, v in mp.items()}, open(os.path.join(OUT5, "mean_pose_median_oof.json"), "w"))
    mats = {"Mean-pose-median": mat(mp, S, R)}
    for name in ("PACE-median", "Gait-LR-median", "Gait-RF-median", "PACE", "Mean-pose", "Gait-LR", "Gait-RF"):
        mats[name] = mat(oof(name), S, R)
    pairs = [("PACE-median", "PACE"), ("Mean-pose-median", "Mean-pose"), ("PACE-median", "Mean-pose-median")]
    res = {"auc_all": boot_auc(y, mats, pairs=pairs), "auc_cropped": boot_auc(y, mats, np.flatnonzero(crop))}
    res["network"] = _network_checks("PACE-median", arrays, S, y, crop)
    # wording rule (F3)
    a = {k: v["auc"] for k, v in res["auc_all"].items() if " - " not in k}
    best = max(("PACE-median", "Gait-LR-median", "Gait-RF-median"), key=lambda k: a[k])
    order = res["network"]["shuffle_all"]["uses_order"]
    velacc = bool(max(res["network"]["zeroing_drop"]["velocity"], res["network"]["zeroing_drop"]["acceleration"]) > 0.05)
    posture = bool(a["Mean-pose-median"] >= a[best] - 0.05)
    if posture and not (order or velacc):
        wording = "mostly posture"
    elif (order or velacc) and not posture:
        wording = "movement over time"
    else:
        wording = "both"
    res["rule"] = {"best_movement_model": best, "uses_order": order, "uses_velocity_or_acceleration": velacc,
                   "posture_suffices": posture, "wording": wording}
    print(f"[scale] {json.dumps(res['rule'])}", flush=True)
    return res


# ── F4: second sensor matched for projection, frame rate and duration ─────────

DEPTH_TO_LANDMARK = {0: "Head", 11: "ShoulderLeft", 12: "ShoulderRight", 13: "ElbowLeft", 14: "ElbowRight",
                     15: "WristLeft", 16: "WristRight", 23: "HipLeft", 24: "HipRight", 25: "KneeLeft",
                     26: "KneeRight", 27: "AnkleLeft", 28: "AnkleRight"}


def projected_depth(t, X, fps=60.0, T=300):
    """Depth joints (camera coordinates) -> a prepared (T, 33, 2) sequence in
    the landmark layout of the video: perspective projection onto the image
    plane of a camera facing the child (x = -X/Z, y = -Y/Z), linear
    resampling to `fps` on the recorded time stamps, hip midpoint subtracted
    and divided by the per-frame shoulder distance (Eq. 1)."""
    import depth_skeleton as ds
    t = np.asarray(t, float) - float(t[0])
    grid = np.arange(0.0, t[-1] + 1e-9, 1.0 / fps)
    uv = np.stack([-X[..., 0] / X[..., 2], -X[..., 1] / X[..., 2]], -1)        # (n, 25, 2)
    out = np.zeros((len(grid), 33, 2))
    hip = None
    for lm, name in DEPTH_TO_LANDMARK.items():
        j = ds.J[name]
        out[:, lm] = np.stack([np.interp(grid, t, uv[:, j, c]) for c in (0, 1)], -1)
    hip = (out[:, 23] + out[:, 24]) / 2
    mapped = list(DEPTH_TO_LANDMARK)
    for lm in range(33):
        if lm not in mapped:
            out[:, lm] = hip
    sd = np.linalg.norm(out[:, 11] - out[:, 12], axis=-1)
    out = (out - hip[:, None]) / np.maximum(sd, 1e-5)[:, None, None]
    seq = np.zeros((T, 33, 2), np.float32)
    seq[:min(T, len(out))] = out[:T]
    return prepare_sequence(seq), len(grid)


def _window_descriptors(a, L, which=None, step=10):
    """Median over every window of L frames of the detected span (start
    every `step` frames) of the 26 descriptors; `which` = 0/1 keeps only the
    windows whose centre lies in the first/second half of the span."""
    from gait_features import feature_matrix
    idx = np.flatnonzero(valid_mask(a))
    a0, a1 = int(idx[0]), int(idx[-1]) + 1
    starts = list(range(a0, max(a0, a1 - L) + 1, step))
    mid = a0 + (a1 - a0) / 2
    if which is not None:
        starts = [s0 for s0 in starts if ((s0 + min(L, a1 - a0) / 2) < mid) == (which == 0)]
    if not starts:
        return np.full(26, np.nan)
    W = []
    for s0 in starts:
        w = np.zeros_like(a)
        w[s0:min(s0 + L, a1)] = a[s0:min(s0 + L, a1)]
        W.append(prepare_sequence(w))
    return np.nanmedian(feature_matrix(W), 0)


def _sb(rho):
    r = max(rho, 0.0)
    return 2 * r / (1 + r)


def _spearman(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(stats.spearmanr(a[ok], b[ok]).correlation), int(ok.sum())


def depth():
    import depth_skeleton as ds
    from feature_sets import raw_skeletons, stored
    from gait_features import FEATURE_NAMES, feature_matrix
    p, S, y = part_info()
    sk = raw_skeletons(S)
    rel3 = json.load(open("results/phase3/analyses/reliability.json"))
    robust = rel3["denominator"]["robust_video"]
    D, Dh, Lmap = [], {0: [], 1: []}, {}
    for s in S:
        t, X = sk[s]
        a, L = projected_depth(t, X)
        Lmap[s] = L
        D.append(feature_matrix([a])[0])
        m = len(t) // 2
        for w, sl in ((0, slice(0, m)), (1, slice(m, None))):
            Dh[w].append(feature_matrix([projected_depth(t[sl], X[sl])[0]])[0])
    D = np.array(D)
    Dh = {w: np.array(v) for w, v in Dh.items()}
    prep = {s: prepare_sequence(stored(s)) for s in S}
    Vw = feature_matrix([prep[s] for s in S])                                  # whole clip
    Vm = np.array([_window_descriptors(prep[s], Lmap[s]) for s in S])          # duration-matched
    Vh = {w: np.array([_window_descriptors(prep[s], Lmap[s], which=w) for s in S]) for w in (0, 1)}
    out = {"n_children": len(S), "depth_frames_60fps": {"median": float(np.median(list(Lmap.values()))),
                                                         "min": int(min(Lmap.values())), "max": int(max(Lmap.values()))},
           "video_detected_frames": {"median": float(np.median([valid_mask(prep[s]).sum() for s in S]))},
           "robust_video": robust, "descriptors": {}}
    groups = {"all": np.ones(len(S), bool), "autistic": y == 1, "td": y == 0}
    for j, name in enumerate(FEATURE_NAMES):
        r_d, n_d = _spearman(Dh[0][:, j], Dh[1][:, j])
        r_v, n_v = _spearman(Vh[0][:, j], Vh[1][:, j])
        rd, rv = _sb(r_d), _sb(r_v)
        row = {"r_depth": rd, "r_video": rv, "ceiling": float(np.sqrt(rv * rd)), "n_rel_depth": n_d, "n_rel_video": n_v}
        for g, sel in groups.items():
            rm, nm = _spearman(Vm[sel, j], D[sel, j])
            rw, nw = _spearman(Vw[sel, j], D[sel, j])
            row[g] = {"rho_matched": rm, "n": nm, "rho_whole_clip": rw}
        row["agrees"] = bool(row["all"]["rho_matched"] >= 0.3)
        row["interpretable"] = bool(row["ceiling"] >= 0.5)
        row["robust_in_video"] = name in robust
        row["depth_auc_all"] = fast_auc(y[np.isfinite(D[:, j])], D[np.isfinite(D[:, j]), j])
        out["descriptors"][name] = row
    rb = {k: out["descriptors"][k] for k in robust}
    out["summary"] = {"n_robust": len(robust),
                      "agree": [k for k, v in rb.items() if v["agrees"]],
                      "disagree_interpretable": [k for k, v in rb.items() if not v["agrees"] and v["interpretable"]],
                      "disagree_not_interpretable": [k for k, v in rb.items() if not v["agrees"] and not v["interpretable"]],
                      "n_all_agree": int(sum(v["agrees"] for v in out["descriptors"].values()))}
    print(f"[depth] {json.dumps(out['summary'])}", flush=True)
    return out


SECTIONS = {"calibration": ("calibration.json", calibration), "scale": ("scale.json", scale),
            "depth": ("depth_matched.json", depth)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", nargs="*", default=list(SECTIONS))
    a = ap.parse_args()
    os.makedirs(OUT5, exist_ok=True)
    for name in a.sections:
        fname, fn = SECTIONS[name]
        res = fn()
        json.dump(res, open(os.path.join(OUT5, fname), "w"), indent=1)
        print(f"[phase5] wrote {fname}", flush=True)


if __name__ == "__main__":
    main()
