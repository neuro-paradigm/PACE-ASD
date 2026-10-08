"""
PACE — apply the pass/fail criteria of docs/validation_criteria.md to the
stored results and write results/criteria.json.

Reads results/cv/summary.json, results/cv/mechanisms.json, the stored
out-of-fold predictions (results/cv/runs) and the planted-event results
(results/synthetic). Every criterion carries its timing: "P" (specified
before the analysis was run) or "R" (written after the first-revision
results had been seen).

Usage:
    python scripts/evaluate_criteria.py
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)

from analyze_mechanisms import fast_auc, mat, oof, part_info   # noqa: E402

SYN = os.path.join("results", "synthetic")
OUT = os.path.join("results", "criteria.json")
RNG = 20261012
ABLATIONS = ["PACE-nogate", "PACE-frames", "PACE-noattn", "PACE-M4", "PACE-nomask", "PACE-noalign",
             "PACE-imagenorm", "PACE-valloss"]
GAIT = ["Gait-LR", "Gait-SVM", "Gait-RF", "Gait-XGB"]
NETS = ["BiGRU", "TCN", "ST-GCN", "CTR-GCN", "ST-Transformer"]
REMEDIES = ["anneal-M2", "gumbel-M2", "perturbed-M2"]


def excludes_zero(ci, positive=True):
    return ci[0] > 0 if positive else ci[1] < 0


def mcnemar(a, b):
    """Exact two-sided McNemar test for paired binary outcomes a, b."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    p = float(stats.binomtest(n10, n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0
    return {"a_only": n10, "b_only": n01, "p": p}


def paired_subset_bootstrap(y, P, Q, idx, B=2000, seed=RNG):
    """AUC difference (P minus Q) within a subset, stratified bootstrap of
    children, repeat-averaged."""
    rng = np.random.default_rng(seed)
    pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
    est = lambda i: np.mean([fast_auc(y[i], P[r, i]) for r in range(len(P))]) - \
        np.mean([fast_auc(y[i], Q[r, i]) for r in range(len(Q))])
    d = [est(np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])) for _ in range(B)]
    return {"diff": float(est(idx)), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def per_sequence(event_dir, arm):
    recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(event_dir, arm, "fold*.json")))]
    return {q["key"]: q for r in recs for q in r["trained"]["per_sequence"]}


def v1_v2(S):
    perf, bd, nb = S["performance"], S["paired_differences_bootstrap"], S["paired_differences_nadeau_bengio"]
    rows = {}
    for a in ABLATIONS + ["best_gait", "best_net"]:
        if a == "best_gait":
            a = max((g for g in GAIT if g in perf), key=lambda g: perf[g]["mean"]["auc"])
        elif a == "best_net":
            a = max((g for g in NETS if g in perf), key=lambda g: perf[g]["mean"]["auc"])
        k = f"PACE - {a}"
        if k not in bd:
            continue
        improves = excludes_zero(bd[k]["auc"]) and excludes_zero(nb[k]["ci95"])
        rows[a] = {"diff": perf["PACE"]["mean"]["auc"] - perf[a]["mean"]["auc"],
                   "bootstrap_ci": bd[k]["auc"], "corrected_t_ci": nb[k]["ci95"], "pace_better": improves}
    return {"timing": "R", "comparisons": rows,
            "any_component_improves": any(v["pace_better"] for k, v in rows.items() if k in ABLATIONS),
            "outperforms_simpler": all(v["pace_better"] for k, v in rows.items() if k not in ABLATIONS)}


def v3(S, M):
    p, Sj, y = part_info()
    vid = pd.read_csv("processed/video_metadata.csv").set_index("clip_id")
    crop = np.flatnonzero([not (vid.loc[s, "width"] == 1080 and vid.loc[s, "height"] == 1920) for s in Sj])
    R = S["n_repeats"]
    cov = mat(oof("Covariates"), Sj, R)
    out = {"movement_evidence_cropped": {"timing": "R"}}
    for m in ["PACE", "Gait-LR", "Gait-RF", "CTR-GCN", "Gait-LR-adjusted"]:
        if os.path.isdir(os.path.join("results", "cv", "runs", m)):
            d = paired_subset_bootstrap(y, mat(oof(m), Sj, R), cov, crop)
            d["passes"] = excludes_zero(d["ci95"])
            out["movement_evidence_cropped"][m] = d
    cf = M.get("confounds", {}).get("models", {})
    ctl = {"timing": "P"}
    for c in ("Scene", "Jitter"):
        if c in cf:
            e = cf[c]["auc_cropped"]
            ctl[c] = {"auc_cropped": e["auc"], "ci95": e["ci95"],
                      "blocks_movement_claim": e["auc"] >= 0.70 and e["ci95"][0] > 0.5}
    out["scene_jitter_controls"] = ctl
    mv = M.get("movement", {})
    out["robust_descriptors"] = {"timing": "P", "names": mv.get("robust_descriptors")}
    fm = M.get("format", {})
    if fm:
        a = fm["within_autistic_format_auc_pixel"]["mean"]
        out["representativeness"] = {"timing": "P", "auc_uncropped_vs_cropped_autistic": a,
                                     "possibly_unrepresentative": a >= 0.70}
    if "Depth-LR" in cf:
        best = max((m for m in ("Depth-LR", "Depth-RF") if m in cf), key=lambda m: cf[m]["auc_cropped"]["auc"])
        e, r = cf[best]["auc_cropped"], cf.get("Depth-recording", {}).get("auc_cropped")
        out["depth_corroboration"] = {
            "timing": "P", "model": best, "auc_cropped": e["auc"], "ci95": e["ci95"],
            "recording_auc_cropped": r["auc"] if r else None,
            "corroborates": bool(e["ci95"][0] > 0.5 and (r is None or e["auc"] > r["auc"]))}
    return out


def v4(Y):
    out = {}
    for key, d in Y.items():
        if not isinstance(d, dict) or "reference" not in d:
            continue
        event, amp = d["event"], d["amplitude"]
        ref = d["reference"]
        timing = "P" if (event != "raise" or any(a in d for a in REMEDIES + ["motion-M2"])) else "R"
        edir = os.path.join(SYN, f"amp{key}" if event == "raise" else key)
        hits = json.load(open(os.path.join(edir, "reference_hits.json")))
        best_det = ref["best"]
        res = {"event": event, "amplitude": amp, "informative": ref["informative"],
               "reference_auc": ref["auc"], "reference_top1": ref["top1"], "configs": {}}
        motion_seq = per_sequence(edir, "motion-M2") if os.path.isdir(os.path.join(edir, "motion-M2")) else {}
        for arm, e in d.items():
            if not isinstance(e, dict) or "trained" not in e:
                continue
            t = e["trained"]
            det = t["auc_mean"] >= 0.80 and min(t["auc_folds"]) >= 0.70
            any_fold = any(a >= 0.80 for a in t["auc_folds"])
            row = {"timing": "R" if arm in ("blocks-M2", "frames-K30", "blocks-M8", "nogate")
                   and event == "raise" else "P",
                   "auc_mean": t["auc_mean"], "auc_folds": t["auc_folds"], "detection": det,
                   "fold_dependent": (not det) and any_fold}
            if "top1_hit" in t:
                loc = (t["top1_hit"] - t["top1_chance"] >= 0.25 and t["top1_sign_p"] < 0.01
                       and t["coverage"] > t["coverage_chance"] and t["coverage_excess_p"] < 0.01)
                row.update({"top1": t["top1_hit"], "top1_chance": t["top1_chance"],
                            "top1_sign_p": t["top1_sign_p"], "localization": loc,
                            "passes": bool(det and loc and ref["informative"])})
                seq = per_sequence(edir, arm)
                keys = sorted(set(seq) & set(hits[best_det]))
                row["vs_best_detector"] = mcnemar([seq[k]["top1_hit"] for k in keys],
                                                  [hits[best_det][k] for k in keys])
                if motion_seq and arm != "motion-M2":
                    keys = sorted(set(seq) & set(motion_seq))
                    row["vs_motion_gate"] = mcnemar([seq[k]["top1_hit"] for k in keys],
                                                    [motion_seq[k]["top1_hit"] for k in keys])
                adds = [row.get("vs_best_detector"), row.get("vs_motion_gate")]
                row["adds_over_heuristics"] = all(
                    a is not None and a["p"] < 0.05 and a["a_only"] > a["b_only"] for a in adds)
            res["configs"][arm] = row
        # the remedies against frame selection with the same budget
        fr = res["configs"].get("frames-K30")
        if fr and "top1" in fr:
            fseq = per_sequence(edir, "frames-K30")
            for arm in REMEDIES + ["blocks-M2"]:
                r = res["configs"].get(arm)
                if not r:
                    continue
                seq = per_sequence(edir, arm)
                keys = sorted(set(seq) & set(fseq))
                mc = mcnemar([seq[k]["top1_hit"] for k in keys], [fseq[k]["top1_hit"] for k in keys])
                r["vs_frames"] = mc
                r["beats_frames"] = bool(r["auc_mean"] >= fr["auc_mean"] and min(r["auc_folds"]) >= 0.70
                                         and mc["p"] < 0.05 and mc["a_only"] > mc["b_only"])
        out[key] = res
    # remedy selection rule (B1)
    r1 = out.get("1.0", {}).get("configs", {})
    cand = [a for a in REMEDIES if a in r1]
    if cand:
        best = max(r1[a]["auc_mean"] for a in cand)
        tied = [a for a in cand if best - r1[a]["auc_mean"] <= 0.01]
        tied.sort(key=lambda a: (-min(r1[a]["auc_folds"]), REMEDIES.index(a)))
        out["chosen_remedy"] = tied[0]
    return out


def v5(M):
    se, sn = M.get("selection", {}), M.get("selection_nulls", {})
    out = {}
    for arm, d in se.items():
        row = {"chance_criterion": {"timing": "R", "jaccard": d["jaccard"]["mean"] if d["jaccard"] else None,
                                    "chance": d["chance"]["mean"] if d["chance"] else None,
                                    "passes": bool(d["wilcoxon_p_excess"] is not None and d["wilcoxon_p_excess"] < 0.01
                                                   and d["jaccard"]["mean"] > d["chance"]["mean"])},
               "mostly_trivial": {"timing": "R", "n_trivial": d["n_trivial"], "n": d["n_children"],
                                  "flag": d["n_trivial"] > d["n_children"] / 2}}
        if arm in sn:
            n = sn[arm]
            ex = n["trained_minus_random"]["mean"]
            row["learned"] = {"timing": "P", "trained": n["trained"]["mean"], "random_init": n["random_init"]["mean"],
                              "excess": ex, "p": n["wilcoxon_p_trained_vs_random"],
                              "passes": bool(ex >= 0.05 and n["wilcoxon_p_trained_vs_random"] < 0.01)}
            row["motion_equivalent"] = {"timing": "P", "trained_vs_motion": n["trained_vs_motion"]["mean"],
                                        "trained": n["trained"]["mean"],
                                        "flag": n["trained_vs_motion"]["mean"] >= n["trained"]["mean"] - 0.02}
        out[arm] = row
    return out


def v6(M, Y):
    out = {"timing": "R", "planted": {}}
    for key, d in Y.items():
        if not isinstance(d, dict) or "reference" not in d:
            continue
        e = d.get("blocks-M2")
        if not e:
            continue
        t, r = e["trained"], e["random_weights"]
        out["planted"][key] = {
            "trained_ratio": t["attr_event_share"] / t["attr_event_chance"],
            "random_ratio": r["attr_event_share"] / r["attr_event_chance"],
            "passes": bool(t["attr_event_share"] >= 1.5 * t["attr_event_chance"]
                           and r["attr_event_share"] < 1.2 * r["attr_event_chance"])}
    at = M.get("attribution")
    if at:
        diffs = [abs(at["stream_share"][k]["mean"] - at["random_init_stream_share"][k]["mean"])
                 for k in at["stream_share"]]
        diffs += [abs(at["region_share_velocity_acceleration"][k]["mean"]
                      - at["random_init_region_share_velocity_acceleration"][k]["mean"])
                  for k in at["region_share_velocity_acceleration"]]
        rho = at["trained_vs_random_landmark_profile_spearman"]["mean"]
        out["real"] = {"max_share_difference": max(diffs), "profile_rho": rho,
                       "passes": bool(max(diffs) > 0.05 and rho < 0.5)}
    return out


def main():
    S = json.load(open("results/cv/summary.json"))
    M = json.load(open("results/cv/mechanisms.json"))
    Y = json.load(open(os.path.join(SYN, "summary.json")))
    res = {"V1_V2": v1_v2(S), "V3": v3(S, M), "V4": v4(Y), "V5": v5(M), "V6": v6(M, Y)}
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps({k: (v if k != "V4" else {kk: {a: {x: y for x, y in c.items() if x in
                                                        ("auc_mean", "passes", "beats_frames", "fold_dependent")}
                                                    for a, c in vv["configs"].items()}
                                               if isinstance(vv, dict) and "configs" in vv else vv
                                               for kk, vv in v.items()})
                      for k, v in res.items()}, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
