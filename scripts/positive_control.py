"""
PACE — semi-synthetic positive control (docs/validation_criteria_phase3.md, D1).

The 45 typically developing children are split at random into two
pseudo-groups (22 and 23 children; 20 splits). In pseudo-group 1 the
low-frequency movement of the wrists and hands relative to the shoulders is
amplified by a gain g (elbows by (1 + g) / 2) in the video landmarks and in
the depth-camera skeletons; tracking noise, frame format, image properties,
time of day and demographics stay as they are. The protocol should then find
the difference, keep every control at chance and conclude that the
discrimination is attributable to movement; with g = 1 it should find
nothing.

  --gain-rule   reference AUC of the arm-swing descriptor alone for each g of
                the grid (no model trained); chooses g
  --fast        feature classifiers and controls, every split, chosen g and g = 1
  --network     the test-case network (splits 0-4 at the chosen g, 0-1 at g = 1;
                --workers/--worker_id split the jobs)
  --summarize   the protocol's criteria per split -> results/phase3/positive_control.json

Usage:
    python scripts/positive_control.py --gain-rule
    python scripts/positive_control.py --fast
    python scripts/positive_control.py --network --workers 3 --worker_id 0
    python scripts/positive_control.py --summarize
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)

from phase3_common import (OUT, fast_auc, incremental_value, paired_auc_bootstrap,  # noqa: E402
                           plant_arm_swing, pseudo_labels, pseudo_partition, run_network_folds)
from sequence import prepare_sequence                        # noqa: E402

GRID = [1.1, 1.2, 1.3, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]   # 2.5-6 added by amendment 1
ORIGINAL_GRID_CHOICE = 2.0
TARGET = 0.80
N_SPLITS = 20
NET_SPLITS = {"chosen": range(5), 1.0: range(2)}
PC = os.path.join(OUT, "positive_control")
MOVEMENT = ["Gait-LR", "Gait-RF", "Mean-pose"]
CONTROLS = ["Frame-format", "Scene", "Time-of-day", "Demographics", "Tracking-quality", "All-controls"]
DEPTH = ["Depth-LR", "Depth-RF", "Depth-recording"]
STACK_CONTROLS = ["Scene", "Time-of-day", "Frame-format", "Demographics"]


def td_children():
    p = json.load(open("splits/cv_partition.json"))
    return sorted(s for s in p["subjects"] if p["labels"][s] == 0)


def base_arrays(td):
    from feature_sets import stored
    return {s: prepare_sequence(stored(s)) for s in td}


def planted(base, td, labels, gain):
    return {s: (plant_arm_swing(base[s], gain) if l == 1 else base[s]) for s, l in zip(td, labels)}


def gain_rule():
    from gait_features import FEATURE_NAMES, feature_matrix
    td = td_children()
    base = base_arrays(td)
    j = FEATURE_NAMES.index("arm_swing")
    x0 = feature_matrix([base[s] for s in td])[:, j]
    res = {}
    for g in GRID:
        # planting depends only on the child and g, so each child's planted
        # descriptor is computed once and taken for the children in pseudo-group 1
        xg = feature_matrix([plant_arm_swing(base[s], g) for s in td])[:, j]
        aucs = []
        for sp in range(N_SPLITS):
            _, lab = pseudo_labels(sp, td)
            x = np.where(lab == 1, xg, x0)
            aucs.append(fast_auc(lab, x))
        res[str(g)] = {"mean_auc": float(np.mean(aucs)), "aucs": aucs}
        print(f"[gain rule] g={g}: mean arm-swing AUC {np.mean(aucs):.3f}", flush=True)
    chosen = min(GRID, key=lambda g: (abs(res[str(g)]["mean_auc"] - TARGET), g))
    out = {"grid": res, "target": TARGET, "chosen": chosen}
    os.makedirs(PC, exist_ok=True)
    json.dump(out, open(os.path.join(PC, "gain_rule.json"), "w"), indent=1)
    return chosen


def chosen_gain():
    return json.load(open(os.path.join(PC, "gain_rule.json")))["chosen"]


def fast(gains=None, splits=None):
    from feature_sets import FeatureSets, fit_predict
    td = td_children()
    base = base_arrays(td)
    for g in gains or sorted({chosen_gain(), ORIGINAL_GRID_CHOICE, 1.0}, reverse=True):
        for sp in (splits if splits is not None else range(N_SPLITS)):
            path = os.path.join(PC, f"g{g}", f"split{sp:02d}.json")
            if os.path.isfile(path):
                continue
            part = pseudo_partition(sp, td)
            lab = [part["labels"][s] for s in td]
            arr = planted(base, td, lab, g)
            F = FeatureSets(td, prep=arr, depth_gain=g, depth_planted=[s for s, l in zip(td, lab) if l == 1])
            probs = {m: fit_predict(m, F, part) for m in MOVEMENT + CONTROLS + DEPTH}
            os.makedirs(os.path.dirname(path), exist_ok=True)
            json.dump({"gain": g, "split": sp, "labels": part["labels"], "probs": probs}, open(path, "w"))
            print(f"[positive control] g={g} split {sp}: " +
                  " ".join(f"{m}={fast_auc(lab, [probs[m][s] for s in td]):.2f}" for m in probs), flush=True)


def network(workers, worker_id):
    td = td_children()
    base = base_arrays(td)
    g_star = chosen_gain()
    jobs = [(g, sp, k) for key, sps in NET_SPLITS.items() for sp in sps for k in range(5)
            for g in [g_star if key == "chosen" else key]]
    for g, sp, k in jobs[worker_id::workers]:
        part = pseudo_partition(sp, td)
        lab = [part["labels"][s] for s in td]
        run_network_folds(part, planted(base, td, lab, g), os.path.join(PC, f"g{g}", "network", f"split{sp:02d}"), [k])


def _robust_arm_swing(arr, td, lab):
    """Criterion (iv): arm swing a robust descriptor difference (adjusted for
    age and sex, q < 0.05 over the 26 descriptors; same sign with interval
    excluding 0 for smoothed trajectories and the first 90 frames)."""
    import pandas as pd
    from analyze_mechanisms import _bh, _ols_group_effect
    from gait_features import FEATURE_NAMES, feature_matrix
    from protocol import crop_valid
    from recording_features import smooth_sequence
    from sequence import mask_summary
    meta = pd.read_csv("processed/participant_metadata.csv").set_index("subject_id")
    age = np.array([meta.loc[s, "age_years"] for s in td], float)
    male = np.array([meta.loc[s, "sex"] == "M" for s in td])
    y = np.array(lab)
    X = feature_matrix([arr[s] for s in td])
    eff = [_ols_group_effect(X[:, j], y, age, male, B=500) for j in range(X.shape[1])]
    q = _bh([e["p"] for e in eff])
    j = FEATURE_NAMES.index("arm_swing")
    span = np.array([mask_summary(arr[s])["span"] for s in td])
    sm = feature_matrix([smooth_sequence(arr[s]) for s in td])[:, j]
    f9 = feature_matrix([crop_valid(arr[s], 90) for s in td])[:, j]
    ok9 = span >= 90
    e_sm = _ols_group_effect(sm, y, age, male, B=500)
    e_f9 = _ols_group_effect(f9[ok9], y[ok9], age[ok9], male[ok9], B=500)
    st = eff[j]
    same = lambda e: np.sign(e["beta"]) == np.sign(st["beta"]) and (e["ci95"][0] > 0 or e["ci95"][1] < 0)
    return {"beta": st["beta"], "q": float(q[j]), "smoothed": e_sm, "first90": e_f9,
            "robust": bool(q[j] < 0.05 and same(e_sm) and same(e_f9)),
            "robust_names": [FEATURE_NAMES[i] for i in range(len(eff))
                             if q[i] < 0.05 and i == j]}


def evaluate_split(rec, td, base, net_probs=None):
    lab = np.array([rec["labels"][s] for s in td])
    P = {m: np.array([v[s] for s in td]) for m, v in rec["probs"].items()}
    if net_probs is not None:
        P["Network"] = np.array([net_probs[s] for s in td])
    auc = {m: fast_auc(lab, p) for m, p in P.items()}
    out = {"auc": auc}
    out["i_controls_below_0.70"] = bool(all(auc[c] < 0.70 for c in CONTROLS))
    movers = MOVEMENT + (["Network"] if "Network" in P else [])
    best = max(movers, key=lambda m: auc[m])
    pb = paired_auc_bootstrap(lab, P[best], P["All-controls"])
    out["best_movement_model"] = best
    out["best_vs_all_controls"] = pb
    out["ii_movement_above_chance_and_controls"] = bool(pb["ci_a"][0] > 0.5 and pb["diff_ci95"][0] > 0)
    logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    C = np.c_[[logit(P[c]) for c in STACK_CONTROLS]].T
    iv = incremental_value(lab, C, logit(P[best]), B=500)
    out["incremental_value"] = iv
    out["iii_incremental_value"] = iv["adds"]
    arr = planted(base, td, lab.tolist(), rec["gain"])
    rb = _robust_arm_swing(arr, td, lab.tolist())
    out["arm_swing"] = rb
    out["iv_robust_descriptor"] = rb["robust"]
    dbest = max(["Depth-LR", "Depth-RF"], key=lambda m: auc[m])
    dpb = paired_auc_bootstrap(lab, P[dbest], P["Depth-recording"])
    out["depth"] = {"model": dbest, **dpb}
    out["v_depth_corroborates"] = bool(dpb["ci_a"][0] > 0.5 and dpb["auc_a"] > dpb["auc_b"])
    out["attributable_to_movement"] = bool(out["i_controls_below_0.70"] and out["ii_movement_above_chance_and_controls"]
                                           and out["iii_incremental_value"] and out["iv_robust_descriptor"]
                                           and out["v_depth_corroborates"])
    return out


def summarize():
    td = td_children()
    base = base_arrays(td)
    res = {"gain_rule": json.load(open(os.path.join(PC, "gain_rule.json"))), "by_gain": {}}
    for g in sorted({chosen_gain(), ORIGINAL_GRID_CHOICE, 1.0}, reverse=True):
        rows = []
        for f in sorted(glob.glob(os.path.join(PC, f"g{g}", "split*.json"))):
            rec = json.load(open(f))
            sp = rec["split"]
            nd = os.path.join(PC, f"g{g}", "network", f"split{sp:02d}")
            net = None
            fs = sorted(glob.glob(os.path.join(nd, "r0_k*.json")))
            if len(fs) == 5:
                net = {}
                for q in fs:
                    r = json.load(open(q))
                    net.update(zip(r["test_subject_ids"], r["probs"]))
            row = {"split": sp, "with_network": net is not None, **evaluate_split(rec, td, base, net)}
            rows.append(row)
            print(f"[summary] g={g} split {sp}: movement={row['attributable_to_movement']} "
                  f"best={row['best_movement_model']} {row['auc'][row['best_movement_model']]:.2f} "
                  f"controls<0.70={row['i_controls_below_0.70']}", flush=True)
        keys = ["i_controls_below_0.70", "ii_movement_above_chance_and_controls", "iii_incremental_value",
                "iv_robust_descriptor", "v_depth_corroborates", "attributable_to_movement"]
        share = {k: float(np.mean([r[k] for r in rows])) for k in keys}
        models = sorted(rows[0]["auc"]) if rows else []
        auc_summary = {m: {"mean": float(np.mean([r["auc"][m] for r in rows if m in r["auc"]])),
                           "min": float(np.min([r["auc"][m] for r in rows if m in r["auc"]])),
                           "max": float(np.max([r["auc"][m] for r in rows if m in r["auc"]])),
                           "n": int(sum(m in r["auc"] for r in rows))}
                       for m in set(models) | {"Network"} if any(m in r["auc"] for r in rows)}
        controls_max = [max(r["auc"][c] for c in CONTROLS) for r in rows]
        res["by_gain"][str(g)] = {"n_splits": len(rows), "share_met": share, "auc": auc_summary,
                                  "largest_control_auc": {"mean": float(np.mean(controls_max)),
                                                          "max": float(np.max(controls_max))},
                                  "splits": rows}
    g_star = str(chosen_gain())
    if g_star in res["by_gain"] and "1.0" in res["by_gain"]:
        res["passes"] = bool(res["by_gain"][g_star]["share_met"]["attributable_to_movement"] >= 0.80
                             and res["by_gain"]["1.0"]["share_met"]["attributable_to_movement"] <= 0.10)
    json.dump(res, open(os.path.join(OUT, "positive_control.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gain-rule", action="store_true")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--network", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--gains", type=float, nargs="*", default=None)
    ap.add_argument("--splits", type=int, nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--worker_id", type=int, default=0)
    a = ap.parse_args()
    if a.gain_rule:
        gain_rule()
    if a.fast:
        fast(a.gains, a.splits)
    if a.network:
        network(a.workers, a.worker_id)
    if a.summarize:
        summarize()
