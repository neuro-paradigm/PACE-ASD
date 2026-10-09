"""
PACE — fourth-phase analyses of the real data (docs/validation_criteria_phase4.md).

  incremental   E2: incremental value of the movement scores (test-case
                network, Gait-RF, Depth-RF) beyond the scene, time-of-day,
                frame-format and demographic scores, by nested stacking with
                a Firth second level; all children and the cropped videos
                -> results/phase4/incremental_nested.json

Usage:
    python scripts/phase4_real.py --sections incremental
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)

from analyze_mechanisms import oof                                          # noqa: E402
from phase4_common import (OUT4, STACK_CONTROLS, inner_from_checkpoints,    # noqa: E402
                           inner_from_stored, nested_incremental)

MOVEMENT = ["PACE", "Gait-RF", "Depth-RF"]


def incremental():
    from feature_sets import FeatureSets
    part = json.load(open("splits/cv_partition.json"))
    R = 3
    S = part["subjects"]
    vid = pd.read_csv("processed/video_metadata.csv").set_index("clip_id")
    cropped = {s for s in S if not (vid.loc[s, "width"] == 1080 and vid.loc[s, "height"] == 1920)}
    F = FeatureSets(S)
    outer = {m: oof(m) for m in STACK_CONTROLS + MOVEMENT}
    inner = {}
    for m in STACK_CONTROLS + ["Gait-RF", "Depth-RF"]:
        t0 = time.time()
        inner[m] = inner_from_stored(m, F, part, R)
        print(f"[inner] {m} ({time.time() - t0:.0f}s)", flush=True)
    inner["PACE"] = inner_from_checkpoints("PACE", part, R)
    print("[inner] PACE", flush=True)
    out = {"controls": STACK_CONTROLS, "repetitions": R, "level2": "Firth", "B": 2000}
    for name, sub in (("all", None), ("cropped", cropped)):
        out[name] = {}
        for m in MOVEMENT:
            r = nested_incremental(part, outer, inner, STACK_CONTROLS, m, subset=sub, R=R)
            out[name][m] = r
            print(f"[incremental] {name} {m}: AUC {r['auc_controls']:.3f} -> {r['auc_with_movement']:.3f} "
                  f"{r['auc_gain_ci95']}; log loss {r['logloss_controls']:.3f} -> {r['logloss_with_movement']:.3f} "
                  f"{r['logloss_reduction_ci95']}; adds={r['adds']}", flush=True)
    # inner-fold AUC of each level-1 model, to show the inner scores are on the scale of the outer ones
    from phase4_common import fast_auc
    out["inner_auc"] = {m: float(np.mean([fast_auc([part["labels"][s] for s in d], list(d.values()))
                                          for r in inner[m] for d in inner[m][r].values()])) for m in inner}
    return out


SECTIONS = {"incremental": ("incremental_nested", incremental)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", nargs="*", default=list(SECTIONS))
    a = ap.parse_args()
    os.makedirs(OUT4, exist_ok=True)
    for s in a.sections:
        fname, fn = SECTIONS[s]
        t0 = time.time()
        json.dump(fn(), open(os.path.join(OUT4, f"{fname}.json"), "w"), indent=1)
        print(f"{s} done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
