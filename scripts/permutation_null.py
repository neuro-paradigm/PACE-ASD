"""
PACE — label-permutation null of the whole pipeline (docs/validation_criteria_phase3.md, D2).

For permutation i the labels of the 95 children are permuted, a new
stratified nested partition is made for the permuted labels, and the models
are trained and evaluated exactly as for the real labels.

  --network    the test-case network, permutations 0-9 (GPU; resumable;
               --workers/--worker_id split the (permutation, fold) jobs)
  --fast       Gait-LR, Scene, Frame-format, Mean-pose, All-controls
               (permutations 0-199) and Gait-RF (0-99)
  --summarize  null distributions and permutation p-values ->
               results/phase3/permutation_null.json

Usage:
    python scripts/permutation_null.py --network --workers 3 --worker_id 0
    python scripts/permutation_null.py --fast
    python scripts/permutation_null.py --summarize
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

from phase3_common import OUT, permuted_partition, run_network_folds   # noqa: E402

NET_PERMS = range(10)
FAST_PERMS = {"Gait-LR": 200, "Scene": 200, "Frame-format": 200, "Mean-pose": 200,
              "All-controls": 200, "Gait-RF": 100}


def cohort():
    p = json.load(open("splits/cv_partition.json"))
    return p["subjects"], [p["labels"][s] for s in p["subjects"]]


def network(workers, worker_id):
    import yaml
    from run_cv import load_arrays
    S, y = cohort()
    cfg = yaml.safe_load(open("configs/config.yaml"))
    arrays = load_arrays(cfg, S, True)
    jobs = [(i, k) for i in NET_PERMS for k in range(5)][worker_id::workers]
    for i, k in jobs:
        part = permuted_partition(i, S, y)
        run_network_folds(part, arrays, os.path.join(OUT, "permutation", "network", f"p{i:03d}"), [k])


def fast(only=None, start=0, stop=None):
    """Permutations start..stop-1 of each model (default: all); a partial
    range is written to <model>_<start>_<stop>.json so that ranges can run in
    parallel processes; summarize() merges the files."""
    from feature_sets import FeatureSets, fit_predict
    S, y = cohort()
    F = FeatureSets(S)
    out_dir = os.path.join(OUT, "permutation", "fast")
    os.makedirs(out_dir, exist_ok=True)
    for name, n in FAST_PERMS.items():
        if only and name not in only:
            continue
        hi = min(n, stop if stop is not None else n)
        part = start > 0 or hi < n
        path = os.path.join(out_dir, f"{name}_{start}_{hi}.json" if part else f"{name}.json")
        done = json.load(open(path)) if os.path.isfile(path) else {}
        for i in range(start, hi):
            if str(i) in done:
                continue
            part_i = permuted_partition(i, S, y)
            probs = fit_predict(name, F, part_i, r=0)
            done[str(i)] = {"labels": part_i["labels"], "probs": probs}
            if i % 5 == 4:
                json.dump(done, open(path, "w"))
                print(f"[{name}] {i + 1}/{hi}", flush=True)
        json.dump(done, open(path, "w"))


def summarize():
    from analyze_mechanisms import fast_auc, mat, oof
    import pandas as pd
    S, y = cohort()
    y = np.array(y)
    vid = pd.read_csv("processed/video_metadata.csv").set_index("clip_id")
    crop = np.array([not (vid.loc[s, "width"] == 1080 and vid.loc[s, "height"] == 1920) for s in S])
    res = {}

    def auc_of(labels, probs, subset=None):
        idx = [s for s in S if subset is None or crop[S.index(s)]]
        return fast_auc(np.array([labels[s] for s in idx]), np.array([probs[s] for s in idx]))

    def observed(name, r_only=None):
        P = mat(oof(name), S, 3)
        rows = range(3) if r_only is None else [r_only]
        return (float(np.mean([fast_auc(y, P[r]) for r in rows])),
                float(np.mean([fast_auc(y[crop], P[r, crop]) for r in rows])))

    fdir = os.path.join(OUT, "permutation", "fast")
    merged = {}
    for f in sorted(glob.glob(os.path.join(fdir, "*.json"))):
        name = os.path.basename(f)[:-5].split("_")[0]
        merged.setdefault(name, {}).update(json.load(open(f)))
    for name, d in sorted(merged.items()):
        null_all = np.array([auc_of(v["labels"], v["probs"]) for v in d.values()])
        null_crop = np.array([auc_of(v["labels"], v["probs"], True) for v in d.values()])
        obs = observed(name) if os.path.isdir(os.path.join("results", "cv", "runs", name)) else (None, None)
        res[name] = _null_summary(null_all, null_crop, obs)
    nets = []
    for d in sorted(glob.glob(os.path.join(OUT, "permutation", "network", "p*"))):
        recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "r0_k*.json")))]
        if len(recs) < 5:
            continue
        labels, probs = {}, {}
        for r in recs:
            labels.update(zip(r["test_subject_ids"], r["test_labels"]))
            probs.update(zip(r["test_subject_ids"], r["probs"]))
        nets.append((auc_of(labels, probs), auc_of(labels, probs, True)))
    if nets:
        a = np.array(nets)
        res["PACE"] = _null_summary(a[:, 0], a[:, 1], observed("PACE"))
    json.dump(res, open(os.path.join(OUT, "permutation_null.json"), "w"), indent=1)
    for k, v in res.items():
        print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()
                  if not isinstance(vv, list)})


def _null_summary(null_all, null_crop, obs):
    o_all, o_crop = obs
    out = {"n_permutations": int(len(null_all)),
           "null_mean_all": float(null_all.mean()), "null_sd_all": float(null_all.std(ddof=1)),
           "null_p95_all": float(np.percentile(null_all, 95)), "null_p99_all": float(np.percentile(null_all, 99)),
           "null_max_all": float(null_all.max()),
           "null_mean_cropped": float(null_crop.mean()), "null_p99_cropped": float(np.percentile(null_crop, 99)),
           "null_max_cropped": float(null_crop.max()),
           "null_all": null_all.tolist(), "null_cropped": null_crop.tolist()}
    if o_all is not None:
        out.update({"observed_all": o_all, "observed_cropped": o_crop,
                    "p_all": float((1 + (null_all >= o_all).sum()) / (1 + len(null_all))),
                    "p_cropped": float((1 + (null_crop >= o_crop).sum()) / (1 + len(null_crop)))})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", action="store_true")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--worker_id", type=int, default=0)
    a = ap.parse_args()
    if a.network:
        network(a.workers, a.worker_id)
    if a.fast:
        fast(a.models, a.start, a.stop)
    if a.summarize:
        summarize()
