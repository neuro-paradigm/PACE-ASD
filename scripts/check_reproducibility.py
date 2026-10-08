"""Rerun one outer fold of a cross-validated arm and compare its out-of-fold
predictions with the stored ones.

    python scripts/check_reproducibility.py --arm PACE --repeat 0 --fold 0

The rerun is written to results/repro/runs/<arm>/ (never over the stored run);
the comparison goes to results/cv/reproducibility.json.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="PACE")
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--stored", default="results/cv/runs")
    ap.add_argument("--rerun_dir", default="results/repro")
    ap.add_argument("--out", default="results/cv/reproducibility.json")
    args = ap.parse_args()
    os.chdir(ROOT)

    name = f"r{args.repeat}_k{args.fold}.json"
    stored_path = os.path.join(args.stored, args.arm, name)
    with open(stored_path) as f:
        stored = json.load(f)
    with open(os.path.join(args.stored, args.arm, "config.json")) as f:
        overrides = json.load(f).get("overrides", {})

    rerun_path = os.path.join(args.rerun_dir, "runs", args.arm, name)
    if os.path.isfile(rerun_path):
        os.remove(rerun_path)
    # run_cv assigns job (r, k) to worker (r * n_outer + k) % workers; one worker
    # per job means only the requested fold runs.
    with open("configs/config.yaml") as f:
        import yaml
        n_outer = yaml.safe_load(f)["cv"]["n_outer"]
    n_jobs = (args.repeat + 1) * n_outer
    job = args.repeat * n_outer + args.fold
    cmd = [sys.executable, "scripts/run_cv.py", "--arm", args.arm,
           "--repeats", str(args.repeat + 1), "--workers", str(n_jobs), "--worker_id", str(job),
           "--set", f"output.cv_dir={args.rerun_dir}",
           *[f"{k}={v}" for k, v in overrides.items()]]
    t0 = time.time()
    subprocess.run(cmd, check=True)
    minutes = (time.time() - t0) / 60
    with open(rerun_path) as f:
        rerun = json.load(f)

    assert stored["test_subject_ids"] == rerun["test_subject_ids"]
    a, b = np.asarray(stored["probs"]), np.asarray(rerun["probs"])
    la = np.asarray([m["test"]["original"]["logits"] for m in stored["inner"]])
    lb = np.asarray([m["test"]["original"]["logits"] for m in rerun["inner"]])
    res = {
        "arm": args.arm, "repeat": args.repeat, "fold": args.fold,
        "n_children": int(len(a)),
        "identical": bool(np.array_equal(a, b)),
        "max_abs_prob_diff": float(np.max(np.abs(a - b))),
        "max_abs_logit_diff": float(np.max(np.abs(la - lb))),
        "best_epochs_stored": [i["best_epoch"] for i in stored["inner"]],
        "best_epochs_rerun": [i["best_epoch"] for i in rerun["inner"]],
        "minutes": minutes,
    }
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
