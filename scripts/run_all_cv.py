"""
PACE-ASD — every cross-validation run reported in the article, as one queue.

Stages (run in order; each stage can be resumed):
  pilot     learning-rate pilot for the comparison architectures: repetition 0,
            first inner fold only, lr in {5e-5, 3e-4, 1e-3}; the rate with the
            highest mean inner-validation AUC is used for that architecture.
            PACE-ASD keeps the reference learning rate (5e-5).
  pace      PACE-ASD, its ablation arms and the validity-only control,
            5 x 5-fold nested cross-validation.
  comparison  the comparison architectures with their pilot learning rates,
            same partition.

Usage:
    python scripts/run_all_cv.py --stage pilot --jobs 4
    python scripts/run_all_cv.py --stage pace --jobs 4
    python scripts/run_all_cv.py --stage comparison --jobs 4
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
from comparison_models import COMPARISON_MODELS          # noqa: E402

PY = sys.executable
PILOT_LRS = ["5e-05", "0.0003", "0.001"]
PACE_ARMS = ["PACE", "PACE-nogate", "PACE-frames", "PACE-noattn", "PACE-M4",
             "PACE-nomask", "PACE-noalign", "PACE-valloss", "PACE-maskonly", "PACE-imagenorm"]
CKPT_ARMS = {"PACE", "PACE-nogate", "PACE-frames", "PACE-M4", "PACE-noalign"}


def pilot_jobs():
    jobs = []
    for name in COMPARISON_MODELS:
        for lr in PILOT_LRS:
            for k in range(5):
                jobs.append([PY, "scripts/run_cv.py", "--arm", name, "--tag", f"pilot/{name}@{lr}",
                             "--set", f"training.lr={lr}", "--repeats", "1", "--inner_folds", "0",
                             "--workers", "5", "--worker_id", str(k)])
    return jobs


def pilot_choice() -> dict:
    """Learning rate per architecture with the highest mean inner-validation AUC."""
    out = {}
    for name in COMPARISON_MODELS:
        scores = {}
        for lr in PILOT_LRS:
            recs = [json.load(open(p)) for p in
                    glob.glob(f"results/cv/runs/pilot/{name}@{lr}/r0_k*.json")]
            if len(recs) == 5:
                scores[lr] = float(np.mean([r["inner"][0]["val_auc"] for r in recs]))
        if len(scores) == len(PILOT_LRS):
            out[name] = {"lr": max(scores, key=scores.get), "val_auc": scores}
    return out


def main_jobs(workers_per_arm: int, stage: str):
    choice = {}
    if stage == "comparison":
        choice = pilot_choice()
        missing = [n for n in COMPARISON_MODELS if n not in choice]
        if missing:
            raise SystemExit(f"pilot incomplete for {missing}")
        with open("results/cv/pilot_learning_rates.json", "w") as f:
            json.dump(choice, f, indent=1)
    jobs = []
    for arm in (PACE_ARMS if stage == "pace" else list(COMPARISON_MODELS)):
        extra = []
        if arm in COMPARISON_MODELS:
            extra = ["--set", f"training.lr={choice[arm]['lr']}"]
        if arm in CKPT_ARMS:
            extra.append("--save_checkpoints")
        for w in range(workers_per_arm):
            jobs.append([PY, "scripts/run_cv.py", "--arm", arm, *extra,
                         "--workers", str(workers_per_arm), "--worker_id", str(w)])
    return jobs


def run_queue(jobs, n_parallel, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    running, i = [], 0
    while i < len(jobs) or running:
        while i < len(jobs) and len(running) < n_parallel:
            log = open(os.path.join(log_dir, f"job{i:03d}.log"), "w")
            running.append((subprocess.Popen(jobs[i], stdout=log, stderr=subprocess.STDOUT), log, i))
            print(f"start job {i}: {' '.join(jobs[i][2:])}", flush=True)
            i += 1
        time.sleep(5)
        for p, log, j in list(running):
            if p.poll() is not None:
                log.close()
                running.remove((p, log, j))
                print(f"done job {j} (exit {p.returncode})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pilot", "pace", "comparison"], required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--workers_per_arm", type=int, default=5)
    a = ap.parse_args()
    jobs = pilot_jobs() if a.stage == "pilot" else main_jobs(a.workers_per_arm, a.stage)
    run_queue(jobs, a.jobs, f"results/cv/logs/{a.stage}")
