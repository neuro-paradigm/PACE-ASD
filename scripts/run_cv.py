"""
PACE-ASD — repeated nested subject-level cross-validation.

Outer loop: R repetitions of stratified K-fold cross-validation over all
children of the cohort (one clip per child). Inner loop: the outer-training
children are split into I stratified folds; one model is trained per inner
fold (epoch selection and logistic recalibration on that fold's validation
children), and each outer-test child receives the mean of the I recalibrated
probabilities. No outer-test child is used for any choice made in training.

The partition is frozen in a JSON file the first time it is needed and read
from disk afterwards. Each (arm, repetition, outer fold) is written to its
own JSON file, so runs can be interrupted, resumed and split across
processes (--workers / --worker_id).

Usage:
    python scripts/run_cv.py --arm PACE
    python scripts/run_cv.py --arm PACE --workers 4 --worker_id 0   # one of four processes
    python scripts/run_cv.py --arm BiGRU --set training.lr=0.001 --tag BiGRU@1e-3
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

from protocol import (ARMS, TEST_VARIANTS, arm_config, fit_model,  # noqa: E402
                      predict_logits)
from sequence import prepare_sequence                             # noqa: E402


# ── cohort and partition ──────────────────────────────────────────────────────

def cohort(config: dict) -> pd.DataFrame:
    """One row per child: the deduplicated autistic and typically developing
    children (the children with severe autism are excluded)."""
    df = pd.read_csv(os.path.join(config["data"]["processed_dir"], "labels.csv"))
    df = df[~df["subject_id"].str.startswith("severe")]
    assert df["subject_id"].is_unique
    return df.sort_values("subject_id").reset_index(drop=True)


def make_partition(config: dict) -> dict:
    cv = config["cv"]
    path = cv["partition_file"]
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    df = cohort(config)
    subj = df["subject_id"].to_numpy()
    y = df["label"].to_numpy()
    reps = []
    for r in range(cv["max_repeats"]):
        outer = StratifiedKFold(cv["n_outer"], shuffle=True, random_state=cv["seed"] + r)
        folds = []
        for k, (tr, te) in enumerate(outer.split(subj, y)):
            inner = StratifiedKFold(cv["n_inner"], shuffle=True,
                                    random_state=cv["seed"] + 1000 * (r + 1) + k)
            inn = [{"train": sorted(subj[tr][a].tolist()), "val": sorted(subj[tr][b].tolist())}
                   for a, b in inner.split(subj[tr], y[tr])]
            folds.append({"test": sorted(subj[te].tolist()), "inner": inn})
        reps.append(folds)
    part = {"seed": cv["seed"], "n_outer": cv["n_outer"], "n_inner": cv["n_inner"],
            "subjects": subj.tolist(), "labels": {s: int(l) for s, l in zip(subj, y)},
            "repeats": reps}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(part, f, indent=1)
    return part


def stored_array(config: dict, subject: str) -> np.ndarray:
    """The (300, 33, 2) array of one recording. With data.coordinates = "image"
    the array is rebuilt from the raw keypoints without conversion to pixel
    units (each axis normalised by its own frame dimension), which is used
    only to measure the effect of that normalisation."""
    pdir = config["data"]["processed_dir"]
    if config["data"].get("coordinates", "pixel") == "image":
        from preprocess import normalise, pad_or_truncate
        raw = np.load(os.path.join(pdir, "keypoints", f"{subject}.npy"))
        return pad_or_truncate(normalise(raw))
    return np.load(os.path.join(pdir, "features", f"{subject}.npy")).astype(np.float32)


def load_arrays(config: dict, subjects, align: bool) -> dict:
    return {s: prepare_sequence(stored_array(config, s),
                                config["data"].get("max_abs_coord", 10.0), align)
            for s in subjects}


def model_seed(r: int, k: int, i: int) -> int:
    """Identical across arms, so arms are compared on identical draws."""
    return 10_000 + 100 * r + 10 * k + i


# ── one outer fold ────────────────────────────────────────────────────────────

def run_outer_fold(arm, cfg, part, arrays, r, k, device, inner_folds, ckpt_dir):
    fold = part["repeats"][r][k]
    lab = part["labels"]
    test = fold["test"]
    mask_only = cfg.get("data", {}).get("input") == "mask_only"
    rec = {"arm": arm, "repeat": r, "outer_fold": k, "test_subject_ids": test,
           "test_labels": [lab[s] for s in test], "inner": []}
    for i in inner_folds:
        inn = fold["inner"][i]
        seed = model_seed(r, k, i)
        model, scaler, info = fit_model(
            arm, cfg,
            [arrays[s] for s in inn["train"]], [lab[s] for s in inn["train"]],
            [arrays[s] for s in inn["val"]], [lab[s] for s in inn["val"]],
            seed, device)
        preds = {}
        for name, fn in TEST_VARIANTS.items():
            xs = [fn(arrays[s]) for s in test]
            lg = predict_logits(model, xs, device, mask_only=mask_only)
            preds[name] = {"logits": [float(v) for v in lg],
                           "probs": [float(v) for v in scaler.calibrate(lg)]}
        info.update({"inner_fold": i, "seed": seed, "test": preds})
        rec["inner"].append(info)
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "config": cfg, "arm": arm,
                        "scaler": {"temperature": float(scaler.temperature),
                                   "bias": float(scaler.bias)},
                        "repeat": r, "outer_fold": k, "inner_fold": i,
                        "train": inn["train"], "val": inn["val"], "test": test},
                       os.path.join(ckpt_dir, f"r{r}_k{k}_i{i}.pt"))
    rec["probs"] = np.mean([inf["test"]["original"]["probs"] for inf in rec["inner"]], 0).tolist()
    return rec


# ── main ──────────────────────────────────────────────────────────────────────

def parse_value(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    try:
        return yaml.safe_load(v)
    except Exception:
        return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--tag", default=None, help="output name (default: arm)")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    ap.add_argument("--repeats", type=int, default=None, help="number of repetitions")
    ap.add_argument("--inner_folds", type=int, nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--worker_id", type=int, default=0)
    ap.add_argument("--save_checkpoints", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    overrides = dict(kv.split("=", 1) for kv in args.set)
    cfg = arm_config(args.arm, config, {k: parse_value(v) for k, v in overrides.items()})
    tag = args.tag or args.arm
    part = make_partition(cfg)
    R = args.repeats or cfg["cv"]["n_repeats"]
    inner_folds = args.inner_folds if args.inner_folds is not None else list(range(part["n_inner"]))
    out_dir = os.path.join(cfg["output"]["cv_dir"], "runs", tag)
    ckpt_root = os.path.join(cfg["output"]["cv_models_dir"], tag) if args.save_checkpoints else None
    os.makedirs(out_dir, exist_ok=True)
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arrays = load_arrays(cfg, part["subjects"], cfg["data"].get("align_onset", True))

    jobs = [(r, k) for r in range(R) for k in range(part["n_outer"])]
    jobs = jobs[args.worker_id::args.workers]
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"arm": args.arm, "tag": tag, "overrides": overrides, "config": cfg}, f, indent=1)
    for r, k in jobs:
        path = os.path.join(out_dir, f"r{r}_k{k}.json")
        if os.path.isfile(path):
            continue
        t0 = time.time()
        rec = run_outer_fold(args.arm, cfg, part, arrays, r, k, device, inner_folds, ckpt_root)
        rec["tag"], rec["overrides"] = tag, overrides
        with open(path, "w") as f:
            json.dump(rec, f)
        y = np.array(rec["test_labels"]); p = np.array(rec["probs"])
        from sklearn.metrics import roc_auc_score
        print(f"[{tag}] r{r} k{k} auc={roc_auc_score(y, p):.3f} "
              f"epochs={[inf['best_epoch'] for inf in rec['inner']]} "
              f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
