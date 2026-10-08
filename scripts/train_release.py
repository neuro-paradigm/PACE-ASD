"""
PACE-ASD — train the released reference ensemble on every child of the cohort.

The children are split into three stratified folds; for each fold and each of
five seeds, one model is trained on the other two folds, its epoch is selected
and its recalibration fitted on the held-out fold. The 15 checkpoints are
written to models/release/ with a manifest. These models have seen every
child, so they are for scoring new recordings, not for estimating
performance on this cohort (use scripts/run_cv.py for that).

Usage:
    python scripts/train_release.py
"""

import json
import os
import sys

import numpy as np
import torch
import yaml
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)

from protocol import arm_config, fit_model          # noqa: E402
from run_cv import cohort, load_arrays               # noqa: E402

OUT = "models/release"
SEEDS = [0, 1, 2, 3, 4]


def main():
    cfg = arm_config("PACE", yaml.safe_load(open("configs/config.yaml")))
    df = cohort(cfg)
    S, y = df["subject_id"].to_numpy(), df["label"].to_numpy()
    arrays = load_arrays(cfg, S.tolist(), True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT, exist_ok=True)
    manifest = []
    folds = StratifiedKFold(3, shuffle=True, random_state=cfg["cv"]["seed"]).split(S, y)
    for f, (tr, va) in enumerate(folds):
        for seed in SEEDS:
            path = os.path.join(OUT, f"pace_fold{f}_seed{seed}.pt")
            if os.path.isfile(path):
                continue
            model, scaler, info = fit_model("PACE", cfg, [arrays[s] for s in S[tr]], y[tr].tolist(),
                                            [arrays[s] for s in S[va]], y[va].tolist(),
                                            20_000 + 10 * f + seed, device)
            torch.save({"state_dict": model.state_dict(), "config": cfg, "arm": "PACE",
                        "scaler": {"temperature": float(scaler.temperature),
                                   "bias": float(scaler.bias)},
                        "train": S[tr].tolist(), "val": S[va].tolist(), "info": info}, path)
            manifest.append({"file": os.path.basename(path), "fold": f, "seed": seed,
                             "n_train": int(len(tr)), "n_val": int(len(va)), **info})
            print(f"fold {f} seed {seed}: val AUC {info['val_auc']:.3f} "
                  f"(epoch {info['best_epoch']}, {info['seconds']} s)", flush=True)
    if manifest:
        json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
