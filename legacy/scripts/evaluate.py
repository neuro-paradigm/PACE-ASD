"""
PACE-ASD — Evaluation Reproducer

Loads pre-trained checkpoints and reproduces evaluation metrics from the
manuscript using frozen dataset splits.

This script does NOT retrain models. It loads existing checkpoints and
re-runs inference on the test set to reproduce the reported metrics.

Usage:
    python scripts/evaluate.py --model_id A1
    python scripts/evaluate.py --model_id A1 --fold 1 --seed 42
    python scripts/evaluate.py --model_id A1 --all_folds_seeds

Outputs:
    stdout: per-fold/seed metrics and aggregate mean±SD
    results/<model_id>_evaluation_repro.json: JSON metrics
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch
import yaml

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

from model import ASDMotionModel
from calibration import PlattScaler
from dataset import ASDMotionDataset, load_labels, extract_subject_id
from metrics import compute_all_metrics, aggregate_seed_metrics
from train import run_inference as run_model_inference, subject_level_eval


MODEL_VARIANTS = {
    "A1": {"use_gate": True,  "use_transformer": True},
    "A2": {"use_gate": False, "use_transformer": True},
    "A3": {"use_gate": True,  "use_transformer": True},
    "A4": {"use_gate": True,  "use_transformer": False},
}


def evaluate_checkpoint(checkpoint_path: str, test_ids: list, test_labels: list,
                        test_subjects: list, features_dir: str,
                        config: dict, device: torch.device,
                        model_kwargs: dict) -> dict:
    """Evaluate one checkpoint on the provided test set."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = ckpt.get("config", config)

    model = ASDMotionModel(model_cfg, **model_kwargs).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    scaler = None
    if "scaler" in ckpt and ckpt["scaler"] is not None:
        try:
            scaler = pickle.loads(ckpt["scaler"])
        except Exception:
            pass

    # Build test dataloader
    test_ds = ASDMotionDataset(test_ids, test_labels, features_dir, augment=False)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=config["training"]["batch_size"],
        shuffle=False, num_workers=0,
    )

    criterion = torch.nn.BCEWithLogitsLoss()
    test_logits, _, test_labels_raw, _ = run_model_inference(
        model, test_loader, criterion, device, desc="eval"
    )

    test_preds, test_probs, test_labels_subj, _ = subject_level_eval(
        test_logits, test_labels_raw, test_subjects, scaler=scaler,
    )
    return compute_all_metrics(test_labels_subj, test_preds, test_probs)


def main():
    parser = argparse.ArgumentParser(description="PACE-ASD Evaluation Reproducer")
    parser.add_argument("--config",   type=str, default="configs/config.yaml")
    parser.add_argument("--model_id", type=str, default="A1",
                        help="Model variant: A1 | A2 | A3 | A4")
    parser.add_argument("--fold",     type=int, default=None,
                        help="Specific fold to evaluate (1-indexed, default: all)")
    parser.add_argument("--seed",     type=int, default=None,
                        help="Specific seed to evaluate (default: all)")
    parser.add_argument("--all_folds_seeds", action="store_true",
                        help="Evaluate all fold/seed combos and report mean±SD")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    print(f"  Model:  {args.model_id}")

    if args.model_id not in MODEL_VARIANTS:
        print(f"  ERROR: Unknown model_id '{args.model_id}'")
        sys.exit(1)

    model_kwargs = MODEL_VARIANTS[args.model_id]
    if args.model_id == "A3":
        config["model"]["event_block_size"] = 1
        config["model"]["event_top_m"]      = 120

    # Load splits
    splits_file = config["output"]["splits_file"]
    if not os.path.isfile(splits_file):
        print(f"  ERROR: Splits file not found: {splits_file}")
        sys.exit(1)

    with open(splits_file) as f:
        splits = json.load(f)

    from train import resolve_clip_lists
    processed_dir = config["data"]["processed_dir"]
    features_dir  = os.path.join(processed_dir, "features")
    models_dir    = os.path.join(config["output"]["models_dir"], args.model_id)
    results_dir   = config["output"].get("results_dir", "results")

    os.makedirs(results_dir, exist_ok=True)

    n_folds = config["training"]["n_folds"]
    seed_base = config["training"]["seed"]
    n_seeds   = config["training"]["n_seeds"]

    # Determine which folds and seeds to evaluate
    folds_to_eval = [args.fold - 1] if args.fold is not None else list(range(n_folds))
    seeds_to_eval = [args.seed] if args.seed is not None else [seed_base + i for i in range(n_seeds)]

    all_metrics = []
    for fold_idx in folds_to_eval:
        (train_ids, train_labels, train_subjects,
         val_ids, val_labels, val_subjects,
         test_ids, test_labels, test_subjects) = resolve_clip_lists(splits, fold_idx, config)

        for seed in seeds_to_eval:
            ckpt_path = os.path.join(models_dir, f"fold{fold_idx+1}_seed{seed}.pt")
            if not os.path.isfile(ckpt_path):
                print(f"  [SKIP] {ckpt_path} not found")
                continue

            metrics = evaluate_checkpoint(
                ckpt_path, test_ids, test_labels, test_subjects,
                features_dir, config, device, model_kwargs,
            )
            all_metrics.append({"fold": fold_idx+1, "seed": seed, **metrics})
            print(
                f"  Fold {fold_idx+1}  Seed {seed}: "
                f"acc={metrics['accuracy']:.3f}  auc={metrics['auc']:.3f}  "
                f"sens={metrics['sensitivity']:.3f}  spec={metrics['specificity']:.3f}  "
                f"ece={metrics['ece']:.4f}"
            )

    agg = {}
    if len(all_metrics) > 1:
        agg = aggregate_seed_metrics(all_metrics)
        print(f"\n  -- Aggregate ({len(all_metrics)} runs) --")
        print(f"  {'Metric':<15} {'Mean':>8} {'SD':>8}")
        print(f"  {'-'*33}")
        for k, v in agg.items():
            print(f"  {k:<15} {v['mean']:>8.4f} {v['std']:>8.4f}")
    elif len(all_metrics) == 1:
        agg = {k: {"mean": v, "std": 0.0} for k, v in all_metrics[0].items()
               if isinstance(v, float)}

    # Save to file
    out_path = os.path.join(results_dir, f"{args.model_id}_evaluation_repro.json")
    with open(out_path, "w") as f:
        json.dump({"model_id": args.model_id, "runs": all_metrics,
                   "aggregate": agg}, f, indent=2)
    print(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    main()
