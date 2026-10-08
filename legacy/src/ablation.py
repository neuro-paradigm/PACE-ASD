"""
PACE-ASD — Ablation Runner (Protocol Section 4)

Runs models A1–A5 in sequence, each with 20 seeds × 3-fold CV.
Saves results/ablation_results.csv and results/ablation_table.pdf.

Optional --eval_supplement evaluates PACE-ASD variants (A1–A4) AND A5 baselines
on the 14-subject supplementary cohort (5 regular ASD + 9 severe-ASD, all ASD,
sensitivity only) and writes results/supplement_results.csv.

Usage:
    python src/ablation.py --config configs/config.yaml
    python src/ablation.py --config configs/config.yaml --models A1 A2
    python src/ablation.py --config configs/config.yaml --eval_supplement
"""

import argparse
import csv
import json
import os
import pickle
import sys
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from dataset import ASDMotionDataset, load_labels, extract_subject_id
from model import ASDMotionModel
from train import (
    freeze_splits, run_all_seeds, run_inference,
    subject_level_eval, set_seed,
)
from baselines import (
    PYTORCH_BASELINES, SKLEARN_BASELINES,
    build_sklearn_baseline, extract_sklearn_features,
    StackedLSTM, Conv1DBiLSTMAttn,
)
from metrics import (
    compute_all_metrics, aggregate_seed_metrics,
    compute_supplement_sensitivity,
)
from calibration import PlattScaler
from report import generate_ablation_table


# ── Ablation arm definitions ─────────────────────────────────────────────────

ABLATION_ARMS = {
    "A1": {
        "desc":         "Full PACE-ASD (L=15, M=8)",
        "model_type":   "pace",
        "model_kwargs": {"use_gate": True,  "use_transformer": True},
        "config_patch": {},
    },
    "A2": {
        "desc":         "No-Block-ESG (dense Transformer)",
        "model_type":   "pace",
        "model_kwargs": {"use_gate": False, "use_transformer": True},
        "config_patch": {},
    },
    "A3": {
        "desc":         "Frame-granularity gate (L=1, M=120)",
        "model_type":   "pace",
        "model_kwargs": {"use_gate": True,  "use_transformer": True},
        "config_patch": {"model.event_block_size": 1, "model.event_top_m": 120},
    },
    "A4": {
        "desc":         "No-Transformer (linear head on ESG output)",
        "model_type":   "pace",
        "model_kwargs": {"use_gate": True,  "use_transformer": False},
        "config_patch": {},
    },
}

# A5 baselines (each trained with the same 20-seed × 3-fold protocol)
A5_BASELINES = {
    # ── Protocol A5 ────────────────────────────────────────────────────────────
    "A5_lstm":           {"desc": "Stacked LSTM",              "type": "lstm"},
    "A5_conv1d_bilstm":  {"desc": "Conv1D-BiLSTM-Attn",        "type": "conv1d_bilstm"},
    "A5_kinematic_cnn":  {"desc": "Kinematic CNN-LSTM",         "type": "kinematic_cnn"},
    "A5_stts":           {"desc": "STTS",                       "type": "stts"},
    "A5_msg3d":          {"desc": "MS-G3D",                     "type": "msg3d"},
    "A5_msg3d_convnext": {"desc": "MS-G3D + ConvNeXt",          "type": "msg3d_convnext"},
    # ── Section 8: Table 1 repositioning (Tier A) ─────────────────────────────
    "A5_skelformer":     {"desc": "SkelFormer",                 "type": "skelformer"},
    "A5_mtcformer":      {"desc": "MTC-Former",                 "type": "mtcformer"},
    "A5_mtt":            {"desc": "MTT",                        "type": "mtt"},
    "A5_star":           {"desc": "STAR",                       "type": "star"},
    # ── MediaPipe + sklearn ────────────────────────────────────────────────────
    "A5_lr":             {"desc": "MediaPipe + LR",             "type": "lr"},
    "A5_svm":            {"desc": "MediaPipe + SVM (RBF)",      "type": "svm"},
    "A5_rf":             {"desc": "MediaPipe + RF",             "type": "rf"},
    "A5_xgboost":        {"desc": "MediaPipe + XGBoost",        "type": "xgboost"},
}


# ── Config patching helper ────────────────────────────────────────────────────

def apply_patch(config: dict, patch: dict) -> dict:
    """Apply dot-separated key patches, e.g. 'model.event_block_size': 1."""
    import copy
    cfg = copy.deepcopy(config)
    for key, val in patch.items():
        parts = key.split(".")
        d = cfg
        for p in parts[:-1]:
            d = d[p]
        d[parts[-1]] = val
    return cfg


# ── PyTorch baseline training ─────────────────────────────────────────────────

def train_pytorch_baseline_fold(
    baseline_class, fold_idx, train_ids, train_labels, train_subjects,
    val_ids, val_labels, val_subjects,
    test_ids, test_labels, test_subjects,
    config, seed, device, save_dir,
):
    """
    Train a PyTorch baseline for one fold with methodological parity:
    subject-level aggregation, combined score checkpoint selection,
    and post-hoc PlattScaler calibration.
    """
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from dataset import create_dataloaders
    from calibration import PlattScaler

    tc           = config["training"]
    features_dir = os.path.join(config["data"]["processed_dir"], "features")

    train_loader, val_loader = create_dataloaders(
        train_ids, train_labels, train_subjects,
        val_ids, val_labels, features_dir,
        batch_size=tc["batch_size"], num_workers=tc["num_workers"],
    )

    model     = baseline_class().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=tc["epochs"], eta_min=1e-6)

    best_score    = -float("inf")
    best_state    = None
    patience_ctr  = 0
    history       = []

    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        total_loss, n = 0.0, 0
        for seqs, labels in train_loader:
            seqs   = seqs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            _, logits = model(seqs)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1
        scheduler.step()
        train_loss = total_loss / max(n, 1)

        # Val inference
        val_logits_arr, _, val_labels_arr, _ = run_inference(
            model, val_loader, criterion, device,
        )
        val_preds, val_probs, val_labels_subj, val_logits_subj = subject_level_eval(
            val_logits_arr, val_labels_arr, val_subjects, scaler=None,
        )

        val_logits_t = torch.from_numpy(val_logits_subj).float().to(device)
        val_labels_t = torch.from_numpy(val_labels_subj).float().to(device)
        val_loss     = nn.BCEWithLogitsLoss()(val_logits_t, val_labels_t).item()

        metrics = compute_all_metrics(val_labels_subj, val_preds, val_probs)
        metrics["train_loss"] = train_loss
        metrics["val_loss"]   = val_loss
        history.append(metrics)

        prev_vl = history[-2]["val_loss"] if len(history) >= 2 else val_loss
        prev_tl = history[-2]["train_loss"] if len(history) >= 2 else train_loss
        smooth_vl = 0.6 * val_loss + 0.4 * prev_vl
        smooth_tl = 0.6 * train_loss + 0.4 * prev_tl
        val_auc   = metrics["auc"] if not np.isnan(metrics.get("auc", float("nan"))) else 0.5
        val_ece   = metrics.get("ece", 0.1)

        gap = max(0.0, smooth_vl - smooth_tl)
        combined = val_auc - 0.5 * smooth_vl - 0.4 * val_ece - 0.3 * gap

        is_collapsed = (
            metrics["sensitivity"] <= 0.05 or
            metrics["specificity"] <= 0.05 or
            np.isnan(val_loss)
        )

        if combined > best_score and not is_collapsed:
            best_score   = combined
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if patience_ctr >= tc["early_stopping_patience"]:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Post-hoc Platt Calibration on validation set
    val_logits_arr, _, val_labels_arr, _ = run_inference(
        model, val_loader, criterion, device,
    )
    _, _, val_labels_subj_cal, val_logits_subj_cal = subject_level_eval(
        val_logits_arr, val_labels_arr, val_subjects, scaler=None,
    )
    scaler = PlattScaler()
    scaler.fit(val_logits_subj_cal, val_labels_subj_cal,
               lr=config["calibration"]["lr"],
               max_iter=config["calibration"]["max_iter"])

    val_preds_cal, val_probs_cal, val_labels_subj_cal, _ = subject_level_eval(
        val_logits_arr, val_labels_arr, val_subjects, scaler=scaler,
    )
    val_metrics = compute_all_metrics(val_labels_subj_cal, val_preds_cal, val_probs_cal)

    # Test evaluation with calibrated scaler
    test_ds = ASDMotionDataset(test_ids, test_labels, features_dir, augment=False)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=tc["batch_size"], shuffle=False,
        num_workers=tc["num_workers"],
    )
    test_logits_arr, _, test_labels_arr, _ = run_inference(
        model, test_loader, criterion, device,
    )
    test_preds, test_probs, test_labels_subj, _ = subject_level_eval(
        test_logits_arr, test_labels_arr, test_subjects, scaler=scaler,
    )
    test_metrics = compute_all_metrics(test_labels_subj, test_preds, test_probs)

    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "scaler":     pickle.dumps(scaler),
        "config":     config,
    }, os.path.join(save_dir, f"fold{fold_idx}_seed{seed}.pt"))

    return val_metrics, test_metrics


# ── Sklearn baseline training ─────────────────────────────────────────────────

def run_sklearn_baseline(
    baseline_name, splits, config, n_seeds, device, baseline_id=None,
):
    """Run one sklearn baseline across n_seeds × 3-fold CV.
    
    Args:
        baseline_name: Key into SKLEARN_BASELINES.
        splits: Frozen split dict from freeze_splits().
        config: Global config dict.
        n_seeds: Number of seeds.
        device: Torch device (unused but kept for API consistency).
        baseline_id: Optional str ID for per-seed JSON filename.
                     If None, raw per-seed data is not persisted.
    """
    import copy
    from train import resolve_clip_lists

    seed_base    = config["training"]["seed"]
    features_dir = os.path.join(config["data"]["processed_dir"], "features")
    all_val, all_test = [], []

    for i in range(n_seeds):
        seed = seed_base + i
        set_seed(seed)
        fold_val, fold_test = [], []

        for fold_idx in range(config["training"]["n_folds"]):
            (train_ids, train_labels, train_subjects,
             val_ids,   val_labels,   val_subjects,
             test_ids,  test_labels,  test_subjects) = resolve_clip_lists(
                 splits, fold_idx, config,
             )

            train_paths = [os.path.join(features_dir, f"{c}.npy") for c in train_ids]
            val_paths   = [os.path.join(features_dir, f"{c}.npy") for c in val_ids]
            test_paths  = [os.path.join(features_dir, f"{c}.npy") for c in test_ids]

            X_train = extract_sklearn_features(train_paths)
            X_val   = extract_sklearn_features(val_paths)
            X_test  = extract_sklearn_features(test_paths)

            pipe = build_sklearn_baseline(baseline_name, seed=seed)
            pipe.fit(X_train, train_labels)

            # Val
            val_probs   = pipe.predict_proba(X_val)[:, 1]
            val_preds   = (val_probs >= 0.5).astype(int)
            fold_val.append(compute_all_metrics(
                np.array(val_labels), val_preds, val_probs,
            ))

            # Test
            test_probs  = pipe.predict_proba(X_test)[:, 1]
            test_preds  = (test_probs >= 0.5).astype(int)
            fold_test.append(compute_all_metrics(
                np.array(test_labels), test_preds, test_probs,
            ))

        all_val.append({
            k: float(np.mean([m[k] for m in fold_val]))
            for k in ["accuracy","auc","f1","sensitivity","specificity","ece","threshold"]
        })
        all_test.append({
            k: float(np.mean([m[k] for m in fold_test]))
            for k in ["accuracy","auc","f1","sensitivity","specificity","ece","threshold"]
        })

    # Persist raw per-seed data when an ID is provided
    if baseline_id is not None:
        results_dir = config["output"].get("results_dir", "results")
        os.makedirs(results_dir, exist_ok=True)
        per_seed_path = os.path.join(results_dir, f"{baseline_id}_per_seed.json")
        with open(per_seed_path, "w") as _psf:
            json.dump({"val": all_val, "test": all_test}, _psf, indent=2)
        print(f"  [INFO] Per-seed metrics -> {per_seed_path}")

    return {
        "val":  aggregate_seed_metrics(all_val),
        "test": aggregate_seed_metrics(all_test),
    }



# ── Supplementary evaluation (Section 6) ─────────────────────────────────────

def eval_supplement(model_id: str, splits: dict, config: dict,
                    device: torch.device) -> dict:
    """
    Evaluate saved A1–A4 models on the 9 severe-ASD subjects.
    Reports sensitivity only (ASD-only group — no AUC/specificity).
    """
    supp        = splits["supplement"]
    clip_ids    = supp["clip_ids"]
    labels      = supp["labels"]
    subject_ids = supp["subject_ids"]
    features_dir = os.path.join(config["data"]["processed_dir"], "features")

    models_dir = os.path.join(config["output"]["models_dir"], model_id)
    if not os.path.isdir(models_dir):
        print(f"  [WARN] No saved models for {model_id} — skipping supplement eval.")
        return {}

    model_files = sorted([f for f in os.listdir(models_dir) if f.endswith(".pt")])
    if not model_files:
        return {}

    all_preds = []
    subj_labels_ref = None
    n_skipped = 0

    for mf in model_files:
        try:
            ckpt = torch.load(os.path.join(models_dir, mf),
                              map_location=device, weights_only=False)

            ckpt_model_cfg = ckpt.get("config", {}).get("model", {})
            if ckpt_model_cfg and ckpt_model_cfg != config.get("model", {}):
                print(f"  [WARN] {mf}: config mismatch, skipping.")
                n_skipped += 1
                continue

            model = ASDMotionModel(config, **_model_kwargs_for_id(model_id)).to(device)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            scaler = pickle.loads(ckpt.get("scaler", pickle.dumps(PlattScaler())))

            ds = ASDMotionDataset(clip_ids, labels, features_dir, augment=False)
            loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
            criterion = torch.nn.BCEWithLogitsLoss()
            logits_arr, _, labels_arr, _ = run_inference(model, loader, criterion, device)

            preds, _, subj_labels_out, _ = subject_level_eval(
                logits_arr, labels_arr, subject_ids, scaler=scaler,
            )
            all_preds.append(preds)
            subj_labels_ref = subj_labels_out

        except Exception as e:
            print(f"  [WARN] {mf}: {e}, skipping.")
            n_skipped += 1
            continue


    if not all_preds:
        print(f"  [WARN] No usable models for {model_id} supplement eval.")
        return {}
    if n_skipped:
        print(f"  [INFO] {model_id}: {n_skipped}/{len(model_files)} checkpoints skipped.")

    ensemble_preds = (np.mean(all_preds, axis=0) >= 0.5).astype(int)
    return compute_supplement_sensitivity(subj_labels_ref, ensemble_preds)


def eval_supplement_baseline(
    baseline_id: str, baseline_spec: dict, splits: dict,
    config: dict, device: torch.device, n_seeds: int,
) -> dict:
    """
    Evaluate an A5 baseline on the supplement cohort.

    PyTorch baselines: load already-saved fold/seed checkpoints from
    models/{baseline_id}/*.pt — identical to eval_supplement for A1-A4.
    No retraining; all 60 checkpoints are already on disk.

    Sklearn baselines: fit once on the full combined trainval pool.
    No checkpoints are saved for sklearn models so refit is unavoidable,
    but it takes <1 s and requires only one pass.
    """
    from train import resolve_clip_lists

    supp         = splits["supplement"]
    supp_ids     = supp["clip_ids"]
    supp_labels  = supp["labels"]
    supp_subjs   = supp["subject_ids"]
    features_dir = os.path.join(config["data"]["processed_dir"], "features")

    btype      = baseline_spec.get("type", "")
    is_sklearn = btype in SKLEARN_BASELINES

    # ── PyTorch baseline: load saved checkpoints (no retraining) ─────────────
    if not is_sklearn:
        from baselines import PYTORCH_BASELINES
        from calibration import PlattScaler

        models_dir = os.path.join(config["output"]["models_dir"], baseline_id)
        if not os.path.isdir(models_dir):
            print(f"  [WARN] No saved checkpoints for {baseline_id} — skipping.")
            return {}

        model_files = sorted([f for f in os.listdir(models_dir) if f.endswith(".pt")])
        if not model_files:
            print(f"  [WARN] No .pt files found in {models_dir} — skipping.")
            return {}

        baseline_class = PYTORCH_BASELINES.get(btype)
        if baseline_class is None:
            print(f"  [WARN] Unknown PyTorch baseline type '{btype}' — skipping.")
            return {}

        ds        = ASDMotionDataset(supp_ids, supp_labels, features_dir, augment=False)
        loader    = torch.utils.data.DataLoader(
            ds, batch_size=config["training"]["batch_size"],
            shuffle=False, num_workers=0,
        )
        criterion = torch.nn.BCEWithLogitsLoss()

        all_preds       = []
        subj_labels_ref = None
        n_skipped       = 0

        for mf in model_files:
            try:
                ckpt = torch.load(os.path.join(models_dir, mf),
                                  map_location=device, weights_only=False)
                model = baseline_class().to(device)
                model.load_state_dict(ckpt["state_dict"])
                model.eval()
                scaler = pickle.loads(ckpt.get("scaler", pickle.dumps(PlattScaler())))

                logits_arr, _, labels_arr, _ = run_inference(
                    model, loader, criterion, device,
                )
                preds, _, subj_labels_out, _ = subject_level_eval(
                    logits_arr, labels_arr, supp_subjs, scaler=scaler,
                )
                all_preds.append(preds)
                subj_labels_ref = subj_labels_out

            except Exception as e:
                print(f"  [WARN] {mf}: {e}, skipping.")
                n_skipped += 1
                continue

        if not all_preds:
            print(f"  [WARN] No usable checkpoints for {baseline_id}.")
            return {}
        if n_skipped:
            print(f"  [INFO] {baseline_id}: {n_skipped}/{len(model_files)} checkpoints skipped.")

        ensemble_preds = (np.mean(all_preds, axis=0) >= 0.5).astype(int)
        return compute_supplement_sensitivity(subj_labels_ref, ensemble_preds)

    # ── Sklearn baseline: fit once on combined trainval pool ──────────────────
    seed = config["training"]["seed"]
    set_seed(seed)

    all_train_ids, all_train_labels = [], []
    for fold_idx in range(config["training"]["n_folds"]):
        (tr_ids, tr_labels, _tr_subjs,
         val_ids, val_labels, _val_subjs,
         _test_ids, _test_labels, _test_subjs) = resolve_clip_lists(
            splits, fold_idx, config,
        )
        all_train_ids    += list(tr_ids)    + list(val_ids)
        all_train_labels += list(tr_labels) + list(val_labels)

    train_paths = [os.path.join(features_dir, f"{c}.npy") for c in all_train_ids]
    supp_paths  = [os.path.join(features_dir, f"{c}.npy") for c in supp_ids]

    X_train = extract_sklearn_features(train_paths)
    X_supp  = extract_sklearn_features(supp_paths)

    pipe = build_sklearn_baseline(btype, seed=seed)
    pipe.fit(X_train, all_train_labels)
    supp_probs = pipe.predict_proba(X_supp)[:, 1]
    supp_subjs = [c.rsplit('_', 1)[0] if 'severe_' in c else c for c in supp_ids]
    unique_subjs = list(dict.fromkeys(supp_subjs))
    subj_probs = {}
    for s, p in zip(supp_subjs, supp_probs):
        subj_probs.setdefault(s, []).append(p)
    subj_preds = [int(np.mean(subj_probs[s]) >= 0.5) for s in unique_subjs]
    subj_labels = [1] * len(unique_subjs)

    return compute_supplement_sensitivity(subj_labels, subj_preds)



def _model_kwargs_for_id(model_id: str) -> dict:
    if model_id in ABLATION_ARMS:
        return ABLATION_ARMS[model_id]["model_kwargs"]
    return {"use_gate": True, "use_transformer": True}


# ── Main ablation runner ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PACE-ASD ablation runner")
    parser.add_argument("--config",  default="configs/config.yaml")
    parser.add_argument("--models",  nargs="*", default=None,
                        help="Subset of models to run (default: all A1–A5)")
    parser.add_argument("--eval_supplement", action="store_true",
                        help="After training, evaluate on 14 supplementary ASD subjects")
    parser.add_argument("--eval_supplement_only", action="store_true",
                        help="Skip training and run only supplementary evaluation using saved checkpoints")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_seeds = config["training"]["n_seeds"]
    results_dir = config["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"  PACE-ASD Ablation Study")
    print(f"  Device  : {device}")
    print(f"  Seeds   : {n_seeds}")
    print(f"  Folds   : {config['training']['n_folds']}")
    print(f"{'#'*60}\n")

    splits = freeze_splits(config)

    # Which arms to run?
    run_pace      = list(ABLATION_ARMS.keys())
    run_baselines = list(A5_BASELINES.keys())
    if args.eval_supplement_only:
        run_pace      = []
        run_baselines = []
        args.eval_supplement = True
    elif args.models is not None:
        run_pace      = [m for m in run_pace      if m in args.models]
        run_baselines = [m for m in run_baselines if m in args.models]

    all_results = {}


    # ── A1–A4 ────────────────────────────────────────────────────────────────
    for model_id, arm in ABLATION_ARMS.items():
        if model_id not in run_pace:
            continue
        print(f"\n{'='*60}")
        print(f"  Running {model_id}: {arm['desc']}")
        print(f"{'='*60}")

        cfg = apply_patch(config, arm["config_patch"])
        res = run_all_seeds(
            cfg, n_seeds, arm["model_kwargs"],
            model_id, splits, device,
        )
        all_results[model_id] = res
        print(f"\n  {model_id} SUMMARY (mean ± SD across {n_seeds} seeds):")
        print(f"    {'Metric':<15} {'Val (CV)':<20} {'Held-Out Test':<20}")
        print(f"    {'-'*55}")
        for k in ["accuracy", "auc", "f1", "sensitivity", "specificity", "ece", "threshold"]:
            v_val  = res.get("val", {}).get(k, {})
            v_test = res.get("test", {}).get(k, {})
            v_str  = f"{v_val.get('mean', float('nan')):.4f} ± {v_val.get('std', float('nan')):.4f}"
            t_str  = f"{v_test.get('mean', float('nan')):.4f} ± {v_test.get('std', float('nan')):.4f}"
            print(f"    {k:<15} {v_str:<20} {t_str:<20}")

    # ── A5 Sklearn baselines ─────────────────────────────────────────────────
    for bid, bspec in A5_BASELINES.items():
        if bid not in run_baselines:
            continue
        btype = bspec["type"]
        print(f"\n{'='*60}")
        print(f"  Running {bid}: {bspec['desc']}")
        print(f"{'='*60}")

        if btype in PYTORCH_BASELINES:
            # PyTorch baseline: reuse run_all_seeds logic with a shim
            cls     = PYTORCH_BASELINES[btype]
            seed_base = config["training"]["seed"]
            all_val, all_test = [], []
            from train import resolve_clip_lists
            for i in range(n_seeds):
                seed = seed_base + i
                set_seed(seed)
                fv, ft = [], []
                for fi in range(config["training"]["n_folds"]):
                    (tr_id, tr_lbl, tr_sub,
                     va_id, va_lbl, va_sub,
                     te_id, te_lbl, te_sub) = resolve_clip_lists(splits, fi, config)
                    vam, tem = train_pytorch_baseline_fold(
                        cls, fi+1, tr_id, tr_lbl, tr_sub,
                        va_id, va_lbl, va_sub,
                        te_id, te_lbl, te_sub,
                        config, seed, device,
                        os.path.join(config["output"]["models_dir"], bid),
                    )
                    fv.append(vam); ft.append(tem)
                keys = ["accuracy","auc","f1","sensitivity","specificity","ece", "threshold"]
                all_val.append({k: float(np.mean([m[k] for m in fv if k in m])) for k in keys if any(k in m for m in fv)})
                all_test.append({k: float(np.mean([m[k] for m in ft if k in m])) for k in keys if any(k in m for m in ft)})
            # Persist raw per-seed metrics for downstream paired tests
            per_seed_path = os.path.join(results_dir, f"{bid}_per_seed.json")
            with open(per_seed_path, "w") as _psf:
                json.dump({"val": all_val, "test": all_test}, _psf, indent=2)
            print(f"  [INFO] Per-seed metrics -> {per_seed_path}")
            res = {"val": aggregate_seed_metrics(all_val),
                   "test": aggregate_seed_metrics(all_test)}
        else:
            res = run_sklearn_baseline(btype, splits, config, n_seeds, device,
                                       baseline_id=bid)

        all_results[bid] = res
        print(f"\n  {bid} SUMMARY (mean ± SD across {n_seeds} seeds):")
        print(f"    {'Metric':<15} {'Val (CV)':<20} {'Held-Out Test':<20}")
        print(f"    {'-'*55}")
        for k in ["accuracy", "auc", "f1", "sensitivity", "specificity", "ece", "threshold"]:
            v_val  = res.get("val", {}).get(k, {})
            v_test = res.get("test", {}).get(k, {})
            v_str  = f"{v_val.get('mean', float('nan')):.4f} ± {v_val.get('std', float('nan')):.4f}"
            t_str  = f"{v_test.get('mean', float('nan')):.4f} ± {v_test.get('std', float('nan')):.4f}"
            print(f"    {k:<15} {v_str:<20} {t_str:<20}")

    # ── Save results CSV (only if models were trained in this run) ────────────
    if all_results:
        csv_path = os.path.join(results_dir, "ablation_results.csv")
        metric_keys = ["accuracy", "auc", "f1", "sensitivity", "specificity", "ece", "threshold"]
        with open(csv_path, "w", newline="") as f:
            cols = ["model_id", "description", "split"] + [
                f"{k}_{s}" for k in metric_keys for s in ["mean", "std"]
            ]
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for mid, res in all_results.items():
                desc = (ABLATION_ARMS.get(mid, A5_BASELINES.get(mid, {}))
                        .get("desc", mid))
                for split_name in ["val", "test"]:
                    row = {"model_id": mid, "description": desc,
                           "split": split_name}
                    for k in metric_keys:
                        v = res[split_name].get(k, {})
                        row[f"{k}_mean"] = v.get("mean", float("nan"))
                        row[f"{k}_std"]  = v.get("std",  float("nan"))
                    writer.writerow(row)
        print(f"\n  Results CSV -> {csv_path}")

        # ── Generate ablation table PDF ───────────────────────────────────────────
        generate_ablation_table(all_results, results_dir)


    # ── Supplementary evaluation (Section 6) ──────────────────────────────────
    if args.eval_supplement:
        n_supp = len(splits.get("supplement", {}).get("clip_ids", []))
        print(f"\n{'='*60}")
        print(f"  Section 6: Supplementary Evaluation ({n_supp} clips, all-ASD, sensitivity only)")
        print(f"{'='*60}")

        supp_rows = []

        # ── A1–A4 PACE-ASD variants (load saved .pt checkpoints) ──────────────
        print(f"\n  A1–A4 (PACE-ASD ablation variants):")
        for model_id in list(ABLATION_ARMS.keys()):
            cfg = apply_patch(config, ABLATION_ARMS[model_id]["config_patch"])
            s   = eval_supplement(model_id, splits, cfg, device)
            if s:
                lo, hi = s.get("sensitivity_ci", (float("nan"), float("nan")))
                print(f"    {model_id}: sensitivity={s.get('sensitivity', float('nan')):.3f} "
                      f"95%CI [{lo:.3f}, {hi:.3f}]  (n={s.get('n',0)})")
                supp_rows.append({
                    "model_id":    model_id,
                    "description": ABLATION_ARMS[model_id]["desc"],
                    "n":           s.get("n", 0),
                    "sensitivity": s.get("sensitivity", float("nan")),
                    "ci_low":      lo,
                    "ci_high":     hi,
                })

        # ── A5 baselines (re-fit on trainval, evaluate on supplement) ──────────
        print(f"\n  A5 baselines (re-fit on trainval, predict on supplement):")
        n_seeds = config["training"].get("n_seeds", 20)
        for bid, bspec in A5_BASELINES.items():
            print(f"    {bid} ({bspec['desc']}) ...", end=" ", flush=True)
            try:
                s = eval_supplement_baseline(
                    bid, bspec, splits, config, device, n_seeds=n_seeds,
                )
                if s:
                    lo, hi = s.get("sensitivity_ci", (float("nan"), float("nan")))
                    print(f"sensitivity={s.get('sensitivity', float('nan')):.3f} "
                          f"95%CI [{lo:.3f}, {hi:.3f}]  (n={s.get('n',0)})")
                    supp_rows.append({
                        "model_id":    bid,
                        "description": bspec["desc"],
                        "n":           s.get("n", 0),
                        "sensitivity": s.get("sensitivity", float("nan")),
                        "ci_low":      lo,
                        "ci_high":     hi,
                    })
                else:
                    print("no result (skipped)")
            except Exception as exc:
                print(f"ERROR: {exc}")

        supp_csv = os.path.join(results_dir, "supplement_results.csv")
        if supp_rows:
            with open(supp_csv, "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["model_id","description","n","sensitivity","ci_low","ci_high"])
                writer.writeheader()
                writer.writerows(supp_rows)
            print(f"\n  Supplement results -> {supp_csv}")


    print(f"\n{'#'*60}")
    print(f"  Ablation complete.")
    print(f"  Results : {results_dir}/")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
