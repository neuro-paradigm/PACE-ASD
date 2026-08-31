"""
PACE-ASD — Training Pipeline (Protocol Sections 1.2, 3, 4, 5)

Entry points:
  freeze_splits(config)
      Create splits_dryad_only_v1.json (idempotent).

  train_one_fold(fold_data, config, seed, model_kwargs)
      Train one fold with early stopping + Platt calibration.

  run_cv(config, seed, model_kwargs, model_id)
      3-fold StratifiedGroupKFold, returns per-fold metrics.

  run_all_seeds(config, n_seeds, model_kwargs, model_id)
      20-seed outer loop → mean ± SD metrics.

CLI (single-model sanity check):
  python src/train.py --config configs/config.yaml --model_id A1 [--seed 0] [--fold 0]
"""

import argparse
import json
import math
import os
import pickle
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from sklearn.model_selection import StratifiedKFold, train_test_split
import yaml
from tqdm import tqdm

# Project imports (run from repo root: python src/train.py)
sys.path.insert(0, os.path.dirname(__file__))
from dataset import (
    ASDMotionDataset, SubjectSampledDataset,
    create_dataloaders, load_labels, extract_subject_id,
)
from model import ASDMotionModel
from metrics import (
    compute_all_metrics, compute_confusion_matrix,
    compute_roc_curve, aggregate_seed_metrics,
)
from calibration import PlattScaler
from report import generate_fold_reports


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Split management ──────────────────────────────────────────────────────────

def freeze_splits(config: dict) -> dict:
    """
    Create and persist splits_dryad_only_v1.json (idempotent).
    Uses 25% of subjects as held-out test, remainder for 3-fold CV.
    All 20 seeds and all ablation arms share the same frozen splits.

    Returns the splits dict.
    """
    splits_path = config["output"]["splits_file"]
    if os.path.isfile(splits_path):
        with open(splits_path, "r") as f:
            return json.load(f)

    # Load labels
    processed_dir = config["data"]["processed_dir"]
    df = load_labels(processed_dir)

    # Regular subjects only (group == 'regular')
    regular_df = df[df["group"] == "regular"].copy()
    clip_ids   = regular_df["clip_id"].tolist()
    labels     = regular_df["label"].tolist()
    subj_ids   = [extract_subject_id(cid) for cid in clip_ids]

    # Unique subjects
    unique_subjs = sorted(set(subj_ids))
    subj_to_label = {}
    for cid, lbl, sid in zip(clip_ids, labels, subj_ids):
        subj_to_label[sid] = int(lbl)
    subj_labels = [subj_to_label[s] for s in unique_subjs]

    # Subject-level train/test split (stratified)
    seed          = config["training"]["seed"]
    test_ratio    = config["training"]["test_split_ratio"]
    train_val_subjs, test_subjs = train_test_split(
        unique_subjs, test_size=test_ratio,
        stratify=subj_labels, random_state=seed,
    )

    # 3-fold StratifiedKFold on train_val subjects
    tv_labels = [subj_to_label[s] for s in train_val_subjs]
    n_folds   = config["training"]["n_folds"]
    skf       = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    folds = []
    for train_idx, val_idx in skf.split(train_val_subjs, tv_labels):
        folds.append({
            "train": [train_val_subjs[i] for i in train_idx],
            "val":   [train_val_subjs[i] for i in val_idx],
        })

    # Supplement subjects
    supp_df     = df[df["group"] == "supplement"].copy()
    supp_clips  = supp_df["clip_id"].tolist()
    supp_labels = supp_df["label"].tolist()
    supp_subjs  = [extract_subject_id(cid) for cid in supp_clips]

    splits = {
        "version":           "splits_dryad_only_v1",
        "seed":              seed,
        "test_subjects":     sorted(test_subjs),
        "trainval_subjects": sorted(train_val_subjs),
        "folds":             folds,
        "supplement": {
            "clip_ids":    supp_clips,
            "labels":      supp_labels,
            "subject_ids": supp_subjs,
        },
    }

    os.makedirs(os.path.dirname(splits_path), exist_ok=True)
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"  Splits frozen → {splits_path}")
    print(f"  Train/val subjects : {len(train_val_subjs)}")
    print(f"  Test subjects      : {len(test_subjs)}")
    print(f"  Supplement clips   : {len(supp_clips)}")
    return splits


def resolve_clip_lists(splits: dict, fold_idx: int, config: dict):
    """
    From the frozen splits, return:
        train_ids, train_labels, train_subjects,
        val_ids,   val_labels,   val_subjects,
        test_ids,  test_labels,  test_subjects
    """
    processed_dir = config["data"]["processed_dir"]
    df = load_labels(processed_dir)
    regular = df[df["group"] == "regular"].copy()

    clip_ids = regular["clip_id"].tolist()
    labels   = regular["label"].tolist()
    subj_ids = [extract_subject_id(cid) for cid in clip_ids]

    fold         = splits["folds"][fold_idx]
    train_set    = set(fold["train"])
    val_set      = set(fold["val"])
    test_set     = set(splits["test_subjects"])

    train_ids, train_labels, train_subjects = [], [], []
    val_ids,   val_labels,   val_subjects   = [], [], []
    test_ids,  test_labels,  test_subjects  = [], [], []

    for cid, lbl, sid in zip(clip_ids, labels, subj_ids):
        if sid in train_set:
            train_ids.append(cid); train_labels.append(lbl); train_subjects.append(sid)
        elif sid in val_set:
            val_ids.append(cid);   val_labels.append(lbl);   val_subjects.append(sid)
        elif sid in test_set:
            test_ids.append(cid);  test_labels.append(lbl);  test_subjects.append(sid)

    return (train_ids, train_labels, train_subjects,
            val_ids,   val_labels,   val_subjects,
            test_ids,  test_labels,  test_subjects)


# ── Training helpers ──────────────────────────────────────────────────────────

def subject_level_eval(all_logits, all_labels, subject_ids, scaler, threshold=0.5):
    """
    Aggregate clip logits to subject level (mean over clips per subject),
    apply calibration, threshold, and return subject-level predictions.

    For the Dryad-only dataset each regular subject has exactly 1 clip, so
    aggregation is a no-op for regular subjects. It matters for severe-ASD
    case2 (2 clips) during supplementary evaluation.
    """
    subj_data = {}
    for logit, lbl, sid in zip(all_logits, all_labels, subject_ids):
        if sid not in subj_data:
            subj_data[sid] = {"logits": [], "label": lbl}
        subj_data[sid]["logits"].append(logit)

    subj_logits = np.array([np.mean(v["logits"]) for v in subj_data.values()])
    subj_labels = np.array([v["label"]            for v in subj_data.values()])

    if scaler is not None:
        subj_probs = scaler.calibrate(subj_logits)
    else:
        subj_probs = torch.sigmoid(torch.tensor(subj_logits)).numpy()

    subj_preds = (subj_probs >= threshold).astype(int)
    return subj_preds, subj_probs, subj_labels, subj_logits


@torch.no_grad()
def run_inference(model, dataloader, criterion, device, desc=""):
    """Collect all logits, probs, labels for a dataset."""
    model.eval()
    all_logits, all_probs, all_labels = [], [], []
    total_loss, n_batches = 0.0, 0

    pbar = tqdm(dataloader, desc=f"  [{desc}]", leave=False, unit="batch")
    for seqs, labels in pbar:
        seqs   = seqs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        probs, logits = model(seqs)
        loss   = criterion(logits, labels)

        total_loss += loss.item()
        n_batches  += 1
        all_logits.append(logits.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    return (
        np.concatenate(all_logits),
        np.concatenate(all_probs),
        np.concatenate(all_labels),
        total_loss / max(n_batches, 1),
    )


def train_one_epoch(model, loader, criterion, optimizer, device,
                    epoch: int, fold: int, total_epochs: int) -> float:
    model.train()
    total_loss, n = 0.0, 0
    label_smooth  = 0.04

    pbar = tqdm(loader, desc=f"  F{fold} E{epoch} [train]",
                leave=False, unit="batch")
    for seqs, labels in pbar:
        seqs   = seqs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        # Sequence Mixup (beta=0.2, 35% probability) for smooth regularization
        if seqs.size(0) > 1 and np.random.rand() < 0.35:
            lam = np.random.beta(0.2, 0.2)
            perm = torch.randperm(seqs.size(0))
            mixed_seqs = lam * seqs + (1.0 - lam) * seqs[perm]
            mixed_labels = lam * labels + (1.0 - lam) * labels[perm]
            _, logits = model(mixed_seqs)
            smooth_labels = mixed_labels * (1.0 - label_smooth) + 0.5 * label_smooth
            loss = criterion(logits, smooth_labels)
        else:
            _, logits = model(seqs)
            smooth_labels = labels * (1.0 - label_smooth) + 0.5 * label_smooth
            loss = criterion(logits, smooth_labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n          += 1
        pbar.set_postfix({"loss": f"{total_loss/n:.4f}"})

    return total_loss / max(n, 1)


# ── Single fold training ──────────────────────────────────────────────────────

def train_one_fold(fold_idx: int, train_ids, train_labels, train_subjects,
                   val_ids, val_labels, val_subjects,
                   test_ids, test_labels, test_subjects,
                   config: dict, seed: int, model_kwargs: dict,
                   device: torch.device, save_dir: str, reports_dir: str):
    """
    Train one fold for one seed.
    Returns (epoch_history, best_val_metrics, test_metrics, scaler, model_path).
    """
    tc           = config["training"]
    features_dir = os.path.join(config["data"]["processed_dir"], "features")

    print(f"\n{'='*60}")
    print(f"  Fold {fold_idx}  |  seed={seed}")
    print(f"  Train: {len(set(train_subjects))} subj ({len(train_ids)} clips)")
    print(f"  Val  : {len(set(val_subjects))} subj ({len(val_ids)} clips)")
    print(f"{'='*60}")

    train_loader, val_loader = create_dataloaders(
        train_ids, train_labels, train_subjects,
        val_ids, val_labels,
        features_dir,
        batch_size=tc["batch_size"],
        num_workers=tc["num_workers"],
        clips_per_subject=tc.get("clips_per_subject", 1),
    )

    model     = ASDMotionModel(config, **model_kwargs).to(device)
    criterion = nn.BCEWithLogitsLoss()

    opt_name = tc.get("optimizer", "sgd").lower()
    if opt_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=tc.get("lr", 0.005),
            momentum=0.9,
            weight_decay=tc.get("weight_decay", 1e-3),
            nesterov=True,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tc["epochs"], eta_min=1e-5
        )
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=tc.get("lr", 1e-4),
            weight_decay=tc.get("weight_decay", 0.01),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tc["epochs"], eta_min=1e-6
        )

    best_score   = -float("inf")
    best_epoch   = 0
    best_metrics = {}
    best_cm      = None
    best_roc     = None
    patience_ctr = 0
    history      = []
    model_path   = os.path.join(save_dir, f"fold{fold_idx}_seed{seed}.pt")

    for epoch in range(1, tc["epochs"] + 1):
        t0 = time.time()

        if hasattr(train_loader.dataset, "_resample"):
            train_loader.dataset._resample()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer,
            device, epoch, fold_idx, tc["epochs"],
        )
        scheduler.step()

        # Val inference
        val_logits, _, val_labels_raw, _ = run_inference(
            model, val_loader, criterion, device,
            desc=f"F{fold_idx} E{epoch} val",
        )
        val_subjects_arr = val_subjects   # list aligned with val_ids

        # Subject-level aggregation (no scaler yet — use raw sigmoid)
        val_preds, val_probs, val_labels_subj, val_logits_subj = subject_level_eval(
            val_logits, val_labels_raw, val_subjects_arr, scaler=None,
        )

        # Subject-level loss for model selection
        val_logits_t = torch.from_numpy(val_logits_subj).float().to(device)
        val_labels_t = torch.from_numpy(val_labels_subj).float().to(device)
        val_loss = nn.BCEWithLogitsLoss()(val_logits_t, val_labels_t).item()

        metrics = compute_all_metrics(val_labels_subj, val_preds, val_probs)
        metrics["train_loss"] = train_loss
        metrics["val_loss"]   = val_loss
        history.append(metrics)

        # Checkpoint selection strictly based on combination of tl + vl + ece + auc
        prev_vl = history[-2]["val_loss"] if len(history) >= 2 else val_loss
        prev_tl = history[-2]["train_loss"] if len(history) >= 2 else train_loss
        smooth_vl = 0.6 * val_loss + 0.4 * prev_vl
        smooth_tl = 0.6 * train_loss + 0.4 * prev_tl
        val_auc = metrics["auc"] if not np.isnan(metrics.get("auc", float("nan"))) else 0.5
        val_ece = metrics.get("ece", 0.1)

        # 4-component holistic score (maximize AUC, minimize val_loss, train_loss, ECE)
        gap = max(0.0, smooth_vl - smooth_tl)
        combined = val_auc - 0.5 * smooth_vl - 0.4 * val_ece - 0.3 * gap
        metrics["combined_score"] = combined

        is_collapsed = (
            metrics["sensitivity"] <= 0.05 or
            metrics["specificity"] <= 0.05 or
            np.isnan(val_loss)
        )

        elapsed = time.time() - t0
        print(f"  E{epoch:>3}/{tc['epochs']} ({elapsed:.0f}s) "
              f"tl={train_loss:.4f} vl={val_loss:.4f} "
              f"acc={metrics['accuracy']:.3f} "
              f"auc={val_auc:.3f} "
              f"ece={val_ece:.3f} "
              f"sens={metrics['sensitivity']:.3f} "
              f"spec={metrics['specificity']:.3f} "
              f"score={combined:.3f}", end="")

        if combined > best_score and not is_collapsed:
            best_score   = combined
            best_epoch   = epoch - 1
            best_metrics = metrics.copy()
            best_cm      = compute_confusion_matrix(val_labels_subj, val_preds)
            try:
                fpr, tpr, _ = compute_roc_curve(val_labels_subj, val_probs)
                best_roc    = (fpr, tpr)
            except Exception:
                best_roc = None
            torch.save({
                "epoch":      epoch,
                "state_dict": model.state_dict(),
                "metrics":    best_metrics,
                "config":     config,
            }, model_path)
            print(" *BEST*")
            patience_ctr = 0
        else:
            print("")
            patience_ctr += 1

        if patience_ctr >= tc["early_stopping_patience"]:
            print(f"\n  >> Early stopping at epoch {epoch} (patience={tc['early_stopping_patience']})")
            break

    # ── Platt Calibration ─────────────────────────────────────────────────────
    if not os.path.isfile(model_path):
        # No best saved (all epochs collapsed) — save last
        torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                    "metrics": {}, "config": config}, model_path)

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    val_logits_calib, _, val_labels_calib, _ = run_inference(
        model, val_loader, criterion, device, desc=f"F{fold_idx} calib",
    )
    # Subject-level logits for calibration
    _, _, val_labels_subj_c, val_logits_subj_c = subject_level_eval(
        val_logits_calib, val_labels_calib, val_subjects, scaler=None,
    )
    scaler = PlattScaler()
    scaler.fit(val_logits_subj_c, val_labels_subj_c,
               lr=config["calibration"]["lr"],
               max_iter=config["calibration"]["max_iter"])

    # Persist scaler
    ckpt["scaler"]    = pickle.dumps(scaler)
    ckpt["threshold"] = 0.5
    torch.save(ckpt, model_path)

    # Re-evaluate validation metrics WITH the fitted PlattScaler
    val_preds_cal, val_probs_cal, val_labels_subj_cal, _ = subject_level_eval(
        val_logits_calib, val_labels_calib, val_subjects, scaler=scaler,
    )
    val_metrics_cal = compute_all_metrics(val_labels_subj_cal, val_preds_cal, val_probs_cal)
    val_metrics_cal["train_loss"] = best_metrics.get("train_loss", 0.0)
    val_metrics_cal["val_loss"]   = best_metrics.get("val_loss", 0.0)
    val_cm_cal = compute_confusion_matrix(val_labels_subj_cal, val_preds_cal)
    try:
        v_fpr, v_tpr, _ = compute_roc_curve(val_labels_subj_cal, val_probs_cal)
        val_roc_cal = (v_fpr, v_tpr)
    except Exception:
        val_roc_cal = None

    print(f"\n  Fold {fold_idx} CALIB VAL: "
          f"acc={val_metrics_cal['accuracy']:.3f} "
          f"sens={val_metrics_cal['sensitivity']:.3f} "
          f"spec={val_metrics_cal['specificity']:.3f} "
          f"auc={val_metrics_cal['auc']:.3f} "
          f"ece={val_metrics_cal['ece']:.4f}")

    # ── Full-Cohort Attention Extraction & Persistence ─────────────────────────
    val_ds  = ASDMotionDataset(val_ids, val_labels, features_dir, augment=False)
    test_ds = ASDMotionDataset(test_ids, test_labels, features_dir, augment=False)
    
    val_attn_records = []
    test_attn_records = []
    try:
        from interpretability import extract_all_attention, save_fold_attention
        val_attn_records = extract_all_attention(model, val_ds, device, scaler=scaler)
        test_attn_records = extract_all_attention(model, test_ds, device, scaler=scaler)
        cur_model_id = config.get("_current_model_id", "A1")
        save_fold_attention(
            val_records=val_attn_records,
            test_records=test_attn_records,
            model_id=cur_model_id,
            fold_idx=fold_idx,
            seed=seed,
            results_dir=config["output"].get("results_dir", "results"),
            model_config=config.get("model", {}),
        )
    except Exception as e:
        err_log_dir = os.path.join(config["output"].get("results_dir", "results"), "attn")
        os.makedirs(err_log_dir, exist_ok=True)
        with open(os.path.join(err_log_dir, "extraction_errors.log"), "a", encoding="utf-8") as f_err:
            f_err.write(f"Fold {fold_idx}, Seed {seed}: {e}\n")
        print(f"  [WARN] Attention extraction failed: {e}")

    # ── Fold PDF report (TRAIN side — val curves, val CM/ROC, cohort attention) ──
    # Test evaluation
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=tc["batch_size"], shuffle=False,
        num_workers=tc["num_workers"], pin_memory=False,
    )
    test_logits, _, test_labels_raw, _ = run_inference(
        model, test_loader, criterion, device, desc=f"F{fold_idx} test",
    )
    test_preds, test_probs, test_labels_subj, _ = subject_level_eval(
        test_logits, test_labels_raw, test_subjects, scaler=scaler,
    )
    test_metrics = compute_all_metrics(test_labels_subj, test_preds, test_probs)

    # Test CM + ROC
    test_cm = compute_confusion_matrix(test_labels_subj, test_preds)
    try:
        t_fpr, t_tpr, _ = compute_roc_curve(test_labels_subj, test_probs)
        test_roc = (t_fpr, t_tpr)
    except Exception:
        test_roc = None

    print(f"  Fold {fold_idx} TEST:      "
          f"acc={test_metrics['accuracy']:.3f} "
          f"sens={test_metrics['sensitivity']:.3f} "
          f"spec={test_metrics['specificity']:.3f} "
          f"auc={test_metrics['auc']:.3f} "
          f"ece={test_metrics['ece']:.4f}\n")

    # Generate both PDFs: fold_N_train_report.pdf + fold_N_test_report.pdf
    generate_fold_reports(
        fold_idx=fold_idx,
        output_dir=reports_dir,
        # Train/val side (calibrated, full-cohort attention)
        epoch_history=history,
        best_epoch=best_epoch,
        val_metrics=val_metrics_cal,
        val_cm=val_cm_cal,
        val_roc=val_roc_cal,
        attention_data=val_attn_records,
        # Test side (calibrated, full-cohort attention)
        test_metrics=test_metrics,
        test_cm=test_cm,
        test_roc=test_roc,
    )

    return history, val_metrics_cal, test_metrics, scaler, model_path


# ── CV runner ─────────────────────────────────────────────────────────────────

def run_cv(config: dict, seed: int, model_kwargs: dict,
           model_id: str, splits: dict, device: torch.device):
    """
    Run 3-fold CV for one seed.
    Returns (fold_val_metrics, fold_test_metrics).
    """
    set_seed(seed)
    config["_current_model_id"] = model_id
    n_folds    = config["training"]["n_folds"]
    save_dir   = os.path.join(config["output"]["models_dir"], model_id)
    rep_dir    = os.path.join(config["output"]["reports_dir"], model_id)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(rep_dir,  exist_ok=True)

    fold_val, fold_test = [], []

    for fold_idx in range(n_folds):
        (train_ids, train_labels, train_subjects,
         val_ids,   val_labels,   val_subjects,
         test_ids,  test_labels,  test_subjects) = resolve_clip_lists(
             splits, fold_idx, config,
         )

        _, best_val, test_m, _, _ = train_one_fold(
            fold_idx + 1,
            train_ids, train_labels, train_subjects,
            val_ids,   val_labels,   val_subjects,
            test_ids,  test_labels,  test_subjects,
            config, seed, model_kwargs, device, save_dir, rep_dir,
        )
        fold_val.append(best_val)
        fold_test.append(test_m)

    return fold_val, fold_test


# ── 20-seed outer loop ────────────────────────────────────────────────────────

def run_all_seeds(config: dict, n_seeds: int, model_kwargs: dict,
                  model_id: str, splits: dict, device: torch.device):
    """
    Run n_seeds independent seeds of the full 3-fold CV.
    Returns aggregated metrics dict {metric: {mean, std}}.
    """
    seed_base     = config["training"]["seed"]
    all_val_met   = []
    all_test_met  = []

    for i in range(n_seeds):
        seed = seed_base + i
        print(f"\n{'#'*60}")
        print(f"  {model_id}  — Seed {i+1}/{n_seeds}  (seed={seed})")
        print(f"{'#'*60}")

        fold_val, fold_test = run_cv(
            config, seed, model_kwargs, model_id, splits, device,
        )
        # Use the mean across folds for this seed
        seed_val_agg  = {
            k: float(np.mean([m[k] for m in fold_val if k in m and not np.isnan(m[k])]))
            for k in ["accuracy", "auc", "f1", "sensitivity", "specificity", "ece", "threshold"]
        }
        seed_test_agg = {
            k: float(np.mean([m[k] for m in fold_test if k in m and not np.isnan(m[k])]))
            for k in ["accuracy", "auc", "f1", "sensitivity", "specificity", "ece", "threshold"]
        }
        all_val_met.append(seed_val_agg)
        all_test_met.append(seed_test_agg)

    # Generate population explainability report (including Page 4 kinematic attributions)
    try:
        from interpretability import generate_explainability_report
        res_dir = config["output"].get("results_dir", "results")
        generate_explainability_report(
            attn_dir=os.path.join(res_dir, "attn"),
            output_dir=res_dir,
            model_id=model_id,
            config=config,
            device=device,
        )
    except Exception as e:
        print(f"  [WARN] Explainability report generation skipped: {e}")

    # ── Persist raw per-seed metrics for downstream paired statistical tests ──
    res_dir = config["output"].get("results_dir", "results")
    os.makedirs(res_dir, exist_ok=True)
    per_seed_path = os.path.join(res_dir, f"{model_id}_per_seed.json")
    with open(per_seed_path, "w") as _f:
        json.dump({"val": all_val_met, "test": all_test_met}, _f, indent=2)
    print(f"  [INFO] Per-seed metrics → {per_seed_path}")

    return {
        "val":  aggregate_seed_metrics(all_val_met),
        "test": aggregate_seed_metrics(all_test_met),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PACE-ASD training")
    parser.add_argument("--config",   default="configs/config.yaml")
    parser.add_argument("--model_id", default="A1",
                        help="A1 | A2 | A3 | A4 (ablation variant)")
    parser.add_argument("--seed",     type=int, default=0,
                        help="Single seed to run (for sanity checks)")
    parser.add_argument("--fold",     type=int, default=None,
                        help="Run only this fold (0-indexed); default: all folds")
    parser.add_argument("--all_seeds", action="store_true",
                        help="Run all n_seeds from config")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # Map model_id to model_kwargs
    model_variants = {
        "A1": {"use_gate": True,  "use_transformer": True},
        "A2": {"use_gate": False, "use_transformer": True},
        "A3": {"use_gate": True,  "use_transformer": True},   # block_size/top_m in config
        "A4": {"use_gate": True,  "use_transformer": False},
    }
    if args.model_id not in model_variants:
        print(f"  Unknown model_id '{args.model_id}'. Use ablation.py for A5.")
        sys.exit(1)

    model_kwargs = model_variants[args.model_id]

    # A3: override block params
    if args.model_id == "A3":
        config["model"]["event_block_size"] = 1
        config["model"]["event_top_m"]      = 120

    splits = freeze_splits(config)

    if args.all_seeds:
        n_seeds = config["training"]["n_seeds"]
        results = run_all_seeds(config, n_seeds, model_kwargs,
                                args.model_id, splits, device)
        print(f"\n{'#'*60}")
        print(f"  {args.model_id} — ALL SEEDS SUMMARY")
        for split_name, agg in results.items():
            print(f"\n  [{split_name.upper()}]")
            print(f"  {'Metric':<15} {'Mean':>8} {'SD':>8}")
            print(f"  {'-'*33}")
            for k, v in agg.items():
                print(f"  {k:<15} {v['mean']:>8.4f} {v['std']:>8.4f}")
    else:
        # Single seed, single fold (or all folds)
        set_seed(args.seed)
        save_dir = os.path.join(config["output"]["models_dir"], args.model_id)
        rep_dir  = os.path.join(config["output"]["reports_dir"], args.model_id)
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(rep_dir,  exist_ok=True)

        folds = (
            [args.fold] if args.fold is not None
            else list(range(config["training"]["n_folds"]))
        )

        config["_current_model_id"] = args.model_id
        for fi in folds:
            (train_ids, train_labels, train_subjects,
             val_ids,   val_labels,   val_subjects,
             test_ids,  test_labels,  test_subjects) = resolve_clip_lists(
                 splits, fi, config,
             )
            train_one_fold(
                fi + 1,
                train_ids, train_labels, train_subjects,
                val_ids,   val_labels,   val_subjects,
                test_ids,  test_labels,  test_subjects,
                config, args.seed, model_kwargs, device, save_dir, rep_dir,
            )

        # Generate explainability report (including Page 4 kinematic attributions)
        try:
            from interpretability import generate_explainability_report
            res_dir = config["output"].get("results_dir", "results")
            generate_explainability_report(
                attn_dir=os.path.join(res_dir, "attn"),
                output_dir=res_dir,
                model_id=args.model_id,
                config=config,
                device=device,
            )
        except Exception as e:
            print(f"  [WARN] Explainability report generation skipped: {e}")


if __name__ == "__main__":
    main()
