"""
PACE-ASD — Metrics Computation (Protocol Section 7)

Functions:
  compute_all_metrics       — accuracy, AUC, F1, sensitivity, specificity, ECE
  compute_confidence_interval — Wilson score CI (used for Section 6 supplement)
  aggregate_seed_metrics    — mean ± SD across 20 seeds for paper table
  compute_confusion_matrix  — raw CM array
  compute_roc_curve         — (fpr, tpr, thresholds)
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    confusion_matrix, roc_curve,
)
from scipy.stats import norm


# ── Confidence intervals ──────────────────────────────────────────────────────

def compute_confidence_interval(p: float, n: int, confidence: float = 0.95):
    """
    Wilson score interval for a binomial proportion p estimated from n samples.

    Returns (lower, upper) clamped to [0, 1].
    Used for sensitivity in the Section 6 supplementary note (n=9).
    """
    if n == 0:
        return (0.0, 0.0)
    z = norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    denom = 1.0 + z ** 2 / n
    centre = p + z ** 2 / (2.0 * n)
    spread = np.sqrt((p * (1.0 - p) + z ** 2 / (4.0 * n)) / n)
    lo = (centre - z * spread) / denom
    hi = (centre + z * spread) / denom
    return float(np.clip(lo, 0.0, 1.0)), float(np.clip(hi, 0.0, 1.0))


# ── Core metric functions ─────────────────────────────────────────────────────

def compute_sensitivity(y_true, y_pred) -> float:
    """True Positive Rate = TP / (TP + FN)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def compute_specificity(y_true, y_pred) -> float:
    """True Negative Rate = TN / (TN + FP)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0


def compute_ece(y_true, y_prob, n_bins: int = 10) -> float:
    """
    Expected Calibration Error.
    Measures how well predicted probabilities match empirical outcomes.
    """
    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y_true)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = boundaries[i], boundaries[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1 else y_prob <= hi)
        n_bin = int(mask.sum())
        if n_bin == 0:
            continue
        bin_acc  = float(y_true[mask].mean())
        bin_conf = float(y_prob[mask].mean())
        ece += (n_bin / total) * abs(bin_acc - bin_conf)
    return ece


def compute_all_metrics(y_true, y_pred, y_prob, threshold: float = 0.5) -> dict:
    """
    Compute the full metric set for one evaluation.

    Args:
        y_true: (N,) int array — ground-truth binary labels
        y_pred: (N,) int array — predicted binary labels (after threshold)
        y_prob: (N,) float array — predicted probabilities
        threshold: float — decision threshold applied

    Returns:
        dict: accuracy, auc, f1, sensitivity, specificity, ece, threshold,
              accuracy_ci, sensitivity_ci, specificity_ci
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    m = {}
    m["accuracy"]    = float(accuracy_score(y_true, y_pred))
    m["f1"]          = float(f1_score(y_true, y_pred, zero_division=0))
    m["sensitivity"] = compute_sensitivity(y_true, y_pred)
    m["specificity"] = compute_specificity(y_true, y_pred)
    m["ece"]         = compute_ece(y_true, y_prob)
    m["threshold"]   = float(threshold)

    try:
        m["auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        m["auc"] = float("nan")

    n_total = len(y_true)
    n_pos   = int(y_true.sum())
    n_neg   = n_total - n_pos

    m["accuracy_ci"]    = compute_confidence_interval(m["accuracy"],    n_total)
    m["sensitivity_ci"] = compute_confidence_interval(m["sensitivity"], n_pos)
    m["specificity_ci"] = compute_confidence_interval(m["specificity"], n_neg)

    return m


def compute_supplement_sensitivity(y_true, y_pred, y_prob=None) -> dict:
    """
    Section 6 supplementary note metrics.
    Reports ONLY sensitivity + Wilson CI (group is ASD-only, so AUC/spec
    are not computable and must not be reported).

    Args:
        y_true: all ones (ASD-only group)
        y_pred: model predictions
        y_prob: not used for CI, kept for API consistency

    Returns:
        dict: sensitivity, sensitivity_ci
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n      = len(y_true)
    sens   = float(np.sum((y_true == 1) & (y_pred == 1))) / max(n, 1)
    return {
        "sensitivity":    sens,
        "sensitivity_ci": compute_confidence_interval(sens, n),
        "n":              n,
    }


# ── Seed aggregation ──────────────────────────────────────────────────────────

SCALAR_METRICS = ["accuracy", "auc", "f1", "sensitivity", "specificity", "ece", "threshold"]


def aggregate_seed_metrics(metric_dicts: list) -> dict:
    """
    Aggregate a list of per-seed metric dicts into mean ± SD.

    Args:
        metric_dicts: list of dicts from compute_all_metrics, one per seed

    Returns:
        dict: {metric: {"mean": float, "std": float}} for every scalar metric
    """
    agg = {}
    for key in SCALAR_METRICS:
        values = [d[key] for d in metric_dicts if key in d and not np.isnan(d[key])]
        if values:
            agg[key] = {
                "mean": float(np.mean(values)),
                "std":  float(np.std(values)),
            }
        else:
            agg[key] = {"mean": float("nan"), "std": float("nan")}
    return agg


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_confusion_matrix(y_true, y_pred) -> np.ndarray:
    """Return (2, 2) confusion matrix."""
    return confusion_matrix(np.asarray(y_true), np.asarray(y_pred), labels=[0, 1])


def compute_roc_curve(y_true, y_prob):
    """Return (fpr, tpr, thresholds) for ROC plotting."""
    return roc_curve(np.asarray(y_true), np.asarray(y_prob))


def find_optimal_threshold(y_true, y_prob, target_sensitivity: float = 0.90) -> float:
    """
    Find the threshold that achieves at least target_sensitivity
    while maximising specificity.
    Falls back to 0.5 if no threshold meets the target.
    """
    fpr, tpr, thresholds = roc_curve(np.asarray(y_true), np.asarray(y_prob))
    valid = np.where(tpr >= target_sensitivity)[0]
    if len(valid) == 0:
        return 0.5
    best_idx = valid[np.argmin(fpr[valid])]
    return float(thresholds[best_idx])
