"""
ASDMotion — Metrics Computation

Computes: Accuracy, AUC, F1, Sensitivity, Specificity, ECE, Confusion Matrix.
Also computes the combined score for model selection.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    confusion_matrix, roc_curve
)
from scipy.stats import norm

def compute_confidence_interval(metric_value, n, confidence=0.95):
    """
    Compute Wilson Score Interval for binomial proportions.
    Used for accuracy, sensitivity, specificity.
    """
    if n == 0:
        return (0.0, 0.0)
    z = norm.ppf(1 - (1 - confidence) / 2)
    p = metric_value
    
    denominator = 1 + z**2/n
    centre_adjusted_probability = p + z**2 / (2*n)
    adjusted_standard_deviation = np.sqrt((p*(1 - p) + z**2 / (4*n)) / n)
    
    lower_bound = (centre_adjusted_probability - z*adjusted_standard_deviation) / denominator
    upper_bound = (centre_adjusted_probability + z*adjusted_standard_deviation) / denominator
    
    return max(0.0, lower_bound), min(1.0, upper_bound)

def find_optimal_threshold(y_true, y_prob, target_sensitivity=0.90):
    """
    Find the threshold that achieves at least target_sensitivity
    while maximizing specificity.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    # Find indices where sensitivity (TPR) is >= target
    valid_idx = np.where(tpr >= target_sensitivity)[0]
    
    if len(valid_idx) == 0:
        return 0.5 # Fallback
    
    # Among those, pick the one with lowest FPR (highest specificity)
    best_idx = valid_idx[np.argmin(fpr[valid_idx])]
    return thresholds[best_idx]



def compute_sensitivity(y_true, y_pred):
    """True Positive Rate = TP / (TP + FN)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def compute_specificity(y_true, y_pred):
    """True Negative Rate = TN / (TN + FP)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def compute_ece(y_true, y_prob, n_bins=10):
    """
    Expected Calibration Error (ECE).
    Measures how well predicted probabilities match actual outcomes.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)

        bin_size = mask.sum()
        if bin_size == 0:
            continue

        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (bin_size / total) * abs(bin_acc - bin_conf)

    return ece


def compute_all_metrics(y_true, y_pred, y_prob):
    """
    Compute all evaluation metrics.

    Args:
        y_true: (N,) ground truth binary labels
        y_pred: (N,) predicted binary labels (thresholded)
        y_prob: (N,) predicted probabilities

    Returns:
        dict with accuracy, auc, f1, sensitivity, specificity, ece
    """
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['sensitivity'] = compute_sensitivity(y_true, y_pred)
    metrics['specificity'] = compute_specificity(y_true, y_pred)
    metrics['ece'] = compute_ece(y_true, y_prob)

    try:
        metrics['auc'] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics['auc'] = 0.0  # Only one class present

    # Compute CIs
    n_total = len(y_true)
    n_pos = sum(y_true)
    n_neg = n_total - n_pos

    metrics['accuracy_ci'] = compute_confidence_interval(metrics['accuracy'], n_total)
    metrics['sensitivity_ci'] = compute_confidence_interval(metrics['sensitivity'], n_pos)
    metrics['specificity_ci'] = compute_confidence_interval(metrics['specificity'], n_neg)

    return metrics


def compute_combined_score(val_loss, sensitivity, specificity, weights=None):
    """
    Combined score for model selection.
    Higher is better.

    Score = w_loss * (1 - normalized_loss) + w_sens * sensitivity + w_spec * specificity
    """
    if weights is None:
        weights = {'loss': 0.4, 'sensitivity': 0.3, 'specificity': 0.3}

    # Clamp loss contribution to [0, 1]
    loss_score = max(0.0, 1.0 - val_loss)

    combined = (
        weights['loss'] * loss_score +
        weights['sensitivity'] * sensitivity +
        weights['specificity'] * specificity
    )
    return combined


def compute_confusion_matrix(y_true, y_pred):
    """Return confusion matrix as numpy array."""
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def compute_roc_curve(y_true, y_prob):
    """Return (fpr, tpr, thresholds) for ROC curve plotting."""
    return roc_curve(y_true, y_prob)
