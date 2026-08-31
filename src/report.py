"""
PACE-ASD — PDF Report Generation

Per-fold, two PDFs are generated:
  fold_{N}_train_report.pdf  — epoch curves, best-val CM/ROC, attention profile
  fold_{N}_test_report.pdf   — test-set CM, ROC, summary table

generate_fold_pdf(...)        — single split PDF (train or test)
generate_fold_reports(...)    — convenience: calls generate_fold_pdf twice
generate_ablation_table(...)  — mean±SD table across all arms for paper
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ── Style ─────────────────────────────────────────────────────────────────────

COLORS = {
    "accuracy":    "#2196F3",
    "auc":         "#4CAF50",
    "f1":          "#FF9800",
    "sensitivity": "#E91E63",
    "specificity": "#9C27B0",
    "ece":         "#795548",
    "train_loss":  "#F44336",
    "val_loss":    "#2196F3",
}

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        10,
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "figure.facecolor": "white",
    "axes.facecolor":   "#FAFAFA",
    "axes.grid":        True,
    "grid.alpha":       0.3,
})

_METRIC_CFG = [
    ("accuracy",    "Accuracy",    COLORS["accuracy"]),
    ("auc",         "AUC",         COLORS["auc"]),
    ("f1",          "F1 Score",    COLORS["f1"]),
    ("sensitivity", "Sensitivity", COLORS["sensitivity"]),
    ("specificity", "Specificity", COLORS["specificity"]),
    ("ece",         "ECE",         COLORS["ece"]),
    ("threshold",   "Threshold",   "#607D8B"),
]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _page_summary(pdf, title: str, metrics: dict,
                  extra_rows: list = None, split_label: str = ""):
    """Page 1: summary table."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.95)

    rows = [["Metric", split_label or "Value"]]
    if extra_rows:
        rows.extend(extra_rows)
    for key, label, _ in _METRIC_CFG:
        val = metrics.get(key, float("nan"))
        rows.append([label, f"{val:.4f}"])
    if "val_loss" in metrics and split_label == "Best Epoch (Val)":
        rows.append(["Val Loss", f"{metrics['val_loss']:.4f}"])

    table = ax.table(
        cellText=rows[1:], colLabels=rows[0],
        cellLoc="center", loc="center", colWidths=[0.35, 0.35],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.8)
    for j in range(2):
        table[0, j].set_facecolor("#1976D2")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows)):
        bg = "#E3F2FD" if i % 2 == 0 else "white"
        for j in range(2):
            table[i, j].set_facecolor(bg)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_epoch_curves(pdf, title: str, epoch_history: list, best_epoch: int):
    """Page 2: per-epoch metric grid."""
    if not epoch_history:
        return
    epochs = list(range(1, len(epoch_history) + 1))

    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    fig.suptitle(f"{title} — Per-Epoch Metrics",
                 fontsize=15, fontweight="bold", y=0.98)
    plot_metrics = [(k, l, c) for (k, l, c) in _METRIC_CFG if k != "threshold"]
    for idx, (key, lbl, color) in enumerate(plot_metrics):
        ax   = axes[idx // 2, idx % 2]
        vals = [h.get(key, float("nan")) for h in epoch_history]
        ax.plot(epochs, vals, color=color, linewidth=2, marker="o", markersize=3)
        ax.axvline(x=best_epoch + 1, color="gray", linestyle="--", alpha=0.5,
                   label=f"best={best_epoch+1}")
        ax.set_title(lbl, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_xlim(0.5, len(epochs) + 0.5)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig)
    plt.close(fig)


def _page_loss_curves(pdf, title: str, epoch_history: list, best_epoch: int):
    """Page 3: train/val loss curves."""
    if not epoch_history:
        return
    epochs = list(range(1, len(epoch_history) + 1))
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f"{title} — Loss Curves", fontsize=15, fontweight="bold")
    ax.plot(epochs, [h.get("train_loss", float("nan")) for h in epoch_history],
            color=COLORS["train_loss"], linewidth=2, label="Train Loss")
    ax.plot(epochs, [h.get("val_loss",   float("nan")) for h in epoch_history],
            color=COLORS["val_loss"],   linewidth=2, label="Val Loss")
    ax.axvline(x=best_epoch + 1, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.set_xlim(0.5, len(epochs) + 0.5)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_confusion_matrix(pdf, title: str, cm):
    """One page: confusion matrix heatmap."""
    if cm is None:
        return
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle(f"{title} — Confusion Matrix", fontsize=15, fontweight="bold")
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1]);  ax.set_xticklabels(["Non-ASD (0)", "ASD (1)"], fontsize=12)
    ax.set_yticks([0, 1]);  ax.set_yticklabels(["Non-ASD (0)", "ASD (1)"], fontsize=12)
    ax.set_xlabel("Predicted");  ax.set_ylabel("True")
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=20, fontweight="bold",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_roc(pdf, title: str, roc_data, auc_val):
    """One page: ROC curve."""
    if roc_data is None:
        return
    fpr, tpr = roc_data
    fig, ax  = plt.subplots(figsize=(8, 7))
    fig.suptitle(f"{title} — ROC Curve", fontsize=15, fontweight="bold")
    ax.plot(fpr, tpr, color="#1976D2", linewidth=2, label=f"AUC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=12)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_attention(pdf, title: str, attention_data):
    """One page: temporal attention profile (cohort-level or single sample)."""
    if attention_data is None:
        return

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x_frames = np.arange(300)

    if isinstance(attention_data, list) and len(attention_data) > 0:
        # Full cohort records list
        asd_profs = [r["frame_profile"] for r in attention_data if r.get("label") == 1 and r.get("frame_profile") is not None and r["frame_profile"].sum() > 0]
        td_profs  = [r["frame_profile"] for r in attention_data if r.get("label") == 0 and r.get("frame_profile") is not None and r["frame_profile"].sum() > 0]

        fig.suptitle(f"{title} — Class-Averaged Temporal Attention Profiles",
                     fontsize=14, fontweight="bold")

        if asd_profs:
            asd_mat = np.stack(asd_profs, axis=0)
            asd_mean = asd_mat.mean(axis=0)
            asd_std  = asd_mat.std(axis=0)
            ax.plot(x_frames, asd_mean, color="#E91E63", linewidth=2,
                    label=f"ASD Class Mean (N={len(asd_profs)})")
            ax.fill_between(x_frames, np.maximum(0, asd_mean - asd_std), asd_mean + asd_std,
                            color="#E91E63", alpha=0.25, label=r"ASD $\pm 1$ SD Band")

        if td_profs:
            td_mat = np.stack(td_profs, axis=0)
            td_mean = td_mat.mean(axis=0)
            td_std  = td_mat.std(axis=0)
            ax.plot(x_frames, td_mean, color="#2196F3", linewidth=2,
                    label=f"TD Class Mean (N={len(td_profs)})")
            ax.fill_between(x_frames, np.maximum(0, td_mean - td_std), td_mean + td_std,
                            color="#2196F3", alpha=0.25, label=r"TD $\pm 1$ SD Band")

        ticks = np.arange(0, 301, 30)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"F{t} ({t//30}s)" for t in ticks], rotation=30, fontsize=8)
        ax.set_xlabel("Timeline (Frames / Seconds @ 30fps)", fontweight="bold")
        ax.set_ylabel("Normalized Attention Salience", fontweight="bold")
        ax.legend(loc="upper right", frameon=True)
        ax.set_xlim(-1, 301)
        ax.set_ylim(0, 1.15)
        ax.grid(True, alpha=0.3)

    elif isinstance(attention_data, dict):
        # Single sample dict fallback
        weights  = attention_data.get("weights")
        indices  = attention_data.get("indices")
        lbl      = attention_data.get("label", 0)
        decision = weights.mean(axis=0) if weights is not None else np.zeros(len(indices))
        k        = len(indices) if indices is not None else 0
        cls_str  = "ASD (1)" if lbl == 1 else "Non-ASD (0)"

        fig.suptitle(f"{title} — Temporal Attention Profile",
                     fontsize=14, fontweight="bold")
        x = np.arange(k)
        ax.fill_between(x, decision, color="#E91E63", alpha=0.3)
        ax.plot(x, decision, color="#E91E63", linewidth=2)
        ticks = np.arange(0, k, max(1, k // 10))
        ax.set_xticks(ticks)
        if indices is not None:
            ax.set_xticklabels([f"F{indices[t]}" for t in ticks], rotation=45, fontsize=8)
        ax.set_xlabel("Frame (Time)")
        ax.set_ylabel("Attention Weight")
        ax.set_title(f"True label: {cls_str}", fontsize=10, style="italic")
        ax.set_xlim(-0.5, k - 0.5)
        ax.set_ylim(0, max(0.02, decision.max() * 1.1))
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ── Public API ────────────────────────────────────────────────────────────────

def generate_fold_pdf(fold_idx, split: str, metrics: dict,
                      output_dir: str,
                      epoch_history: list = None,
                      best_epoch: int = 0,
                      cm=None,
                      roc_data=None,
                      attention_data: dict = None) -> str:
    """
    Generate one PDF for one fold / one split.

    Args:
        fold_idx     : int (1-indexed) or string e.g. 'test'
        split        : 'train' | 'val' | 'test'
                       Used in filename and page titles.
        metrics      : dict from compute_all_metrics
        output_dir   : directory to write PDF
        epoch_history: list of per-epoch metric dicts (train-side report only)
        best_epoch   : 0-indexed best epoch (train-side report only)
        cm           : (2,2) confusion matrix or None
        roc_data     : (fpr, tpr) tuple or None
        attention_data: dict {weights, indices, label} or None

    Returns:
        absolute path to the written PDF
    """
    os.makedirs(output_dir, exist_ok=True)

    fold_str = str(fold_idx)
    title    = f"PACE-ASD — Fold {fold_str} [{split.upper()}]"
    pdf_name = f"fold_{fold_str}_{split}_report.pdf"
    pdf_path = os.path.join(output_dir, pdf_name)

    with PdfPages(pdf_path) as pdf:
        # Page 1 — Summary table
        extra = []
        if split in ("train", "val") and epoch_history:
            extra = [
                ["Best Epoch",   str(best_epoch + 1)],
                ["Total Epochs", str(len(epoch_history))],
            ]
        split_col = "Best Epoch (Val)" if split in ("train", "val") else "Test Value"
        _page_summary(pdf, title, metrics, extra_rows=extra, split_label=split_col)

        # Pages 2-3 — Epoch curves (training-side report only)
        if split in ("train", "val") and epoch_history:
            _page_epoch_curves(pdf, title, epoch_history, best_epoch)
            _page_loss_curves(pdf, title, epoch_history, best_epoch)

        # Confusion matrix
        _page_confusion_matrix(pdf, title, cm)

        # ROC curve
        _page_roc(pdf, title, roc_data, metrics.get("auc", float("nan")))

        # Attention profile (training-side only)
        if split in ("train", "val"):
            _page_attention(pdf, title, attention_data)

    print(f"  [PDF] {pdf_path}")
    return pdf_path


def generate_fold_reports(fold_idx: int, output_dir: str,
                           # Train/val side
                           epoch_history: list, best_epoch: int,
                           val_metrics: dict, val_cm, val_roc,
                           attention_data: dict,
                           # Test side
                           test_metrics: dict, test_cm, test_roc) -> tuple:
    """
    Convenience wrapper: generate both PDFs for one fold.

    Returns (train_pdf_path, test_pdf_path).
    """
    train_pdf = generate_fold_pdf(
        fold_idx=fold_idx,
        split="train",
        metrics=val_metrics,
        output_dir=output_dir,
        epoch_history=epoch_history,
        best_epoch=best_epoch,
        cm=val_cm,
        roc_data=val_roc,
        attention_data=attention_data,
    )
    test_pdf = generate_fold_pdf(
        fold_idx=fold_idx,
        split="test",
        metrics=test_metrics,
        output_dir=output_dir,
        epoch_history=None,
        cm=test_cm,
        roc_data=test_roc,
    )
    return train_pdf, test_pdf


# ── Legacy shim (keeps old call sites working) ────────────────────────────────

def generate_fold_report(fold_idx, epoch_history, best_epoch, best_metrics,
                         cm, roc_data, output_dir, attention_data=None):
    """
    Backward-compatible wrapper.
    Generates fold_{N}_train_report.pdf only (no test PDF).
    New code should call generate_fold_reports() or generate_fold_pdf() directly.
    """
    return generate_fold_pdf(
        fold_idx=fold_idx, split="train",
        metrics=best_metrics,
        output_dir=output_dir,
        epoch_history=epoch_history,
        best_epoch=best_epoch,
        cm=cm, roc_data=roc_data,
        attention_data=attention_data,
    )


# ── Ablation summary table ────────────────────────────────────────────────────

def generate_ablation_table(results: dict, output_dir: str) -> str:
    """
    Generate a paper-ready ablation table PDF.

    Args:
        results : {model_id: {"val": {...}, "test": {...}}}
                  Each inner dict: {metric: {"mean": float, "std": float}}
        output_dir: directory to save ablation_table.pdf

    Returns: path to the PDF
    """
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, "ablation_table.pdf")

    model_ids     = list(results.keys())
    metric_keys   = ["accuracy", "auc", "f1", "sensitivity", "specificity", "ece", "threshold"]
    metric_labels = ["Accuracy", "AUC", "F1", "Sensitivity", "Specificity", "ECE", "Threshold"]

    with PdfPages(pdf_path) as pdf:
        for split in ("val", "test"):
            split_label = "Validation" if split == "val" else "Held-Out Test"
            col_labels  = ["Model"] + metric_labels
            rows        = []

            for mid in model_ids:
                agg = results[mid].get(split, results[mid])  # fallback for old format
                row = [mid]
                for key in metric_keys:
                    m   = agg.get(key, {})
                    mn  = m.get("mean", float("nan"))
                    std = m.get("std",  float("nan"))
                    row.append(f"{mn:.3f} ± {std:.3f}")
                rows.append(row)

            fig, ax = plt.subplots(figsize=(15, max(4, 1.2 * len(rows) + 2)))
            ax.axis("off")
            fig.suptitle(
                f"PACE-ASD — Ablation Results [{split_label}] "
                f"(mean ± SD, 20 seeds × 3-fold CV)",
                fontsize=13, fontweight="bold", y=0.97,
            )
            table = ax.table(
                cellText=rows, colLabels=col_labels,
                cellLoc="center", loc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2.0)
            for j in range(len(col_labels)):
                table[0, j].set_facecolor("#1565C0")
                table[0, j].set_text_props(color="white", fontweight="bold")
            for i in range(1, len(rows) + 1):
                bg = "#E8F5E9" if i % 2 == 0 else "white"
                for j in range(len(col_labels)):
                    table[i, j].set_facecolor(bg)
            # Highlight A1
            if rows:
                for j in range(len(col_labels)):
                    table[1, j].set_facecolor("#FFF9C4")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"  [PDF] Ablation table saved: {pdf_path}")
    return pdf_path
