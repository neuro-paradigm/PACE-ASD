"""
ASDMotion - Per-Fold & Test PDF Report Generation

Generates a multi-page PDF report after each fold containing:
  - Summary table (best epoch, final metrics)
  - Per-epoch metric line plots (Accuracy, AUC, F1, Sensitivity, Specificity, ECE)
  - Training & validation loss curves
  - Confusion matrix heatmap
  - ROC curve
  - Temporal Decision Attention Profile (Explainability)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# -- Consistent styling --
COLORS = {
    'accuracy': '#2196F3',
    'auc': '#4CAF50',
    'f1': '#FF9800',
    'sensitivity': '#E91E63',
    'specificity': '#9C27B0',
    'ece': '#795548',
    'train_loss': '#F44336',
    'val_loss': '#2196F3',
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.facecolor': 'white',
    'axes.facecolor': '#FAFAFA',
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def generate_fold_report(fold_idx, epoch_history, best_epoch,
                         best_metrics, cm, roc_data, output_dir, attention_data=None):
    """
    Generate a PDF report for a single fold or the held-out test set.

    Args:
        fold_idx: Fold number (1-indexed) or 'test' for test set report
        epoch_history: list of dicts, one per epoch (empty for test)
        best_epoch: int, best epoch index (0-indexed)
        best_metrics: dict with best epoch metrics
        cm: (2,2) confusion matrix numpy array
        roc_data: tuple (fpr, tpr) for ROC curve
        output_dir: directory to save PDF
        attention_data: dict containing weights, indices, and label
    """
    os.makedirs(output_dir, exist_ok=True)
    is_test = (fold_idx == 'test')
    report_label = 'Held-Out Test Set (Ensemble)' if is_test else f'Fold {fold_idx}'
    pdf_name = 'test_report.pdf' if is_test else f'fold_{fold_idx}_report.pdf'
    pdf_path = os.path.join(output_dir, pdf_name)
    epochs = list(range(1, len(epoch_history) + 1))

    with PdfPages(pdf_path) as pdf:
        # -- Page 1: Summary Table --
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        fig.suptitle(f'ASDMotion - {report_label} Report',
                     fontsize=18, fontweight='bold', y=0.95)

        value_col_name = 'Test Value' if is_test else 'Best Epoch Value'
        table_data = [
            ['Metric', value_col_name],
        ]
        if not is_test:
            table_data.append(['Best Epoch', str(best_epoch + 1)])
            table_data.append(['Total Epochs', str(len(epoch_history))])

        table_data.extend([
            ['Accuracy', f"{best_metrics.get('accuracy', 0):.4f}"],
            ['AUC', f"{best_metrics.get('auc', 0):.4f}"],
            ['F1 Score', f"{best_metrics.get('f1', 0):.4f}"],
            ['Sensitivity', f"{best_metrics.get('sensitivity', 0):.4f}"],
            ['Specificity', f"{best_metrics.get('specificity', 0):.4f}"],
            ['ECE', f"{best_metrics.get('ece', 0):.4f}"],
        ])
        if not is_test:
            table_data.append(['Combined Score', f"{best_metrics.get('combined_score', 0):.4f}"])
            table_data.append(['Validation Loss', f"{best_metrics.get('val_loss', 0):.4f}"])

        table = ax.table(
            cellText=table_data[1:],
            colLabels=table_data[0],
            cellLoc='center', loc='center',
            colWidths=[0.35, 0.35]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.8)

        # Style header row
        for j in range(2):
            table[0, j].set_facecolor('#1976D2')
            table[0, j].set_text_props(color='white', fontweight='bold')
        # Alternate row colors
        for i in range(1, len(table_data)):
            color = '#E3F2FD' if i % 2 == 0 else 'white'
            for j in range(2):
                table[i, j].set_facecolor(color)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # -- Pages 2-3: Epoch metrics & loss (only for fold reports) --
        if not is_test and len(epoch_history) > 0:
            # Page 2: Metric Curves
            fig, axes = plt.subplots(3, 2, figsize=(12, 14))
            fig.suptitle(f'{report_label} - Per-Epoch Metrics',
                         fontsize=16, fontweight='bold', y=0.98)

            metric_configs = [
                ('accuracy', 'Accuracy', COLORS['accuracy']),
                ('auc', 'AUC (ROC)', COLORS['auc']),
                ('f1', 'F1 Score', COLORS['f1']),
                ('sensitivity', 'Sensitivity (Recall)', COLORS['sensitivity']),
                ('specificity', 'Specificity', COLORS['specificity']),
                ('ece', 'ECE (Expected Calibration Error)', COLORS['ece']),
            ]

            for idx, (key, title, color) in enumerate(metric_configs):
                ax = axes[idx // 2, idx % 2]
                values = [h[key] for h in epoch_history]
                ax.plot(epochs, values, color=color, linewidth=2, marker='o',
                        markersize=3, label=title)
                ax.axvline(x=best_epoch + 1, color='gray', linestyle='--',
                           alpha=0.5, label=f'Best epoch ({best_epoch + 1})')
                ax.set_title(title, fontweight='bold')
                ax.set_xlabel('Epoch')
                ax.set_ylabel(title)
                ax.legend(fontsize=8)
                ax.set_xlim(0.5, len(epochs) + 0.5)

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig)
            plt.close(fig)

            # Page 3: Loss Curves
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.suptitle(f'{report_label} - Training & Validation Loss',
                         fontsize=16, fontweight='bold')

            train_losses = [h['train_loss'] for h in epoch_history]
            val_losses = [h['val_loss'] for h in epoch_history]

            ax.plot(epochs, train_losses, color=COLORS['train_loss'],
                    linewidth=2, label='Train Loss', marker='o', markersize=3)
            ax.plot(epochs, val_losses, color=COLORS['val_loss'],
                    linewidth=2, label='Val Loss', marker='s', markersize=3)
            ax.axvline(x=best_epoch + 1, color='gray', linestyle='--',
                       alpha=0.5, label=f'Best epoch ({best_epoch + 1})')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.set_xlim(0.5, len(epochs) + 0.5)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # -- Confusion Matrix (both fold and test) --
        if cm is not None:
            fig, ax = plt.subplots(figsize=(8, 7))
            cm_title = 'Confusion Matrix (Ensemble)' if is_test else 'Confusion Matrix (Best Epoch)'
            fig.suptitle(f'{report_label} - {cm_title}',
                         fontsize=16, fontweight='bold')

            im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            classes = ['Non-ASD (0)', 'ASD (1)']
            tick_marks = [0, 1]
            ax.set_xticks(tick_marks)
            ax.set_xticklabels(classes, fontsize=12)
            ax.set_yticks(tick_marks)
            ax.set_yticklabels(classes, fontsize=12)
            ax.set_xlabel('Predicted Label', fontsize=13)
            ax.set_ylabel('True Label', fontsize=13)

            # Annotate cells
            thresh = cm.max() / 2.0
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, format(cm[i, j], 'd'),
                            ha='center', va='center', fontsize=20,
                            fontweight='bold',
                            color='white' if cm[i, j] > thresh else 'black')

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # -- ROC Curve (both fold and test) --
        if roc_data is not None:
            fpr, tpr = roc_data
            fig, ax = plt.subplots(figsize=(8, 7))
            fig.suptitle(f'{report_label} - ROC Curve',
                         fontsize=16, fontweight='bold')

            auc_val = best_metrics.get('auc', 0)
            ax.plot(fpr, tpr, color='#1976D2', linewidth=2,
                    label=f'AUC = {auc_val:.4f}')
            ax.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic')
            ax.legend(loc='lower right', fontsize=12)
            ax.set_xlim(-0.01, 1.01)
            ax.set_ylim(-0.01, 1.01)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # -- Explainability Page: Temporal Decision Attention Profile (1D) --
        if attention_data is not None:
            weights = attention_data['weights']  # (K, K)
            indices = attention_data['indices']  # (K,)
            label = attention_data['label']
            class_str = 'ASD (1)' if label == 1 else 'Non-ASD (0)'

            # Extract attention weights averaged across all queries
            # Under bidirectional mean pooling, this represents each frame's true contribution to the global decision.
            decision_attention = weights.mean(axis=0)

            fig, ax = plt.subplots(figsize=(10, 5))
            fig.suptitle(f'{report_label} - Temporal Decision Attention Profile (Explainability)',
                         fontsize=15, fontweight='bold')

            # Plot as a beautiful filled area/line chart
            x_vals = np.arange(len(indices))
            ax.fill_between(x_vals, decision_attention, color='#E91E63', alpha=0.3, label='Attention Weight')
            ax.plot(x_vals, decision_attention, color='#E91E63', linewidth=2)

            # Label every 10th index to avoid clutter
            k = len(indices)
            ticks = np.arange(0, k, max(1, k // 10))
            labels = [f"F{indices[t]}" for t in ticks]
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, rotation=45, fontsize=8)

            ax.set_xlabel('Kinematic Video Frames (Time ->)')
            ax.set_ylabel('Attention Weight / Decision Influence')
            ax.set_title(f"Classification Decision Influence Profile (True Label: {class_str})\n"
                         f"Peaks highlight exactly which frames the model queried to make its final diagnosis.",
                         fontsize=10, style='italic', pad=10)
            ax.grid(True, alpha=0.3)
            
            # Set limits to fit data
            ax.set_xlim(-0.5, k - 0.5)
            ax.set_ylim(0, max(0.02, decision_attention.max() * 1.1))

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"  [PDF] Report saved: {pdf_path}")
    return pdf_path
