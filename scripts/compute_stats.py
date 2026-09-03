"""Compute summary statistics for ISWA manuscript tables."""
import json
import numpy as np
from scipy import stats
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.exists(os.path.join(ROOT_DIR, "results")):
    os.chdir(ROOT_DIR)

def load_test_metrics(path):
    with open(path) as f:
        d = json.load(f)
    return d['test']

def summarize(recs):
    keys = ['auc', 'accuracy', 'f1', 'sensitivity', 'specificity', 'ece']
    out = {}
    for k in keys:
        vals = [r[k] for r in recs]
        out[k] = {'mean': np.mean(vals), 'std': np.std(vals, ddof=1), 'vals': vals}
    return out

model_paths = {
    'A1': 'results/A1_per_seed.json',
    'A2': 'results/A2_per_seed.json',
    'A3': 'results/A3_per_seed.json',
    'A4': 'results/A4_per_seed.json',
    'MTC-Former': 'results/A5_mtcformer_per_seed.json',
    'LSTM': 'results/A5_lstm_per_seed.json',
    'Conv1D-BiLSTM': 'results/A5_conv1d_bilstm_per_seed.json',
    'KinCNN-LSTM': 'results/A5_kinematic_cnn_per_seed.json',
    'MSG3D': 'results/A5_msg3d_per_seed.json',
    'SkelFormer': 'results/A5_skelformer_per_seed.json',
    'STTS': 'results/A5_stts_per_seed.json',
    'MTT': 'results/A5_mtt_per_seed.json',
    'STAR': 'results/A5_star_per_seed.json',
    'LR': 'results/A5_lr_per_seed.json',
    'SVM': 'results/A5_svm_per_seed.json',
    'RF': 'results/A5_rf_per_seed.json',
    'XGBoost': 'results/A5_xgboost_per_seed.json',
}

summaries = {}
for name, path in model_paths.items():
    recs = load_test_metrics(path)
    summaries[name] = summarize(recs)

print("=" * 100)
print(f"{'Model':<20} | {'AUC':>12} | {'Accuracy':>12} | {'F1':>12} | {'Sensitivity':>13} | {'Specificity':>13} | {'ECE':>10}")
print("=" * 100)
for name, s in summaries.items():
    auc = s['auc']
    acc = s['accuracy']
    f1 = s['f1']
    sens = s['sensitivity']
    spec = s['specificity']
    ece = s['ece']
    print(f"{name:<20} | {auc['mean']:.4f}({auc['std']:.4f}) | {acc['mean']:.4f}({acc['std']:.4f}) | "
          f"{f1['mean']:.4f}({f1['std']:.4f}) | {sens['mean']:.4f}({sens['std']:.4f}) | "
          f"{spec['mean']:.4f}({spec['std']:.4f}) | {ece['mean']:.4f}({ece['std']:.4f})")

# --- Paired tests A1 vs MTC-Former ---
print("\n--- Paired Wilcoxon tests (A1 vs MTC-Former) ---")
a1 = summaries['A1']
mtc = summaries['MTC-Former']

for metric in ['auc', 'specificity', 'sensitivity', 'f1']:
    a1_vals = np.array(a1[metric]['vals'])
    mtc_vals = np.array(mtc[metric]['vals'])
    w, p = stats.wilcoxon(a1_vals, mtc_vals, alternative='two-sided')
    diff = a1[metric]['mean'] - mtc[metric]['mean']
    print(f"  {metric}: A1={a1[metric]['mean']:.4f}, MTC={mtc[metric]['mean']:.4f}, diff={diff:+.4f}, W={w:.0f}, p={p:.4f}")

# Specificity: one-sided A1 > MTC
a1_spec = np.array(a1['specificity']['vals'])
mtc_spec = np.array(mtc['specificity']['vals'])
w_os, p_os = stats.wilcoxon(a1_spec, mtc_spec, alternative='greater')
print(f"  specificity one-sided (A1>MTC): W={w_os:.0f}, p={p_os:.4f}")

# --- Holm-Bonferroni corrected AUC comparison A1 vs all baselines ---
print("\n--- A1 AUC vs baselines (two-sided Wilcoxon, raw p) ---")
a1_auc = np.array(a1['auc']['vals'])
baseline_names = ['MTC-Former', 'LSTM', 'Conv1D-BiLSTM', 'KinCNN-LSTM', 'MSG3D',
                  'SkelFormer', 'STTS', 'MTT', 'STAR', 'LR', 'SVM', 'RF', 'XGBoost']
pvals = []
for bn in baseline_names:
    if bn in summaries:
        bl_auc = np.array(summaries[bn]['auc']['vals'])
        _, p = stats.wilcoxon(a1_auc, bl_auc, alternative='two-sided')
        pvals.append((bn, p, a1['auc']['mean'] - summaries[bn]['auc']['mean']))

pvals.sort(key=lambda x: x[1])
n = len(pvals)
print(f"{'Baseline':<20} | {'p (raw)':>10} | {'Diff (A1-BL)':>14}")
for i, (bn, p, diff) in enumerate(pvals):
    # Holm correction
    alpha_corrected = 0.05 / (n - i)
    sig = "***" if p < alpha_corrected else ""
    print(f"  {bn:<18} | {p:>10.4f} | {diff:>14.4f} {sig}")

# --- Collapse rates ---
print("\n--- Collapse rates (seeds where accuracy < 0.55) ---")
collapse_names = list(model_paths.keys())
for name in collapse_names:
    recs = load_test_metrics(model_paths[name])
    accs = [r['accuracy'] for r in recs]
    n_collapsed = sum(1 for a in accs if a < 0.55)
    print(f"  {name:<20}: {n_collapsed}/20 collapsed")
