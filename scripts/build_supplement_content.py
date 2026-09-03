import json, re
import numpy as np

# 1. A1 per-seed table
with open('results/A1_per_seed.json') as f:
    a1_seeds = json.load(f)['test']

a1_rows = []
for i, s in enumerate(a1_seeds):
    a1_rows.append(f"{i+1}  & {s['auc']:.3f} & {s['accuracy']:.3f} & {s['f1']:.3f} & {s['sensitivity']:.3f} & {s['specificity']:.3f} & {s['ece']:.3f} \\\\")

a1_means = {k: np.mean([s[k] for s in a1_seeds]) for k in ['auc', 'accuracy', 'f1', 'sensitivity', 'specificity', 'ece']}
a1_stds  = {k: np.std([s[k] for s in a1_seeds])  for k in ['auc', 'accuracy', 'f1', 'sensitivity', 'specificity', 'ece']}

mean_row = f"\\textbf{{Mean}} & \\textbf{{{a1_means['auc']:.3f}}} & \\textbf{{{a1_means['accuracy']:.3f}}} & \\textbf{{{a1_means['f1']:.3f}}} & \\textbf{{{a1_means['sensitivity']:.3f}}} & \\textbf{{{a1_means['specificity']:.3f}}} & \\textbf{{{a1_means['ece']:.3f}}} \\\\"
std_row  = f"\\textbf{{SD}}   & \\textbf{{{a1_stds['auc']:.3f}}} & \\textbf{{{a1_stds['accuracy']:.3f}}} & \\textbf{{{a1_stds['f1']:.3f}}} & \\textbf{{{a1_stds['sensitivity']:.3f}}} & \\textbf{{{a1_stds['specificity']:.3f}}} & \\textbf{{{a1_stds['ece']:.3f}}} \\\\"

table_s3_1 = "\n".join(a1_rows) + "\n\\midrule\n" + mean_row + "\n" + std_row

# 2. A2, A3, A4 summaries
summaries = {}
for arm in ['A2', 'A3', 'A4']:
    with open(f'results/{arm}_per_seed.json') as f:
        data = json.load(f)['test']
    auc_m, auc_s = np.mean([s['auc'] for s in data]), np.std([s['auc'] for s in data])
    spec_m, spec_s = np.mean([s['specificity'] for s in data]), np.std([s['specificity'] for s in data])
    acc_m, acc_s = np.mean([s['accuracy'] for s in data]), np.std([s['accuracy'] for s in data])
    f1_m, f1_s = np.mean([s['f1'] for s in data]), np.std([s['f1'] for s in data])
    sens_m, sens_s = np.mean([s['sensitivity'] for s in data]), np.std([s['sensitivity'] for s in data])
    ece_m, ece_s = np.mean([s['ece'] for s in data]), np.std([s['ece'] for s in data])
    summaries[arm] = (
        f"Per-seed results available in \\texttt{{results/{arm}\\_per\\_seed.json}}.\n"
        f"Summary: AUC $= {auc_m:.3f} \\pm {auc_s:.3f}$, Specificity $= {spec_m:.3f} \\pm {spec_s:.3f}$, "
        f"Accuracy $= {acc_m:.3f} \\pm {acc_s:.3f}$, F1 $= {f1_m:.3f} \\pm {f1_s:.3f}$, "
        f"Sensitivity $= {sens_m:.3f} \\pm {sens_s:.3f}$, ECE $= {ece_m:.3f} \\pm {ece_s:.3f}$."
    )

# 3. Parse Table S4 (wilcoxon_results.txt)
with open('results/wilcoxon_results.txt', encoding='utf-8') as f:
    w_text = f.read()

sections = w_text.split('=' * 80)

model_names = [
    ('A5_mtcformer', 'MTC-Former'),
    ('A5_lstm', 'Stacked LSTM'),
    ('A5_conv1d_bilstm', 'Conv1D-BiLSTM-Attn'),
    ('A5_kinematic_cnn', 'Kinematic CNN-LSTM'),
    ('A5_msg3d', 'MS-G3D'),
    ('A5_msg3d_convnext', 'MS-G3D + ConvNeXt'),
    ('A5_skelformer', 'SkelFormer'),
    ('A5_stts', 'STTS'),
    ('A5_mtt', 'MTT'),
    ('A5_star', 'STAR'),
    ('A5_lr', 'MediaPipe + LR'),
    ('A5_svm', 'MediaPipe + SVM (RBF)'),
    ('A5_rf', 'MediaPipe + RF'),
    ('A5_xgboost', 'MediaPipe + XGBoost')
]

s4_rows = []
metric_display = {
    'auc': 'AUC',
    'accuracy': 'Accuracy',
    'f1': 'F1',
    'sensitivity': 'Sensitivity',
    'specificity': 'Specificity',
    'ece': 'ECE'
}

for mid, mname in model_names:
    target_pattern = rf"A1 vs {mid}\s+\|"
    target_body = None
    for i, s in enumerate(sections):
        if re.search(target_pattern, s) and i + 1 < len(sections):
            target_body = sections[i+1]
            break
    if not target_body:
        continue

    # Extract each metric row
    lines = [l.strip() for l in target_body.strip().splitlines() if l.strip() and not l.strip().startswith('Metric') and not l.strip().startswith('---') and not l.strip().startswith('**') and not l.strip().startswith('*')]
    first = True
    n_metrics = len(lines)
    for l in lines:
        parts = l.split()
        if len(parts) < 7:
            continue
        m_raw = parts[0]
        a1_val = float(parts[1])
        b_val = float(parts[2])
        diff_val = parts[3]
        # p_val and d_val and stars
        # Find p-val and d
        # Format: auc 0.8364 0.7781 +0.0583 [+0.0437, +0.0724] 0.0000 +1.704 ** PACE better
        # Let's use regex
        rm = re.search(r"([a-z0-9_]+)\s+([0-9.]+)\s+([0-9.]+)\s+([+-][0-9.]+)\s+\[(.*?)\]\s+([0-9.]+)\s+([+-][0-9.]+)(?:\s+(\**|\*))?", l)
        if not rm:
            continue
        m_raw, a1_v, b_v, d_v, ci_v, p_v, cohen_d, stars = rm.groups()
        m_clean = metric_display.get(m_raw, m_raw.upper())
        p_num = float(p_v)
        p_str = "<0.001" if p_num < 0.001 else f"{p_num:.3f}"
        d_str = f"$+${cohen_d[1:]}" if cohen_d.startswith('+') else (f"$-${cohen_d[1:]}" if cohen_d.startswith('-') else cohen_d)
        diff_str = f"$+${d_v[1:]}" if d_v.startswith('+') else (f"$-${d_v[1:]}" if d_v.startswith('-') else d_v)
        
        sig_str = "Yes" if stars == "**" else ("Nominal" if stars == "*" else "No")
        
        if first:
            model_col = f"\\multirow{{{n_metrics}}}{{*}}{{{mname}}}"
            first = False
        else:
            model_col = ""
        s4_rows.append(f"{model_col} & {m_clean} & {float(a1_v):.3f} & {float(b_v):.3f} & {diff_str} & {p_str} & {d_str} & {sig_str} \\\\")
    s4_rows.append("\\midrule")

table_s4_tex = "\n".join(s4_rows)
print(f"Table S4 generated with {len(s4_rows)} lines across {len(model_names)} models.")

with open('scripts/generated_tables.json', 'w', encoding='utf-8') as f:
    json.dump({
        'table_s3_1': table_s3_1,
        'summaries': summaries,
        'table_s4': table_s4_tex
    }, f, indent=2)
print("Saved scripts/generated_tables.json successfully.")
