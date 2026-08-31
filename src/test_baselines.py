"""Test all baselines forward pass — write results to a log file."""
import sys, os, ast, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

log_lines = []

def log(msg):
    print(msg, flush=True)
    log_lines.append(msg)

# 1. Syntax
ast.parse(open('src/baselines.py', encoding='utf-8').read())
log("OK  syntax")

# 2. Imports
from baselines import (
    StackedLSTM, Conv1DBiLSTMAttn, KinematicCNNLSTM,
    STTS, MSG3D, MSG3DConvNeXt,
    SkelFormer, MTCFormer, MTT, STAR,
    build_sklearn_baseline, ALL_BASELINE_IDS,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log("Device: %s" % device)

x = torch.randn(2, 300, 33, 2).to(device)

models = {
    'StackedLSTM':      StackedLSTM,
    'Conv1DBiLSTMAttn': Conv1DBiLSTMAttn,
    'KinematicCNNLSTM': KinematicCNNLSTM,
    'STTS':             STTS,
    'MSG3D':            MSG3D,
    'MSG3DConvNeXt':    MSG3DConvNeXt,
    'SkelFormer':       SkelFormer,
    'MTCFormer':        MTCFormer,
    'MTT':              MTT,
    'STAR':             STAR,
}

all_ok = True
for name, cls in models.items():
    try:
        m = cls().to(device)
        _, logits = m(x)
        assert logits.shape == (2,)
        params = sum(v.numel() for v in m.parameters()) / 1e6
        log("OK  %-22s params=%.2fM  dev=%s" % (name, params, logits.device))
    except Exception as e:
        log("ERR %-22s  %s" % (name, e))
        all_ok = False

for n in ['lr', 'svm', 'rf', 'xgboost']:
    try:
        build_sklearn_baseline(n)
        log("OK  sklearn/%-10s" % n)
    except Exception as e:
        log("ERR sklearn/%s: %s" % (n, e))
        all_ok = False

log("")
log("ALL_BASELINE_IDS: " + str(ALL_BASELINE_IDS))
log("")
log("RESULT: " + ("ALL PASSED" if all_ok else "SOME FAILED"))

# Write to log file
with open('baseline_test_results.txt', 'w') as f:
    f.write('\n'.join(log_lines))
