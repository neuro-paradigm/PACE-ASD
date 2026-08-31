"""Final environment verification — run with the venv python."""
import sys
sys.path.insert(0, "src")

import torch
import numpy as np

ok = []
fail = []

# ── 1. CUDA ────────────────────────────────────────────────────────────────────
cuda = torch.cuda.is_available()
device_name = torch.cuda.get_device_name(0) if cuda else "CPU"
msg = "torch %s  CUDA=%s  device=%s" % (torch.__version__, cuda, device_name)
print("  OK  " + msg)
ok.append("torch+cuda")

# ── 2. Core packages ───────────────────────────────────────────────────────────
try:
    import numpy, scipy, sklearn, matplotlib, pandas, yaml, tqdm, cv2
    print("  OK  numpy=%s  sklearn=%s  cv2=%s" % (
        numpy.__version__, sklearn.__version__, cv2.__version__))
    ok.append("core_packages")
except Exception as e:
    print("  ERR core_packages:", e); fail.append(e)

# ── 3. Model forward passes (GPU) ──────────────────────────────────────────────
try:
    import yaml as _yaml
    from model import ASDMotionModel
    cfg = _yaml.safe_load(open("configs/config.yaml"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x   = torch.randn(2, 300, 33, 2).to(dev)
    for name, kw in [
        ("A1", {"use_gate": True,  "use_transformer": True}),
        ("A2", {"use_gate": False, "use_transformer": True}),
        ("A3", {"use_gate": True,  "use_transformer": True}),
        ("A4", {"use_gate": True,  "use_transformer": False}),
    ]:
        m = ASDMotionModel(cfg, **kw).to(dev)
        _, logits = m(x)
        assert logits.shape == (2,)
        print("  OK  model %s  shape=%s  device=%s" % (name, logits.shape, logits.device))
    ok.append("model_variants")
except Exception as e:
    print("  ERR model:", e); fail.append(e)

# ── 4. Metrics + calibration ───────────────────────────────────────────────────
try:
    from metrics import compute_all_metrics, aggregate_seed_metrics, compute_supplement_sensitivity
    from calibration import PlattScaler
    y    = np.array([1, 1, 0, 0, 1])
    pred = np.array([1, 0, 0, 0, 1])
    prob = np.array([0.9, 0.3, 0.2, 0.1, 0.8])
    m    = compute_all_metrics(y, pred, prob)
    assert abs(m["accuracy"] - 0.8) < 0.01
    sc = PlattScaler()
    sc.fit(np.array([-1., 0., 1., 2.]), np.array([0., 0., 1., 1.]))
    assert sc.calibrate(np.array([0.0])).shape == (1,)
    print("  OK  metrics  acc=%.2f  sens=%.2f  spec=%.2f" % (
        m["accuracy"], m["sensitivity"], m["specificity"]))
    print("  OK  calibration  T=%.4f" % sc.temperature.item())
    ok.append("metrics+calib")
except Exception as e:
    print("  ERR metrics/calib:", e); fail.append(e)

# ── 5. Dataset helpers ─────────────────────────────────────────────────────────
try:
    from dataset import extract_subject_id, augment_sequence
    assert extract_subject_id("asd_1")          == "asd_1"
    assert extract_subject_id("td_10")           == "td_10"
    assert extract_subject_id("severe_case2_v1") == "severe_case2"
    assert extract_subject_id("severe_case2_v2") == "severe_case2"
    seq = np.random.randn(300, 33, 2).astype(np.float32)
    aug = augment_sequence(seq)
    assert aug.shape == (300, 33, 2)
    print("  OK  dataset helpers + augmentation")
    ok.append("dataset")
except Exception as e:
    print("  ERR dataset:", e); fail.append(e)

# ── 6. Baselines ───────────────────────────────────────────────────────────────
try:
    from baselines import StackedLSTM, Conv1DBiLSTMAttn, build_sklearn_baseline
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x2  = torch.randn(2, 300, 33, 2).to(dev)
    for cls in [StackedLSTM, Conv1DBiLSTMAttn]:
        model = cls().to(dev)
        _, logits = model(x2)
        assert logits.shape == (2,)
    for name in ["lr", "svm", "rf", "xgboost"]:
        assert build_sklearn_baseline(name) is not None
    print("  OK  baselines (LSTM, Conv1D-BiLSTM, LR, SVM, RF, XGBoost)")
    ok.append("baselines")
except Exception as e:
    print("  ERR baselines:", e); fail.append(e)

# ── 7. Syntax check remaining files ───────────────────────────────────────────
try:
    import ast
    for f in ["src/preprocess.py", "src/train.py",
              "src/ablation.py", "src/report.py", "src/interpretability.py"]:
        ast.parse(open(f).read())
    print("  OK  syntax: preprocess, train, ablation, report, interpretability")
    ok.append("syntax")
except SyntaxError as e:
    print("  ERR syntax:", e); fail.append(e)

# ── Summary ────────────────────────────────────────────────────────────────────
print()
print("=" * 50)
if not fail:
    print("  ALL CHECKS PASSED (%d/7)" % len(ok))
    print("  Environment is ready.")
    print("  Next step: python src/audit.py --raw_dir D:/dryad")
else:
    print("  FAILED: %d error(s)" % len(fail))
    sys.exit(1)
