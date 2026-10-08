# REPRODUCE.md — Reproduction Guide

**Purpose:** Protocol Section 5 — one command per result, exact environment.

---

## Environment

```
Python  : 3.11.x
CUDA    : 12.1
OS      : Windows 11 / Ubuntu 22.04
```

```bash
pip install -r requirements.txt
```

Key pinned packages:
```
torch==2.1.2   mediapipe==0.10.14   opencv-python==4.8.1.78
numpy==1.26.4  scikit-learn==1.3.2  scipy==1.11.4
```

---

## Step 1 — Audit

Verify all raw videos are present (109 subjects, 110 videos):

```bash
python src/audit.py --raw_dir "D:/dryad"
```

Expected output: `✓ Audit PASSED — all expected files present.`

---

## Step 2 — Preprocess

Run MediaPipe on all raw videos and produce `processed/features/*.npy`:

```bash
python src/preprocess.py --raw_dir "D:/dryad" --out_dir processed
```

Dry-run (no files written, just reports what would be processed):
```bash
python src/preprocess.py --raw_dir "D:/dryad" --out_dir processed --dry_run
```

Single-subject sanity check:
```bash
python src/preprocess.py --raw_dir "D:/dryad" --out_dir processed --subjects asd_1 td_1
python -c "import numpy as np; x=np.load('processed/features/asd_1.npy'); print(x.shape, x.dtype)"
# Expected: (300, 33, 2) float32
```

---

## Step 3 — Freeze Splits

Splits are frozen automatically on first run. To generate them explicitly:

```bash
python -c "
import yaml, sys
sys.path.insert(0,'src')
from train import freeze_splits
cfg = yaml.safe_load(open('configs/config.yaml'))
freeze_splits(cfg)
"
```

Output: `splits/splits_dryad_only_v1.json`

---

## Step 4 — Sanity-check one fold

```bash
python src/train.py --config configs/config.yaml --model_id A1 --seed 0 --fold 0
```

---

## Step 5 — Full Ablation (A1–A5, 20 seeds × 3 folds)

```bash
python src/ablation.py --config configs/config.yaml
```

Run a subset:
```bash
python src/ablation.py --config configs/config.yaml --models A1 A2 A3
```

---

## Step 6 — Section 6 Supplementary Evaluation

```bash
python src/ablation.py --config configs/config.yaml --eval_supplement
```

---

## Outputs

| Path | Content |
|---|---|
| `processed/features/*.npy` | (300, 33, 2) float32 keypoint arrays |
| `processed/labels.csv` | clip metadata |
| `splits/splits_dryad_only_v1.json` | frozen subject-level splits |
| `models/{model_id}/fold*_seed*.pt` | checkpoints + Platt scaler |
| `reports/{model_id}/fold*_report.pdf` | per-fold PDFs |
| `results/ablation_results.csv` | mean ± SD across 20 seeds |
| `results/ablation_table.pdf` | paper-ready comparison table |
| `results/supplement_results.csv` | Section 6 sensitivity + CI |
| `audit_report.txt` | dataset audit log |

---

## Commit Hash

> **TODO:** Pin commit hash here after final run before submission.
> ```
> git tag dryad_only_v1
> git rev-parse HEAD
> ```
