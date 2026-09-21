# PACE-ASD — Installation Guide

## System Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.11 | 3.11.x |
| OS | Windows 10 / Ubuntu 20.04 | Windows 11 / Ubuntu 22.04 |
| RAM | 8 GB | 16 GB |
| GPU (training) | — | NVIDIA CUDA 12.1+ |
| GPU (inference) | Not required | Optional |
| Storage | 2 GB (code + checkpoints) | 10 GB (+ raw dataset) |

## Step 1: Clone the Repository

```bash
git clone https://github.com/neuro-paradigm/PACE-ASD.git
cd PACE-ASD
```

## Step 2: Create a Virtual Environment

### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux/macOS
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

## Step 3: Install Dependencies

### CPU-only (inference, testing, evaluation — no GPU required)
```bash
pip install -r requirements.txt
```

> **Note:** The `requirements.txt` installs the CPU version of PyTorch by default.
> See below for GPU installation.

### GPU (CUDA 12.1 — required for training from scratch)
```bash
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Step 4: Verify Installation

```bash
python src/verify.py
```

Expected output:
```
  OK  torch 2.1.2  CUDA=False  device=CPU
  OK  numpy=1.26.4  sklearn=1.3.2  cv2=4.8.1.78
  OK  model A1  shape=torch.Size([2])  device=cpu
  OK  model A2  shape=torch.Size([2])  device=cpu
  OK  model A3  shape=torch.Size([2])  device=cpu
  OK  model A4  shape=torch.Size([2])  device=cpu
  OK  metrics  acc=0.80  sens=1.00  spec=0.67
  OK  calibration  T=1.0000
  OK  dataset helpers + augmentation
  OK  baselines (LSTM, Conv1D-BiLSTM, LR, SVM, RF, XGBoost)
  OK  syntax: preprocess, train, ablation, report, interpretability

  ALL CHECKS PASSED (7/7)
```

## Step 5: Run Tests

```bash
python -m pytest tests/ -v
```

All tests use synthetic data — no dataset files needed.

## MediaPipe Note

MediaPipe is required **only** for processing raw `.mp4`/`.avi` video files.
If you are using pre-extracted `.npy` feature files (provided in `processed/features/`),
MediaPipe is **not** needed for inference, evaluation, or testing.

### Known MediaPipe Issues

- **Windows ARM/M-series Mac:** MediaPipe 0.10.x may not support ARM architecture natively.
- **Python 3.12+:** MediaPipe 0.10.14 requires Python ≤ 3.11.
- **GPU conflict:** If CUDA libraries conflict with MediaPipe, use `CUDA_VISIBLE_DEVICES=-1` when running preprocessing.

## Troubleshooting

See `docs/troubleshooting.md` for common issues.
