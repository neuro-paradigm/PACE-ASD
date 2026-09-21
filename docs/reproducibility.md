# PACE-ASD — Reproducibility Guide

## Environment

| Component | Version |
|---|---|
| Python | 3.11.x |
| PyTorch | 2.1.2 |
| MediaPipe | 0.10.14 |
| NumPy | 1.26.4 |
| scikit-learn | 1.3.2 |
| OS | Windows 11 / Ubuntu 22.04 |
| CUDA (training) | 12.1 (optional) |

```bash
pip install -r requirements.txt
```

## Random Seeds

- **Split seed:** 42 (frozen in `splits/splits_dryad_v2_dedup.json`)
- **Training seeds:** 42, 43, 44, ..., 61 (20 seeds)
- **Seed control:** `random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`; `cudnn.deterministic=True`, `cudnn.benchmark=False`

## Dataset

The **Dryad ASD Kinematic Dataset** (CC0 1.0, publicly available):
> Aljubouri et al. (2020). DOI: [10.5061/dryad.s7h44j150](https://doi.org/10.5061/dryad.s7h44j150)

See `data/README.md` for download and directory structure instructions.

**Pre-processed features** (already included):
- `processed/features/*.npy` — 105 clips, (300, 33, 2) float32
- `processed/labels.csv` — clip metadata

## Dataset Split

Frozen split file: `splits/splits_dryad_v2_dedup.json`
- 90 subjects (45 ASD + 45 TD) after deduplication
- 25% held-out test set (subject-level stratified split, seed=42)
- 3-fold StratifiedKFold on remaining 75%
- 9 severe-ASD supplement subjects (never in train/val/test)

## Preprocessing

Already completed — features in `processed/features/`. To re-extract from raw video:

```bash
# Verify raw videos are present
python src/audit.py --raw_dir /path/to/dryad

# Re-extract (overwrites existing .npy only if absent)
python src/preprocess.py --raw_dir /path/to/dryad --out_dir processed
```

Preprocessing is deterministic given fixed MediaPipe version and input video.

## Reproducing Evaluation Metrics

### Option A: Using Provided Checkpoints (Fastest)

```bash
# Reproduce A1 (Full PACE-ASD) test metrics
python scripts/evaluate.py --model_id A1

# Compare with stored per-seed results
python scripts/compute_stats.py
```

### Option B: Full Retraining from Scratch

```bash
# Single sanity check (fold 0, seed 42, ~5-10min on GPU)
python src/train.py --config configs/config.yaml --model_id A1 --seed 42 --fold 0

# Full A1 (20 seeds × 3 folds = 60 runs, ~8h on GPU)
python src/ablation.py --config configs/config.yaml --models A1

# All ablation arms (A1-A4 + 14 baselines, ~60h on GPU)
python src/ablation.py --config configs/config.yaml
```

### Wilcoxon Statistical Tests

```bash
# Paired Wilcoxon signed-rank tests (A1 vs. all baselines)
python scripts/wilcoxon_test.py --results_dir results
```

## Expected Output Format

Evaluation produces per-seed metrics in `results/<model_id>_per_seed.json`:

```json
{
  "val":  [{"accuracy": 0.72, "auc": 0.84, "f1": 0.74, ...}, ...],
  "test": [{"accuracy": 0.73, "auc": 0.84, "f1": 0.74, ...}, ...]
}
```

Aggregated as mean ± SD across 20 seeds.

## Checkpoints

- Location: `models/A1/fold{1-3}_seed{42-61}.pt`
- Included in repository (tracked by git)
- Each checkpoint includes: model weights, training config, Platt scaler, validation metrics

## Known Determinism Limitations

- PyTorch CUDA operations may not be fully deterministic across hardware
- Results on CPU are expected to be deterministic
- Minor numerical differences (±0.001) may occur across operating systems due to floating-point implementation differences
