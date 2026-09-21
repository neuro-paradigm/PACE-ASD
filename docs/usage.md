# PACE-ASD — Usage Guide

## Quick Reference

| Task | Command |
|---|---|
| Inference (.npy) | `python scripts/infer.py --input_npy processed/features/asd_1.npy --checkpoint models/A1/fold1_seed42.pt --config configs/inference.yaml --output outputs/result` |
| Inference (video) | `python scripts/infer.py --input video.mp4 --checkpoint models/A1/fold1_seed42.pt --config configs/inference.yaml --output outputs/result` |
| Evaluate model | `python scripts/evaluate.py --model_id A1` |
| Case study | `python scripts/generate_case_study.py --clip_id asd_1 --checkpoint models/A1/fold1_seed42.pt` |
| Train (single) | `python src/train.py --config configs/config.yaml --model_id A1 --seed 42 --fold 0` |
| Train (full) | `python src/ablation.py --config configs/config.yaml` |
| Preprocess | `python src/preprocess.py --raw_dir /path/to/dryad --out_dir processed` |
| Tests | `python -m pytest tests/ -v` |
| Verify env | `python src/verify.py` |

## Inference

### From Pre-Extracted Features

```bash
python scripts/infer.py \
  --input_npy processed/features/asd_1.npy \
  --checkpoint models/A1/fold1_seed42.pt \
  --config configs/inference.yaml \
  --output outputs/asd1_result
```

This does NOT require MediaPipe or raw videos.

### From Raw Video

```bash
python scripts/infer.py \
  --input path/to/video.mp4 \
  --checkpoint models/A1/fold1_seed42.pt \
  --config configs/inference.yaml \
  --output outputs/video_result
```

This runs MediaPipe Pose extraction followed by the full PACE-ASD pipeline.

### Python API

```python
import sys
sys.path.insert(0, 'src')
from inference_api import PACEASDPredictor

predictor = PACEASDPredictor(
    checkpoint='models/A1/fold1_seed42.pt',
    config='configs/inference.yaml',
)

# From .npy file
result = predictor.predict_npy('processed/features/asd_1.npy')

# From video file
result = predictor.predict_video('path/to/video.mp4')

# Auto-detect source type
result = predictor.predict('processed/features/asd_1.npy')

print(f"Calibrated P(ASD): {result['calibrated_probability']:.3f}")
print(f"Prediction: {result['prediction']}")
print(f"Body-region attribution: {result['body_region_attribution']}")
```

## Evaluation

### Reproduce Published Results

```bash
# Evaluate A1 across all fold/seed combinations
python scripts/evaluate.py --model_id A1

# Evaluate specific fold and seed
python scripts/evaluate.py --model_id A1 --fold 1 --seed 42

# Evaluate all ablation arms
for MODEL in A1 A2 A3 A4; do
  python scripts/evaluate.py --model_id $MODEL
done
```

### Reproduce Paper Statistics

```bash
# Compute summary table
python scripts/compute_stats.py

# Run paired Wilcoxon tests (A1 vs. all baselines)
python scripts/wilcoxon_test.py --results_dir results
```

## Training

### Single Fold (Quick Sanity Check)

```bash
python src/train.py \
  --config configs/config.yaml \
  --model_id A1 \
  --seed 42 \
  --fold 0
```

### Full 20-Seed Ablation

```bash
# All PACE-ASD variants (A1–A4), 20 seeds, 3 folds each
python src/ablation.py --config configs/config.yaml --models A1 A2 A3 A4

# Add baseline comparisons
python src/ablation.py --config configs/config.yaml
```

### Dataset Required for Training

Training requires the raw Dryad dataset. See `data/README.md` for instructions.
Preprocessed `.npy` files in `processed/features/` are sufficient for evaluation and inference.

## Case Study

```bash
python scripts/generate_case_study.py \
  --clip_id asd_1 \
  --checkpoint models/A1/fold1_seed42.pt \
  --config configs/inference.yaml \
  --output outputs/case_study_asd1
```

## Output Files

See `docs/inference.md` for complete output file documentation.
