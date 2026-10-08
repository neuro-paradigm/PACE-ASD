# PACE-ASD — Reviewer Quick-Start Guide

Thank you for reviewing PACE-ASD. This guide covers everything needed to
install, run, test, and reproduce the reported results within 15–30 minutes.

---

## 1. Installation

```bash
git clone https://github.com/neuro-paradigm/PACE-ASD.git
cd PACE-ASD
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
python src/verify.py  # should end: ALL CHECKS PASSED (7/7)
```

---

## 2. Run Tests (No Dataset Required)

All tests use synthetic data. Should complete in < 60 seconds.

```bash
python -m pytest tests/ -v
```

Expected: all tests pass.

---

## 3. Example Inference (No Dataset, No MediaPipe Required)

Pre-extracted features are included in `processed/features/`.
Pre-trained checkpoints are in `models/A1/`.

```bash
python scripts/infer.py \
  --input_npy processed/features/asd_1.npy \
  --checkpoint models/A1/fold1_seed42.pt \
  --config configs/inference.yaml \
  --output outputs/reviewer_test
```

**Expected outputs in `outputs/reviewer_test/`:**
- `result.json` — calibrated probability, prediction, attribution
- `kinematics.npz` — position/velocity/acceleration arrays
- `selected_events.json` — Block-ESG selected frame blocks
- `attribution.json` — body-region and stream attribution
- `visualizations/` — PNG plots

---

## 4. Generate Case Study

```bash
python scripts/generate_case_study.py \
  --clip_id asd_1 \
  --checkpoint models/A1/fold1_seed42.pt \
  --output outputs/case_study_asd1
```

Produces 8 visualization panels demonstrating the full pipeline.

---

## 5. Reproduce Evaluation Metrics

```bash
# Reproduce test metrics for A1 (Full PACE-ASD) across all folds and seeds
python scripts/evaluate.py --model_id A1

# View pre-computed results (all 18 models)
python scripts/compute_stats.py
```

---

## 6. Inspect Intermediate Outputs

```python
import json, numpy as np

# Kinematic outputs
data = np.load('outputs/reviewer_test/kinematics.npz')
print('Positions shape:', data['positions'].shape)      # (300, 33, 2)
print('Velocities shape:', data['velocities'].shape)    # (300, 33, 2)
print('Accelerations shape:', data['accelerations'].shape)  # (300, 33, 2)

# Event saliency
events = json.load(open('outputs/reviewer_test/selected_events.json'))
print('Selected blocks:', events['selected_block_starts'])

# Attribution
attr = json.load(open('outputs/reviewer_test/attribution.json'))
print('Body-region attribution:', attr['body_region_attribution'])
print('Kinematic-stream attribution:', attr['kinematic_stream_attribution'])

# Final result
result = json.load(open('outputs/reviewer_test/result.json'))
print(f"Calibrated P(ASD): {result['calibrated_probability']:.3f}")
print(f"Prediction: {result['prediction']}")
```

---

## 7. Known Limitations

- **Training requires raw video** from the Dryad dataset (not included; see `data/README.md`)
- **MediaPipe is only needed** for processing raw `.mp4`/`.avi` video; pre-extracted `.npy` files are already provided
- **CPU inference is fully supported**; GPU accelerates training but is not required for inference or testing
- **Single-subject inference is a research output**, not a clinical diagnosis

---

## 8. Repository Navigation

| Question | Where to look |
|---|---|
| Model architecture | `src/model.py`, `docs/architecture.md` |
| Training | `src/train.py`, `docs/reproducibility.md` |
| Inference pipeline | `scripts/infer.py`, `docs/inference.md` |
| Output file meanings | `docs/inference.md` |
| Dataset download | `data/README.md` |
| Checkpoint format | `checkpoints/README.md` |
| Test suite | `tests/` |
| Comparison table | `docs/software_comparison.md` |
