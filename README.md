# PACE-ASD: Pose-Aware Contiguous Event Saliency-Gated Transformer for Markerless Monocular Video-Based ASD Screening Research

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1.2-orange.svg)](https://pytorch.org/)
[![Dataset: Dryad CC0](https://img.shields.io/badge/Dataset-Dryad%20CC0-green.svg)](https://doi.org/10.5061/dryad.s7h44j150)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

Official research software implementation and reproduction package for **PACE-ASD (Pose-Aware Contiguous Event Saliency-Gated Transformer for Markerless Monocular Video-Based ASD Screening Research)**, prepared for submission to *BMC Medical Informatics and Decision Making*.

---

## Overview

Early screening for Autism Spectrum Disorder (ASD) is critical for timely developmental intervention, yet gold-standard diagnostic assessments (such as ADOS-2) require specialized clinicians and face extensive waitlists. Markerless 2D skeletal pose estimation from standard, monocular RGB video offers an accessible and non-invasive alternative for quantitative motor screening.

Existing deep action recognition architectures (e.g., spatio-temporal GCNs and dense video Transformers) face two major obstacles in pediatric motor screening:
1. **Temporal Dilution:** Transient, clinically informative atypical motor events are drowned within lengthy sequences of ordinary background movement when applying dense attention or global pooling.
2. **Catastrophic Training Collapse:** Over-parameterized architectures frequently collapse during optimization when trained on modest clinical cohorts ($N < 100$).

**PACE-ASD** resolves these challenges by introducing an inductive temporal bias via the **Block-Level Event Saliency Gate (Block-ESG)**:
- Instead of attending over all 300 frames or selecting arbitrary, disconnected individual frames, Block-ESG dynamically identifies and routes the **$M = 8$ most kinematically salient contiguous 15-frame blocks** (500 ms motion primitives at 30 fps; 120 frames total) to a lightweight temporal Transformer encoder.
- Provides **complete training stability** (0/20 collapsed seeds vs. up to 10/20 collapse in literature baselines).
- Enforces **mechanistic interpretability** through a two-stage coherence audit: validating that gating saliency and self-attention weights reinforce the same motor events ($r = 0.822$).

---

## What the Software Does

The software implements an end-to-end reproducible research pipeline:
1. **Pose Extraction:** Extracts 33 2D skeletal keypoints per frame from monocular RGB video using MediaPipe Pose.
2. **Kinematic Normalization:** Normalizes coordinates via mid-hip centering and inter-shoulder distance scaling; computes velocity and acceleration finite differences.
3. **Multi-Scale Temporal Convolution:** Extracts short-timescale motion features via parallel 1D convolutions (k=1, 3, 5) with GroupNorm.
4. **Contiguous Event Saliency Gating:** Scores 15-frame blocks and selects the top-8 most salient contiguous segments.
5. **Temporal Transformer Attention:** Aggregates selected event tokens using self-attention with absolute sinusoidal positional encoding.
6. **Calibrated Screening Inference:** Outputs raw and Platt-calibrated P(ASD) probabilities with Expected Calibration Error (ECE) monitoring.
7. **Comprehensive Interpretability:** Decomposes decision evidence across body regions (head, arms, torso, legs) and kinematic streams (position, velocity, acceleration).

---

## System Architecture

```
Monocular RGB Video (.mp4 / .avi)
        │
        ▼  MediaPipe Pose (33 2-D Keypoints, T = 300 frames)
Normalized Pose Sequence (B, 300, 33, 2)
        │
        ├──► SpatialEncoder (Per-frame MLP: pos + vel + acc -> D_c = 198) ──► (B, 300, 128)
        │                                                                            │
        └──► MicrokineticEncoder (Conv1D: k=1, 3, 5 + GroupNorm)                      │
                    │                                                                │
                    ▼                                                                │
             Salience Gate (Linear 96 -> 48 -> 1)                                    │
                    │                                                                │
                    ▼                                                                │
             Block-ESG Pooling (L = 15 frames, M = 8 blocks)                         │
                    │                                                                │
                    ▼ (Top-8 Contiguous Blocks Selected)                             │
             Selected Tokens (B, 120, 128) ◄─────────────────────────────────────────┘
                    │
                    ▼
             Temporal Event Transformer (1 Layer, 4 Heads, d = 128)
                    │
                    ▼ Mean Pooling across Active Tokens
             Classification Head (MLP: 128 -> 64 -> 1)
                    │
                    ▼ Platt Temperature Scaling
             Calibrated Prediction: P(ASD) ∈ [0, 1]
```

### Module Reference

| Component | File | Function |
|---|---|---|
| `SpatialEncoder` | `src/model.py` | Per-frame MLP with LayerNorm; computes position, velocity, and acceleration streams |
| `MicrokineticEncoder` | `src/model.py` | Multi-scale temporal feature extractor (Conv1d k=1, 3, 5 + GroupNorm) |
| `EventSaliencyGate` | `src/model.py` | Block-ESG: groups frames into contiguous blocks, scores saliency, routes top-M |
| `TemporalEventTransformer`| `src/model.py` | Sparse event token transformer with sinusoidal position encoding |
| `ASDMotionModel` | `src/model.py` | Complete end-to-end model with ablation flags (A1, A2, A3, A4) |
| `PlattScaler` | `src/calibration.py` | Post-hoc temperature and bias recalibration preserving AUC ranking |
| `compute_bilateral_asymmetry` | `src/asymmetry.py` | Descriptive bilateral movement difference analysis |
| `PACEASDPredictor` | `src/inference_api.py` | High-level Python API for single-sample inference |

---

## Features

- **Markerless Monocular Input:** Operates on standard consumer RGB video without requiring depth sensors or wearable markers.
- **Automated Pose Extraction:** Built-in MediaPipe Pose pipeline with automatic hip-centering and shoulder-distance scale normalization.
- **Exposed Kinematic Trajectories:** Saves full position, velocity, and acceleration tensors (`kinematics.npz`).
- **Contiguous Temporal Event Discovery:** Identifies and extracts specific salient movement windows (`selected_events.json`).
- **Multi-Level Interpretability:** Provides body-region (head, arms, torso, legs) and kinematic-stream (position, velocity, acceleration) attributions (`attribution.json`).
- **Calibrated Decision Outputs:** Uses Platt scaling to ensure predicted probabilities match empirical risk.
- **Descriptive Bilateral Asymmetry:** Computes left-right movement differences as descriptive movement features (not claimed as standalone biomarkers).
- **Dual Inference Entry Points:** Accepts either raw video (`.mp4`, `.avi`) or pre-extracted landmark tensors (`.npy`).
- **Complete Test Suite:** 9 unit and end-to-end tests covering all modules with synthetic data.
- **Strict Reproducibility:** Frozen cross-validation splits, fixed seeds, and pre-trained checkpoints included.

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/neuro-paradigm/PACE-ASD.git
cd PACE-ASD
```

### 2. Environment Setup

```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate

# Linux / macOS:
source .venv/bin/activate
```

### 3. Install Dependencies

For CPU inference and testing (no GPU required):
```bash
pip install -r requirements.txt
```

For GPU-accelerated training (CUDA 12.1):
```bash
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 4. Verify Setup

```bash
python src/verify.py
```

---

## Quick Start

### Run Inference on a Pre-extracted Sample (No MediaPipe required)

```bash
python scripts/infer.py \
    --input_npy processed/features/asd_1.npy \
    --checkpoint models/A1/fold1_seed42.pt \
    --config configs/inference.yaml \
    --output outputs/example_asd1
```

### Run Inference on a Raw Video

```bash
python scripts/infer.py \
    --input path/to/video.mp4 \
    --checkpoint models/A1/fold1_seed42.pt \
    --config configs/inference.yaml \
    --output outputs/example_video
```

### Run Unit Tests

```bash
python -m pytest tests/ -v
```

---

## Inference

The inference command produces structured JSON and NPZ files along with publication-quality visualizations.

```bash
python scripts/infer.py \
    --input_npy processed/features/asd_1.npy \
    --checkpoint models/A1/fold1_seed42.pt \
    --config configs/inference.yaml \
    --output outputs/demo
```

### Python API

```python
import sys
sys.path.insert(0, 'src')
from inference_api import PACEASDPredictor

predictor = PACEASDPredictor(
    checkpoint='models/A1/fold1_seed42.pt',
    config='configs/inference.yaml'
)

# Accepts .npy or .mp4
result = predictor.predict('processed/features/asd_1.npy')

print("Prediction:", result["prediction"])
print("Calibrated P(ASD):", result["calibrated_probability"])
print("Body-Region Attribution:", result["body_region_attribution"])
print("Selected Events:", result["selected_events"])
```

---

## Output Files

Each inference execution populates the target directory with:

```
outputs/demo/
├── result.json                   # Calibrated probability, prediction, metadata
├── kinematics.npz                # 3D arrays: positions, velocities, accelerations
├── selected_events.json          # Block-ESG selected temporal blocks & scores
├── attribution.json              # Body-region and kinematic-stream attribution
└── visualizations/
    ├── kinematics.png            # Trajectory graphs (position, velocity, acceleration)
    ├── event_saliency.png        # Block-ESG saliency bar plot
    └── evidence_summary.png      # 4-panel comprehensive evidence summary
```

See [`docs/inference.md`](docs/inference.md) for full schema details.

---

## Movement and Kinematic Outputs

All kinematic representations are preserved and exposed in `kinematics.npz`:
- `positions`: $(300, 33, 2)$ float32, mid-hip centred and inter-shoulder scaled.
- `velocities`: $(300, 33, 2)$ float32, first-order finite differences with boundary protection ($\times 10$ scaling).
- `accelerations`: $(300, 33, 2)$ float32, second-order finite differences ($\times 5$ scaling).
- `frame_indices`: $(300,)$ int32, temporal timeline indices.

---

## Temporal Event Outputs

The Block-ESG module exposes:
- Selected contiguous blocks of $L=15$ frames (~500 ms motion primitives).
- Exact block boundary frames in `selected_events.json`.
- Gate saliency scores across all candidate blocks, indicating which segments exhibited atypical kinematic profiles.

---

## Interpretability Outputs

Interpretability is exposed via two complementary mechanisms:
1. **Body-Region Attribution:** Proportional Gradient $\times$ Input importance grouped into Head (landmarks 0–10), Arms (11–22), Torso (23–24), and Legs (25–32).
2. **Kinematic-Stream Attribution:** Relative contribution of spatial position vs. velocity vs. acceleration streams.
3. **Mechanistic Coherence:** Consistency between gating saliency and Transformer self-attention density ($r = 0.822$).

---

## Training

To train a single PACE-ASD model fold:
```bash
python src/train.py --config configs/config.yaml --model_id A1 --seed 42 --fold 0
```

To run the complete 20-seed $\times$ 3-fold cross-validation experiment:
```bash
python src/ablation.py --config configs/config.yaml --models A1
```

*Note:* Training requires the underlying raw Dryad dataset. See [`data/README.md`](data/README.md).

---

## Evaluation

To reproduce evaluation metrics across frozen splits using pre-trained checkpoints:
```bash
python scripts/evaluate.py --model_id A1
```

To compute comprehensive benchmark statistics and collapse audits:
```bash
python scripts/compute_stats.py
```

To run multi-model paired Wilcoxon signed-rank tests:
```bash
python scripts/wilcoxon_test.py --results_dir results
```

---

## Reproducibility

- **Frozen Splits:** Stored in `splits/splits_dryad_v2_dedup.json`.
- **Pre-computed Results:** Stored in `results/A1_per_seed.json` through `results/A5_*_per_seed.json`.
- **Pre-trained Checkpoints:** Included in `models/A1/` (60 checkpoints across 3 folds and 20 seeds).
- Detailed reproduction steps are provided in [`docs/reproducibility.md`](docs/reproducibility.md).

---

## Dataset

Evaluated on the open-access **Dryad ASD Kinematic Dataset**:
> Aljubouri, A. A., Hadi, I., & Rajihy, Y. (2020). *Three Dimensional Dataset Combining Gait and Full Body Movement of Children with Autism Spectrum Disorders Collected by Kinect v2 Camera.* Dryad Digital Repository. [doi:10.5061/dryad.s7h44j150](https://doi.org/10.5061/dryad.s7h44j150).

### Forensic Deduplication Audit
An MD5 cryptographic audit identified 5 pairs of byte-identical TD video recordings in the public release. One duplicate from each pair was quarantined in `processed/removed_duplicates/` to prevent train-test contamination, yielding a clean main cohort of **$N = 90$ subjects (45 ASD, 45 TD)**. See [`data/README.md`](data/README.md).

---

## Checkpoints

- Checkpoints for the primary A1 model (60 models: 3 folds $\times$ 20 seeds) are located in `models/A1/`.
- Additional baseline model checkpoints are stored in `models/A2/`, `models/A3/`, `models/A4/`, and `models/A5_*/`.
- If retraining is needed, scripts and hyperparameters are fully documented in [`checkpoints/README.md`](checkpoints/README.md).

---

## Repository Structure

```
PACE-ASD/
├── configs/
│   ├── config.yaml                    # Master training configuration
│   ├── inference.yaml                 # Inference settings & thresholds
│   └── evaluation.yaml                # Model evaluation configurations
├── docs/
│   ├── installation.md                # Environment setup instructions
│   ├── usage.md                       # Comprehensive CLI & API usage
│   ├── architecture.md                # Mathematical architecture details
│   ├── inference.md                   # Output schemas and interpretation
│   ├── reproducibility.md             # Benchmark reproduction protocol
│   ├── software_comparison.md         # Comparison matrix vs. existing tools
│   ├── troubleshooting.md             # FAQ and common fixes
│   └── SUBMISSION_READINESS.md        # BMC submission audit checklist
├── src/
│   ├── model.py                       # Core PACE-ASD model & ablation variants
│   ├── inference_api.py               # PACEASDPredictor class
│   ├── asymmetry.py                   # Descriptive bilateral asymmetry utility
│   ├── calibration.py                 # PlattScaler temperature scaling
│   ├── interpretability.py            # Gradient x Input & attention coherence
│   ├── preprocess.py                  # MediaPipe pose extraction & normalization
│   ├── dataset.py                     # Sequence loaders & data augmentation
│   ├── train.py                       # Model training loop & early stopping
│   ├── ablation.py                    # Multi-seed CV ablation runner
│   ├── baselines.py                   # 14 literature deep learning & ML baselines
│   ├── metrics.py                     # Evaluation metrics (AUC, ECE, F1, CI)
│   ├── report.py                      # PDF report generator
│   └── verify.py                      # Installation verification script
├── scripts/
│   ├── infer.py                       # Single-sample inference CLI
│   ├── evaluate.py                    # Checkpoint evaluation reproducer
│   ├── generate_case_study.py         # 8-panel case study generator
│   ├── compute_stats.py               # Benchmark table calculator
│   ├── wilcoxon_test.py               # Statistical significance testing
│   └── audit_clip_lengths.py          # Frame count and duration audit
├── tests/
│   ├── conftest.py                    # Pytest fixtures & synthetic data
│   ├── test_pose.py                   # Keypoint normalization tests
│   ├── test_normalization.py          # Scale & centering tests
│   ├── test_kinematics.py             # Velocity & acceleration tests
│   ├── test_asymmetry.py              # Bilateral asymmetry tests
│   ├── test_block_esg.py              # Event saliency gate tests
│   ├── test_model_shapes.py           # Forward pass tensor shape tests
│   ├── test_calibration.py            # Platt scaling tests
│   └── test_end_to_end.py             # End-to-end inference tests
├── processed/
│   ├── features/                      # Preprocessed landmark arrays (*.npy)
│   ├── removed_duplicates/            # Quarantined duplicate recordings
│   └── labels.csv                     # Cohort metadata
├── models/                            # Trained model checkpoints (*.pt)
├── splits/                            # Frozen 3-fold CV splits
├── results/                           # Benchmark JSON & CSV results
├── examples/                          # Example inputs and expected outputs
├── release/                           # Reviewer release bundle & guide
├── requirements.txt                   # Pinned dependency specification
├── pyproject.toml                     # Package build configuration
├── CITATION.cff                       # Citation metadata
└── README.md                          # Main repository documentation
```

---

## Limitations

- **Research Software Only:** PACE-ASD is an academic research software tool, not a cleared medical device. It does not provide clinical diagnoses.
- **Demographic Representation:** Evaluated on the Dryad pediatric cohort ($N=90$); generalization across diverse camera angles, lighting conditions, and age groups requires broader clinical validation.
- **2D Keypoints:** 3D depth is not reconstructed; out-of-plane rotational movements may affect tracking fidelity.
- **Descriptive Asymmetry:** Bilateral asymmetry metrics are descriptive kinematic measures and are not validated standalone clinical biomarkers.

---

## Citation

If you utilize this codebase or research in your work, please cite:

```bibtex
@article{puppala2026paceasd,
  title   = {{PACE-ASD}: Pose-Aware Contiguous Event Saliency-Gated Transformer
             for Markerless Monocular Video-Based {ASD} Screening Research},
  author  = {Puppala, Sireesha and {Kasi}, {Vamshi Mohan} and
             Tanuku, {VVS Sai Tejesh} and Kota, Preetham and
             Annabathula, {Yuva Dhanvanth}},
  journal = {BMC Medical Informatics and Decision Making},
  year    = {2026}
}
```

See [`CITATION.cff`](CITATION.cff) for complete metadata.

---

## Data License

- **Dataset:** [CC0 1.0 Universal Public Domain](https://creativecommons.org/publicdomain/zero/1.0/) (Dryad Digital Repository)

---

## Authors & Contact

| Author | Role | Affiliation | ORCID |
|---|---|---|---|
| **Sireesha Puppala** (Corresponding Author) | Conceptualization, Methodology, Supervision | Dept. of CSE, KMIT; NeuroParadigm Pvt. Ltd., Hyderabad | [0009-0008-3984-9665](https://orcid.org/0009-0008-3984-9665) |
| **Kasi Vamshi Mohan** (Co-First Author) | Training infrastructure, statistical analysis | Dept. of CSE (AI&ML), KMIT; NeuroParadigm Pvt. Ltd. | [0009-0005-7715-8388](https://orcid.org/0009-0005-7715-8388) |
| **Tanuku VVS Sai Tejesh** (Co-First Author) | Architecture formulation, Block-ESG mathematics | Dept. of CSE (AI&ML), KMIT; NeuroParadigm Pvt. Ltd. | [0009-0003-9845-8190](https://orcid.org/0009-0003-9845-8190) |
| **Preetham Kota** (Co-First Author) | Baseline re-implementation and evaluation | Dept. of CSE (AI&ML), KMIT; NeuroParadigm Pvt. Ltd. | [0009-0003-6753-5608](https://orcid.org/0009-0003-6753-5608) |
| **Annabathula Yuva Dhanvanth** (Co-First Author) | MediaPipe extraction, MD5 audit | Dept. of CSE (AI&ML), KMIT; NeuroParadigm Pvt. Ltd. | [0009-0003-7401-1394](https://orcid.org/0009-0003-7401-1394) |

**Corresponding Author:** Sireesha Puppala ([sireesha@neuroparadigm.in](mailto:sireesha@neuroparadigm.in))
