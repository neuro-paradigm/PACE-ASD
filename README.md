# PACE-ASD: Pose-Aware Contiguous Event Saliency-Gated Transformer for Markerless Monocular Video-Based ASD Screening Research

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1.2-orange.svg)](https://pytorch.org/)
[![Dataset: Dryad CC0](https://img.shields.io/badge/Dataset-Dryad%20CC0-green.svg)](https://doi.org/10.5061/dryad.s7h44j150)
[![Protocol: SAP v1.0 Locked](https://img.shields.io/badge/Protocol-SAP%20v1.0%20(55bdce4)-purple.svg)](STATISTICAL_ANALYSIS_PLAN.md)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

Official research software implementation and reproduction codebase for **PACE-ASD (Pose-Aware Contiguous Event Saliency-Gated Transformer for Markerless Monocular Video-Based ASD Screening Research)**.

---

## 📌 Executive Summary & Scope

**PACE-ASD** is an open-source research software framework designed to quantify atypical motor patterns in children with Autism Spectrum Disorder (ASD) from ordinary, monocular 2D RGB video. 

### Key Characteristics & Study Boundaries
- **Strictly Single-Site Pediatric Cohort:** Evaluated exclusively on the open-access **Dryad ASD Kinematic Dataset** ($N = 90$ children, 45 ASD and 45 TD).
- **No External Multi-Center Validation Claimed:** Exploratory experiments with external adult datasets (such as Move4AS) were **completely excluded and discarded** from this study to avoid cross-age and cross-protocol clinical confounds. External multi-center validation remains an open limitation.
- **Pre-Registered Locked Protocol:** All reported headline statistics are anchored to the pre-registered **Statistical Analysis Plan (SAP v1.0, commit `55bdce4`)** with a frozen 23-subject held-out test split (`splits/splits_dryad_v2_dedup.json`) evaluated across 20 independent random seeds.
- **Contiguous Block Routing (Block-ESG):** The primary method routes **contiguous 15-frame blocks** (~500 ms motor primitives). Frame-level unconstrained gating was evaluated as an inferior ablation arm (A3) that breaks temporal attention coherence ($r = 0.054$ vs. $r = 0.822$ for Block-ESG).
- **Platt Temperature Calibration:** Post-hoc calibration ($\hat{p} = \sigma(\text{logit} / T)$) is fitted strictly on out-of-fold validation logits ($T = 1.63$ for A1), reducing Expected Calibration Error (ECE) from $0.213 \rightarrow 0.173$ without test data leakage.

---

## 🏗️ System Architecture & Data Flow

```
Monocular RGB Video (.mp4 / .avi)
        │
        ▼  MediaPipe Pose (33 2D Landmarks, T = 300 frames)
Hip-Centering & Inter-Shoulder Scale Normalization
        │
        ▼  Normalized Kinematic Sequence: (B, 300, 33, 2)
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ SpatialEncoder (Per-frame MLP: pos + vel + acc -> D_c = 198) ──► Tokens: (B, 300, 128)      │
│                                                                        │                    │
│ MicrokineticEncoder (Parallel Conv1D: k=1, 3, 5 + GroupNorm)           │                    │
│   │                                                                    │                    │
│   ▼ Saliency Gate (Linear 96 -> 48 -> 1)                               │                    │
│   │                                                                    │                    │
│   ▼ Block-ESG Routing: Groups into 20 blocks of L=15 frames            │                    │
│     Selects Top-M=8 Contiguous Blocks (120 frames total)               │                    │
│   │                                                                    │                    │
│   ▼ Selected Tokens: (B, 120, 128) ◄───────────────────────────────────┘                    │
│                                                                                             │
│ Temporal Event Transformer (1 Layer, 4 Heads, d = 128, Sinusoidal Positional Encoding)      │
│   │                                                                                         │
│   ▼ Mean-pooling across active tokens -> Linear(128 -> 64 -> 1)                             │
│ Raw Logit                                                                                   │
│   │                                                                                         │
│   ▼ Platt Temperature Scaling: logit_cal = logit / T (fitted on validation set, T = 1.63)    │
│ Calibrated P(ASD) ∈ [0, 1]                                                                  │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Module Reference

| Component | File | Function |
|---|---|---|
| `SpatialEncoder` | `src/model.py` | Computes per-frame position, velocity ($\times 10$), and acceleration ($\times 5$) streams with LayerNorm |
| `MicrokineticEncoder` | `src/model.py` | Multi-scale temporal feature extractor (Conv1D $k \in \{1, 3, 5\}$ with GroupNorm) |
| `EventSaliencyGate` (Block-ESG) | `src/model.py` | Groups sequence into 20 blocks ($L=15$), scores saliency, and routes top-$M=8$ contiguous segments |
| `TemporalEventTransformer`| `src/model.py` | Lightweight Transformer encoder with absolute sinusoidal positional encoding |
| `ASDMotionModel` | `src/model.py` | End-to-end architecture with ablation flags (`use_gate`, `use_transformer`) |
| `PlattScaler` | `src/calibration.py` | Post-hoc temperature and bias recalibration preserving AUC rank order |
| `compute_bilateral_asymmetry` | `src/asymmetry.py` | Descriptive bilateral movement difference analysis (explicitly non-biomarker) |
| `PACEASDPredictor` | `src/inference_api.py` | High-level Python API for single-sample inference from video or `.npy` |

---

## 📊 Benchmark Results & Manuscript Findings

All results reflect the pre-registered 20-seed protocol evaluated on the locked 23-subject held-out test partition (`splits/splits_dryad_v2_dedup.json`). Values represent **mean ± standard deviation across 20 independent seeds** ($\text{seed} \in [42, 61]$), with 3 cross-validation fold models trained per seed ($3 \times 20 = 60$ checkpoints per deep architecture).

### Primary Comparative Benchmark (Test Partition, $N = 23$)

| Model Family | Model / Ablation Variant | ROC AUC | Accuracy | Sensitivity | Specificity | F1-Score | ECE | Collapse Rate |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Primary Model** | **A1: Full PACE-ASD ($L=15, M=8$)** | **0.836 ± 0.026** | **0.733 ± 0.040** | **0.814 ± 0.077** | **0.660 ± 0.072** | **0.740 ± 0.052** | **0.173 ± 0.026** | **0 / 20 (0%)** |
| *Closest Peer* | MTC-Former (Zhu et al., 2025) | 0.778 ± 0.024 | 0.625 ± 0.031 | 0.794 ± 0.056 | 0.469 ± 0.067 | 0.668 ± 0.029 | 0.257 ± 0.029 | 0 / 20 (0%) |
| *Ablation Arm* | A2: No-Block-ESG (Dense Transformer) | 0.816 ± 0.028 | 0.697 ± 0.028 | 0.892 ± 0.050 | 0.518 ± 0.072 | 0.738 ± 0.022 | 0.182 ± 0.027 | 0 / 20 (0%) |
| *Ablation Arm* | A3: Frame-Gate ($L=1, M=120$)* | 0.857 ± 0.039 | 0.739 ± 0.048 | 0.827 ± 0.078 | 0.658 ± 0.080 | 0.750 ± 0.058 | 0.193 ± 0.030 | 0 / 20 (0%) |
| *Ablation Arm* | A4: No-Transformer (Linear Head) | 0.809 ± 0.033 | 0.667 ± 0.033 | 0.750 ± 0.092 | 0.592 ± 0.089 | 0.672 ± 0.055 | 0.196 ± 0.037 | 0 / 20 (0%) |
| *Literature DL* | Kinematic CNN-LSTM | 0.806 ± 0.031 | 0.736 ± 0.045 | 0.873 ± 0.051 | 0.610 ± 0.074 | 0.764 ± 0.040 | 0.151 ± 0.015 | 0 / 20 (0%) |
| *Literature DL* | Stacked LSTM | 0.820 ± 0.032 | 0.696 ± 0.033 | 0.756 ± 0.096 | 0.642 ± 0.110 | 0.700 ± 0.050 | 0.140 ± 0.031 | 0 / 20 (0%) |
| *Literature DL* | MS-G3D (Liu et al., 2020) | 0.652 ± 0.050 | 0.540 ± 0.048 | 0.670 ± 0.141 | 0.421 ± 0.201 | 0.557 ± 0.054 | 0.106 ± 0.021 | 10 / 20 (50%) |
| *Literature DL* | MS-G3D + ConvNeXt | 0.822 ± 0.031 | 0.720 ± 0.048 | 0.774 ± 0.094 | 0.669 ± 0.117 | 0.710 ± 0.070 | 0.194 ± 0.026 | 0 / 20 (0%) |
| *Literature DL* | SkelFormer (Yan et al., 2026) | 0.703 ± 0.075 | 0.633 ± 0.075 | 0.515 ± 0.168 | 0.740 ± 0.177 | 0.520 ± 0.152 | 0.142 ± 0.071 | 2 / 20 (10%) |
| *Literature DL* | STTS (Zunino et al., 2018) | 0.714 ± 0.074 | 0.646 ± 0.066 | 0.486 ± 0.180 | 0.793 ± 0.162 | 0.499 ± 0.162 | 0.147 ± 0.046 | 1 / 20 (5%) |
| *Classical ML* | MediaPipe + Random Forest | 0.819 ± 0.011 | 0.675 ± 0.021 | 0.936 ± 0.023 | 0.436 ± 0.034 | 0.734 ± 0.016 | 0.184 ± 0.016 | 0 / 20 (0%) |
| *Classical ML* | MediaPipe + XGBoost | 0.825 ± 0.008 | 0.736 ± 0.011 | 0.964 ± 0.012 | 0.528 ± 0.018 | 0.778 ± 0.008 | 0.222 ± 0.015 | 0 / 20 (0%) |

*\*Important Architectural Finding regarding A3:* While frame-level gating (A3) exhibits marginally higher raw discrimination metrics, its mechanistic attention coherence completely collapses ($r = 0.054$ vs. $r = 0.822$), selecting disconnected, uninterpretable frame tokens. Contiguous 15-frame blocks (A1) are required for interpretable temporal routing.

### Statistical Significance vs. Primary Architecture Comparator (MTC-Former)
Paired two-sided Wilcoxon signed-rank tests across matched random seed pairs ($N = 20$, with Bonferroni correction $\alpha_{\text{adj}} = 0.05 / 6 = 0.0083$):
- **ROC AUC:** $\Delta = +0.0583$ [95% CI: $+0.0437, +0.0724$], $p = 0.0000$, Cohen's $d = +1.704$ (**PACE-ASD significantly superior**)
- **Specificity:** $\Delta = +0.1903$ [95% CI: $+0.1402, +0.2306$], $p = 0.0000$, Cohen's $d = +1.770$ (**PACE-ASD significantly superior**)
- **Accuracy:** $\Delta = +0.1087$ [95% CI: $+0.0877, +0.1297$], $p = 0.0000$, Cohen's $d = +2.240$ (**PACE-ASD significantly superior**)
- **F1-Score:** $\Delta = +0.0720$ [95% CI: $+0.0457, +0.0945$], $p = 0.0003$, Cohen's $d = +1.220$ (**PACE-ASD significantly superior**)
- **Calibration (ECE):** $\Delta = -0.0833$ [95% CI: $-0.0995, -0.0657$], $p = 0.0000$, Cohen's $d = -2.104$ (**PACE-ASD significantly superior**)

---

## 🔍 Core Methodological Innovations

### 1. Contiguous Block Selection vs. Unconstrained Frame Gating
A core architectural finding of the manuscript is the necessity of **block-level contiguity**:
- **PACE-ASD (A1, $L=15, M=8$):** Groups the 300 frames into 20 candidate contiguous segments of $L=15$ frames (~500 ms motion primitives at 30 fps), routing the top $M=8$ blocks (120 frames total) to the Transformer.
- **Frame-Level Gating Ablation (A3, $L=1, M=120$):** Scores individual frames and routes arbitrary disconnected frame tokens.
- **The Coherence Audit:** In A1, the gate saliency scores correlate strongly with Transformer self-attention density ($r = 0.822$, Spearman $\rho = 0.801$). In A3, this cross-check completely breaks down ($r = 0.054, \rho = 0.050$). Unconstrained frame routing selects isolated, discontinuous frames that lack atomic kinematic context. Contiguous 15-frame blocks ground attention in coherent sub-movements, ensuring structural interpretability.

### 2. Platt Temperature Scaling
In the manuscript, post-hoc probability recalibration is conducted via **Platt Temperature Scaling**:
$$\hat{p} = \sigma\left(\frac{\text{logit}}{T} + b\right)$$
- The positive temperature parameter $T > 0$ and optional intercept $b$ are fitted strictly on out-of-fold validation logits via L-BFGS ($T = 1.63$ for A1).
- No test-set data or labels are exposed during calibration fitting.
- Recalibration reduces test ECE from $0.213 \rightarrow 0.173$ without altering prediction rank order or ROC AUC.

### 3. Pediatric Kinematic Alignment
Gradient $\times$ Input attribution reveals that PACE-ASD decisions are driven primarily by:
- **Kinematic Streams:** Acceleration features account for $>72\%$ of attribution mass across variants (acceleration $72.9\%$, position $17.6\%$, velocity $9.6\%$).
- **Body Regions:** Atypical movement signals concentrate in head ($36.9\%$) and arm ($32.4\%$) kinematics, aligning directly with pediatric motor development literature on atypical upper-body stereotypic movement.

---

## 📁 Cohort Partitioning & Forensic Deduplication

The experiment is conducted exclusively on the open-access **Dryad ASD Kinematic Dataset**:
> **Aljubouri, A. A., Hadi, I., & Rajihy, Y. (2020).** *Three Dimensional Dataset Combining Gait and Full Body Movement of Children with Autism Spectrum Disorders Collected by Kinect v2 Camera.* Dryad Digital Repository. [doi:10.5061/dryad.s7h44j150](https://doi.org/10.5061/dryad.s7h44j150).

### Forensic MD5 Checksum Deduplication
A forensic MD5 audit revealed **5 pairs of byte-identical TD video files** in the public deposit. To eliminate train-test data leakage, one duplicate recording from each pair was quarantined in `processed/removed_duplicates/`:
- `td_39` (byte-identical duplicate of `td_17`)
- `td_5` (byte-identical duplicate of `td_22`)
- `td_4` (byte-identical duplicate of `td_23`)
- `td_7` (byte-identical duplicate of `td_24`)
- `td_50` (byte-identical duplicate of `td_26`)

### Frozen Cohort Partitions (`splits/splits_dryad_v2_dedup.json`)
- **Main Clean Cohort ($N = 90$ children, 45 ASD + 45 TD):**
  - **Held-Out Test Set ($N = 23$ subjects, 11 ASD + 12 TD):** Exactly ~25% of the cohort, locked prior to model fitting (`splits/splits_dryad_v2_dedup.json`).
  - **Train/Validation Set ($N = 67$ subjects, 34 ASD + 33 TD):** Partitioned into 3 balanced stratified folds for cross-validation and calibration fitting.
- **Supplementary Protocol-Shift Cohort ($N = 9$ severe-ASD subjects):** Never included in train, validation, or test partitions; evaluated solely in supplementary sensitivity analyses.

---

## 🚀 Installation & Quick Start

### 1. Environment Setup

```bash
git clone https://github.com/neuro-paradigm/PACE-ASD.git
cd PACE-ASD
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Linux/macOS:
source .venv/bin/activate

# Install dependencies (CPU-only is sufficient for inference and testing)
pip install -r requirements.txt

# Verify environment
python src/verify.py
```

*For GPU-accelerated training (CUDA 12.1):*
```bash
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. Run Comprehensive Unit Tests (37/37 Passing)
All unit tests use synthetic data fixtures and run in <5 seconds without requiring dataset downloads:
```bash
python -m pytest tests/ -v
```

### 3. Run Single-Sample Inference

**From pre-extracted features (no MediaPipe required):**
```bash
python scripts/infer.py \
    --input_npy processed/features/asd_1.npy \
    --checkpoint models/A1/fold1_seed42.pt \
    --config configs/inference.yaml \
    --output outputs/example_asd1
```

**From raw video (`.mp4`, `.avi`):**
```bash
python scripts/infer.py \
    --input path/to/video.mp4 \
    --checkpoint models/A1/fold1_seed42.pt \
    --config configs/inference.yaml \
    --output outputs/my_video_output
```

**Python Programmatic API:**
```python
import sys
sys.path.insert(0, 'src')
from inference_api import PACEASDPredictor

predictor = PACEASDPredictor(
    checkpoint='models/A1/fold1_seed42.pt',
    config='configs/inference.yaml'
)
result = predictor.predict('processed/features/asd_1.npy')

print(f"Prediction: {result['prediction']}")
print(f"Calibrated P(ASD): {result['calibrated_probability']:.4f}")
print(f"Body-Region Attribution: {result['body_region_attribution']}")
```

### 4. Reproduce Benchmark Statistics & Wilcoxon Tests
```bash
# Evaluate pre-trained A1 checkpoints across all folds/seeds:
python scripts/evaluate.py --model_id A1

# Compute headline summary table and collapse statistics:
python scripts/compute_stats.py

# Compute paired Wilcoxon signed-rank tests with Bonferroni correction:
python scripts/wilcoxon_test.py --results_dir results
```

### 5. Generate Multi-Panel Case Study Visualizations
```bash
python scripts/generate_case_study.py \
    --clip_id asd_1 \
    --checkpoint models/A1/fold1_seed42.pt \
    --config configs/inference.yaml \
    --output outputs/case_study_asd1
```

---

## 📦 Output Artifacts Reference

Each inference execution writes structured, reproducible outputs:
- **`result.json`:** Raw logit, raw probability, calibrated probability $P(\text{ASD})$, binary classification, threshold, runtime, and summary metadata.
- **`kinematics.npz`:** Full $(300, 33, 2)$ position, velocity ($\times 10$), and acceleration ($\times 5$) coordinate tensors.
- **`selected_events.json`:** Block-ESG selected contiguous 15-frame blocks, exact frame ranges, and all candidate block saliency scores.
- **`attribution.json`:** Gradient $\times$ Input attributions decomposed across 4 body regions (Head, Arms, Torso, Legs) and 3 kinematic streams (position, velocity, acceleration).
- **`visualizations/`:**
  - `kinematics.png`: Trajectory plots for position, velocity, and acceleration.
  - `event_saliency.png`: Bar plot showing Block-ESG saliency across all 20 blocks.
  - `evidence_summary.png`: 4-panel diagnostic figure visualizing trajectories, selected blocks, and regional attributions.

---

## 📂 Repository Organization

```
PACE-ASD/
├── configs/
│   ├── config.yaml                    # Master training & cross-validation configuration
│   ├── inference.yaml                 # Inference parameters, thresholds, and output flags
│   └── evaluation.yaml                # Model evaluation configurations (A1–A4)
├── docs/
│   ├── installation.md                # System requirements & setup guide
│   ├── usage.md                       # Comprehensive CLI and Python API usage
│   ├── architecture.md                # Mathematical architecture details
│   ├── inference.md                   # Complete output schema reference
│   ├── reproducibility.md             # Benchmark reproduction protocol
│   ├── software_comparison.md         # Comparison matrix vs. literature tools
│   ├── troubleshooting.md             # Common errors and solutions
│   └── SUBMISSION_READINESS.md        # BMC submission audit matrix (22 items)
├── src/
│   ├── model.py                       # ASDMotionModel, SpatialEncoder, Microkinetic, Block-ESG
│   ├── inference_api.py               # PACEASDPredictor programmatic inference interface
│   ├── asymmetry.py                   # Descriptive bilateral movement asymmetry utility
│   ├── calibration.py                 # PlattScaler positive-temperature scaling
│   ├── interpretability.py            # Gradient x Input & gate-attention coherence
│   ├── preprocess.py                  # MediaPipe pose extraction & scale normalization
│   ├── dataset.py                     # Sequence loaders & sequence mixup augmentation
│   ├── train.py                       # Training loop, early stopping, Platt fitting
│   ├── ablation.py                    # Multi-seed CV ablation runner
│   ├── baselines.py                   # 14 literature deep learning & ML baselines
│   ├── metrics.py                     # Evaluation metrics (AUC, ECE, F1, CI, collapse guards)
│   ├── report.py                      # PDF report generator
│   └── verify.py                      # Checkpoint and data integrity audits
├── scripts/
│   ├── infer.py                       # Single-sample inference CLI (video + .npy)
│   ├── evaluate.py                    # Checkpoint evaluation reproducer
│   ├── generate_case_study.py         # 8-panel case study generator
│   ├── compute_stats.py               # Benchmark table calculator
│   ├── wilcoxon_test.py               # Statistical significance testing
│   ├── audit_clip_lengths.py          # Frame count and duration audit
│   └── create_reviewer_package.py     # Reviewer release bundle packaging script
├── tests/                             # 9 unit & integration test files (37/37 passing)
├── processed/
│   ├── features/                      # Preprocessed landmark arrays (*.npy, 105 files)
│   ├── removed_duplicates/            # Quarantined duplicate recordings from MD5 audit
│   └── labels.csv                     # Cohort metadata and diagnostic labels
├── models/
│   └── A1/                            # 60 checkpoints for PACE-ASD (fold1-3 x seed42-61)
├── splits/
│   └── splits_dryad_v2_dedup.json     # Frozen subject-level train/test partitions
├── results/                           # Master results, per-seed JSONs, Wilcoxon outputs
├── examples/                          # Example inputs and verified expected outputs
├── release/                           # Reviewer bundle & quick-start guide
├── requirements.txt                   # Pinned dependency specification
├── pyproject.toml                     # Build system configuration
├── CITATION.cff                       # Citation metadata with full author details
├── CHANGELOG.md                       # Version history
├── STATISTICAL_ANALYSIS_PLAN.md       # Pre-registered locked Statistical Analysis Plan (v1.0)
└── README.md                          # Main repository documentation
```

---

## ⚠️ Limitations & Ethical Considerations

1. **Research Software Only:** PACE-ASD is an academic research software tool, not a cleared medical device. It does not provide medical diagnoses or replace clinical evaluations.
2. **Single-Site Pediatric Cohort:** Evaluated solely on the Dryad dataset ($N=90$ children). Generalization across diverse recording devices, ambient lighting conditions, and broader age groups has not been clinically validated.
3. **No External OOD Generalization Claimed:** Due to clinical confounds in external adult datasets, no out-of-distribution generalization is claimed. Multi-center pediatric validation is an essential future milestone.
4. **2D Landmark Trajectories:** Coordinates are extracted in 2D ($x, y$); out-of-plane rotational movements are not reconstructed in 3D.
5. **Descriptive Asymmetry:** Bilateral movement asymmetry metrics are descriptive kinematic measures and are not validated clinical biomarkers.

---

## 📖 Citation

If you use PACE-ASD in your research, please cite this software repository:

```bibtex
@software{paceasd2026,
  title  = {{PACE-ASD}: Pose-Aware Contiguous Event Saliency-Gated Transformer
             for Markerless Monocular Video-Based {ASD} Screening Research},
  author = {Puppala, Sireesha and {Kasi}, {Vamshi Mohan} and
             Tanuku, {VVS Sai Tejesh} and Kota, Preetham and
             Annabathula, {Yuva Dhanvanth}},
  url    = {https://github.com/neuro-paradigm/PACE-ASD},
  year   = {2026}
}
```

See [`CITATION.cff`](CITATION.cff) for machine-readable citation metadata.

---

## 👥 Authors & Contact

| Author | Role | Affiliation | ORCID |
|---|---|---|---|
| **Sireesha Puppala** (Corresponding Author) | Conceptualization, Methodology, Supervision | Dept. of CSE, KMIT; NeuroParadigm Pvt. Ltd., Hyderabad | [0009-0008-3984-9665](https://orcid.org/0009-0008-3984-9665) |
| **Kasi Vamshi Mohan** (Co-First Author) | Training infrastructure, statistical analysis | Dept. of CSE (AI&ML), KMIT; NeuroParadigm Pvt. Ltd. | [0009-0005-7715-8388](https://orcid.org/0009-0005-7715-8388) |
| **Tanuku VVS Sai Tejesh** (Co-First Author) | Architecture formulation, Block-ESG mathematics | Dept. of CSE (AI&ML), KMIT; NeuroParadigm Pvt. Ltd. | [0009-0003-9845-8190](https://orcid.org/0009-0003-9845-8190) |
| **Preetham Kota** (Co-First Author) | Baseline re-implementation and evaluation | Dept. of CSE (AI&ML), KMIT; NeuroParadigm Pvt. Ltd. | [0009-0003-6753-5608](https://orcid.org/0009-0003-6753-5608) |
| **Annabathula Yuva Dhanvanth** (Co-First Author) | MediaPipe extraction, MD5 audit | Dept. of CSE (AI&ML), KMIT; NeuroParadigm Pvt. Ltd. | [0009-0003-7401-1394](https://orcid.org/0009-0003-7401-1394) |

**Corresponding Author:** Sireesha Puppala ([sireesha@neuroparadigm.in](mailto:sireesha@neuroparadigm.in))
