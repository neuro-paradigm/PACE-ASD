# PACE-ASD: Pose-Aware Contiguous Event Saliency-Gated Transformer for Markerless Autism Screening

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset: Dryad](https://img.shields.io/badge/Dataset-Dryad%20CC0-green.svg)](https://doi.org/10.5061/dryad.s7h44j150)

Official PyTorch implementation and reproduction codebase for **PACE-ASD (Pose-Aware Contiguous Event Saliency-Gated Transformer for Markerless Autism Screening)**.

---

## 📌 Overview

Early detection of Autism Spectrum Disorder (ASD) is critical for improving developmental outcomes, yet gold-standard clinical assessments (such as the ADOS-2) require specialized clinicians and suffer from prolonged diagnostic waitlists. Markerless 2D skeleton pose estimation from standard, monocular RGB video offers an accessible and non-invasive alternative for motor screening.

However, existing deep learning action recognition models (e.g., spatio-temporal GCNs and video Transformers) face two major obstacles in pediatric motor screening:
1. **Temporal Dilution:** Transient, informative atypical motor events are drowned within lengthy sequences of ordinary background movement when applying dense attention or global pooling.
2. **Catastrophic Instability:** Over-parameterized architectures frequently suffer catastrophic training collapse when trained on modest clinical cohorts ($N < 100$).

**PACE-ASD** resolves these challenges by introducing an inductive temporal bias via the **Block-Level Event Saliency Gate (Block-ESG)**:
- Instead of attending over all 300 frames or selecting arbitrary, disconnected individual frames, Block-ESG dynamically identifies and routes the **$M = 8$ most kinematically salient contiguous 15-frame blocks** (500 ms motion primitives at 30 fps; 120 frames total) to a lightweight temporal Transformer encoder.
- Provides **complete training stability** (0/20 collapsed seeds vs. up to 10/20 collapse in literature baselines).
- Enforces **mechanistic interpretability** through a two-stage coherence audit: validating that gating saliency and self-attention weights reinforce the same motor events ($r = 0.822$).

---

## 🏗️ Model Architecture & Data Flow

```
Monocular RGB Video
        │
        ▼
MediaPipe Pose (33 2-D Keypoints, T = 300 frames)
        │
        ▼ Hip-Centering & Inter-Shoulder Scale Normalization
Kinematic Skeletal Sequence (B, 300, 33, 2)
        │
        ├──► Spatial Encoder (Per-frame MLP) ──────────────► Spatial Tokens (B, 300, 128)
        │                                                           │
        └──► Microkinetic Encoder (Conv1D: pos, vel, acc)           │
                    │                                               │
                    ▼                                               │
             Salience Gate (Linear 96 -> 48 -> 1)                   │
                    │                                               │
                    ▼                                               │
             Block-ESG Pooling (L = 15 frames, M = 8 blocks)        │
                    │                                               │
                    ▼ (Top-8 Contiguous Blocks Selected)            │
             Selected Tokens (B, 120, 128) ◄────────────────────────┘
                    │
                    ▼
             Temporal Transformer (1 Layer, 4 Heads, d = 128, with Padding Mask)
                    │
                    ▼ Mean Pooling across Active Tokens
             Classification Head (MLP: 128 -> 64 -> 1)
                    │
                    ▼ Temperature Platt Scaling
             Calibrated Prediction: P(ASD) ∈ [0, 1]
```

### Architectural Ablation Arms
* **A1 (Full PACE-ASD):** Block-ESG ($L=15, M=8$) + 1-layer 4-head Temporal Transformer (223,107 params).
* **A2 (Dense Transformer / No-Block-ESG):** Direct self-attention across all 300 frames without gating (218,402 params).
* **A3 (Frame-Granularity Gate):** Unconstrained single-frame selection ($L=1, M=120$) (223,107 params).
* **A4 (No-Transformer):** Microkinetic spatial encoder + Block-ESG with linear pooling head (139,011 params).

---

## 📊 Benchmark Results

Evaluated on the **deduplicated main cohort ($N = 90$, 45 ASD + 45 TD)** with 3-fold subject-level cross-validation and 20 independent random seeds per fold (**240 total runs per arm**). Collapsed seeds ($\text{test accuracy} < 0.55$) are strictly monitored.

### Main Performance Comparison (Test Split, 20-Seed Mean $\pm$ SD)

| Model Category | Model | AUC | Accuracy | F1 | Sensitivity | Specificity | ECE | Collapsed |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **PACE-ASD Arms** | **A1 (Full PACE-ASD)** | **0.836 ± 0.026** | **0.733 ± 0.040** | **0.740 ± 0.052** | **0.814 ± 0.077** | **0.660 ± 0.072** | **0.173 ± 0.026** | **0 / 20** |
| | A2 (No-Block-ESG) | 0.816 ± 0.028 | 0.697 ± 0.028 | 0.738 ± 0.022 | 0.892 ± 0.050 | 0.518 ± 0.072 | 0.182 ± 0.027 | 0 / 20 |
| | A3 (Frame Gate, L=1) | 0.857 ± 0.039 | 0.739 ± 0.048 | 0.750 ± 0.058 | 0.827 ± 0.078 | 0.658 ± 0.080 | 0.193 ± 0.030 | 0 / 20 |
| | A4 (No-Transformer) | 0.809 ± 0.033 | 0.667 ± 0.033 | 0.672 ± 0.055 | 0.750 ± 0.092 | 0.592 ± 0.089 | 0.196 ± 0.037 | 0 / 20 |
| **Deep Learning Baselines** | MTC-Former (Zhu et al., 2025) | 0.778 ± 0.024 | 0.625 ± 0.031 | 0.668 ± 0.029 | 0.794 ± 0.056 | 0.469 ± 0.067 | 0.257 ± 0.029 | 0 / 20 |
| | Stacked LSTM | 0.820 ± 0.032 | 0.696 ± 0.033 | 0.700 ± 0.050 | 0.756 ± 0.096 | 0.642 ± 0.110 | 0.140 ± 0.031 | 0 / 20 |
| | Conv1D-BiLSTM-Attn | 0.781 ± 0.022 | 0.609 ± 0.021 | 0.674 ± 0.021 | 0.850 ± 0.048 | 0.388 ± 0.047 | 0.232 ± 0.042 | 0 / 20 |
| | Kinematic CNN-LSTM | 0.806 ± 0.031 | 0.736 ± 0.045 | 0.764 ± 0.040 | 0.873 ± 0.051 | 0.610 ± 0.074 | 0.151 ± 0.015 | 0 / 20 |
| | MS-G3D (Liu et al., 2020) | 0.652 ± 0.050 | 0.540 ± 0.048 | 0.557 ± 0.054 | 0.670 ± 0.141 | 0.421 ± 0.201 | 0.106 ± 0.021 | **10 / 20** |
| | MS-G3D + ConvNeXt | 0.822 ± 0.031 | 0.720 ± 0.048 | 0.710 ± 0.070 | 0.774 ± 0.094 | 0.669 ± 0.117 | 0.194 ± 0.026 | 0 / 20 |
| | SkelFormer (Yan et al., 2026) | 0.703 ± 0.075 | 0.633 ± 0.075 | 0.520 ± 0.152 | 0.515 ± 0.168 | 0.740 ± 0.177 | 0.142 ± 0.071 | **2 / 20** |
| | STTS (Wang et al., 2022) | 0.714 ± 0.074 | 0.646 ± 0.066 | 0.499 ± 0.162 | 0.486 ± 0.180 | 0.793 ± 0.162 | 0.147 ± 0.046 | **1 / 20** |
| | MTT (Kong et al., 2022) | 0.768 ± 0.061 | 0.668 ± 0.049 | 0.586 ± 0.139 | 0.602 ± 0.160 | 0.729 ± 0.131 | 0.130 ± 0.039 | 0 / 20 |
| | STAR (Shi et al., 2021) | 0.748 ± 0.059 | 0.638 ± 0.056 | 0.587 ± 0.109 | 0.606 ± 0.161 | 0.667 ± 0.156 | 0.137 ± 0.047 | **1 / 20** |
| **Classical ML Baselines** | MediaPipe + LR | 0.725 ± 0.000 | 0.696 ± 0.000 | 0.725 ± 0.000 | 0.818 ± 0.000 | 0.583 ± 0.000 | 0.200 ± 0.000 | 0 / 20 |
| | MediaPipe + SVM (RBF) | 0.871 ± 0.000 | 0.694 ± 0.006 | 0.755 ± 0.005 | 0.971 ± 0.012 | 0.440 ± 0.016 | 0.189 ± 0.026 | 0 / 20 |
| | MediaPipe + Random Forest | 0.819 ± 0.011 | 0.675 ± 0.021 | 0.734 ± 0.016 | 0.936 ± 0.023 | 0.436 ± 0.034 | 0.184 ± 0.016 | 0 / 20 |
| | MediaPipe + XGBoost | 0.825 ± 0.008 | 0.736 ± 0.011 | 0.778 ± 0.008 | 0.964 ± 0.012 | 0.528 ± 0.018 | 0.222 ± 0.015 | 0 / 20 |

### Key Findings & Statistical Significance
1. **Superiority over SOTA Temporal Transformer Baseline:** A1 significantly outperforms MTC-Former on AUC ($\Delta = +0.058, p < 0.001$, Cohen's $d = +1.704$), specificity ($\Delta = +0.190, p < 0.001, d = +1.770$), accuracy ($\Delta = +0.109, p < 0.001$), F1 ($\Delta = +0.072, p < 0.001$), and calibration error ECE ($\Delta = -0.083, p < 0.001$).
2. **Architectural Training Stability:** PACE-ASD arms achieve **0/20 collapse** across all seeds. In contrast, 4 of 10 deep action baselines collapse frequently (MS-G3D collapses in 50% of seeds [10/20], SkelFormer in 2/20, STTS in 1/20, and STAR in 1/20).
3. **Mechanistic Self-Consistency:** In A1, the gate saliency scores correlate strongly with Transformer self-attention density ($r = 0.822$, Spearman $\rho = 0.801$). In frame-level gating (A3), this cross-check completely collapses ($r = 0.054, \rho = 0.050$), revealing that unconstrained frame selection selects disjoint, uninterpretable frame tokens.
4. **Pediatric Kinematic Alignment:** Feature attribution converges on acceleration kinematics ($>72\%$ importance) and head/arm regions across all variants (A1, A2, A3), directly aligning with developmental motor literature.

---

## 📁 Dataset & Deduplication Audit

The experiment is conducted on the open-access **Dryad ASD Kinematic Dataset**:
> **Aljubouri, A. A., Hadi, I., & Rajihy, Y. (2020).** *Three Dimensional Dataset Combining Gait and Full Body Movement of Children with Autism Spectrum Disorders Collected by Kinect v2 Camera.* Dryad Digital Repository. [doi:10.5061/dryad.s7h44j150](https://doi.org/10.5061/dryad.s7h44j150).

### Deduplication Audit
A forensic MD5 checksum audit revealed **5 pairs of byte-identical TD video files** in the public deposit (duplicate recordings uploaded under multiple subject IDs). To prevent train-test data leakage, one subject from each identical pair was segregated into `processed/removed_duplicates/`:
- `td_39` (duplicate of `td_17`)
- `td_5` (duplicate of `td_22`)
- `td_4` (duplicate of `td_23`)
- `td_7` (duplicate of `td_24`)
- `td_50` (duplicate of `td_26`)

### Cohort Partitions
- **Main Deduplicated Cohort ($N = 90$, 45 ASD + 45 TD):** Used for all primary cross-validation and baseline benchmarking (`splits/splits_dryad_v2_dedup.json`).
- **Supplementary Protocol-Shift Cohort ($N = 14$, 5 regular ASD + 9 severe-ASD):** Held out completely from training to evaluate cross-session and protocol-shift robustness.

---

## 🚀 Quickstart & Reproduction

### 1. Environment Setup

```bash
git clone https://github.com/neuro-paradigm/PACE-ASD.git
cd PACE-ASD
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Preprocessing (Optional — Preprocessed Features are Included)
Preprocessed 2D landmark sequences (`(300, 33, 2)` float32 arrays) are already generated and tracked in `processed/features/`.
To re-extract from raw video files:
```bash
python src/preprocess.py --raw_dir "path/to/dryad" --out_dir processed
```

### 3. Reproduce Statistical Tables & Wilcoxon Tests
To immediately compute summary metrics, collapse rates, and paired Wilcoxon statistics:
```bash
# Compute comprehensive benchmark summary and paired tests vs MTC-Former:
python scripts/compute_stats.py

# Run full paired Wilcoxon tests across all models with Bonferroni correction:
python scripts/wilcoxon_test.py --results_dir results
```

### 4. Train Models from Scratch
Train a single model arm (e.g., A1, seed 0, fold 0):
```bash
python src/train.py --config configs/config.yaml --model_id A1 --seed 0 --fold 0
```

Run the complete 20-seed ablation suite:
```bash
python src/ablation.py --config configs/config.yaml
```

Run the supplementary protocol-shift evaluation:
```bash
python src/ablation.py --config configs/config.yaml --eval_supplement_only
```

### 5. Interpretability & Mechanistic Coherence Audit
Extract attention maps, kinematic attributions, and compute gate-attention coherence:
```bash
python src/interpretability.py --config configs/config.yaml --models A1 A2 A3
```

---

## 📂 Repository Organization

```
PACE-ASD/
├── configs/
│   └── config.yaml                     # Authoritative hyperparameters & architecture specs
├── src/
│   ├── model.py                        # ASDMotionModel, SpatialEncoder, Microkinetic, Block-ESG
│   ├── train.py                        # Training loops, loss formulation, Platt calibration
│   ├── ablation.py                     # Multi-seed cross-validation harness
│   ├── baselines.py                    # Classical ML and Deep Learning benchmark models
│   ├── dataset.py                      # PyTorch Dataset, padding masking, landmark loaders
│   ├── interpretability.py             # Feature attribution, attention extraction, coherence
│   ├── metrics.py                      # AUC, ECE, F1, sensitivity, specificity, collapse guards
│   ├── preprocess.py                   # MediaPipe pose extraction and scale normalization
│   └── verify.py                       # Checkpoint and data integrity audits
├── scripts/
│   ├── compute_stats.py                # Standalone evaluation table reproduction script
│   ├── wilcoxon_test.py                # Multi-model paired Wilcoxon signed-rank tests
│   ├── build_supplement_content.py     # Evaluation table builder
│   └── audit_clip_lengths.py           # Clip duration and sampling rate validation
├── processed/
│   ├── features/                       # Preprocessed keypoint arrays (*.npy)
│   ├── removed_duplicates/             # Segregated duplicate recordings from MD5 audit
│   └── labels.csv                      # Cohort metadata and diagnostic labels
├── splits/
│   └── splits_dryad_v2_dedup.json      # Frozen 3-fold subject-level train/test splits
├── results/
│   ├── A1_per_seed.json                # Per-seed metrics for A1 (PACE-ASD)
│   ├── A2_per_seed.json                # Per-seed metrics for A2 (No-Block-ESG)
│   ├── A3_per_seed.json                # Per-seed metrics for A3 (Frame Gate)
│   ├── A4_per_seed.json                # Per-seed metrics for A4 (No-Transformer)
│   ├── A5_*_per_seed.json              # Per-seed metrics for all 14 baseline models
│   ├── ablation_results.csv            # Master benchmark results table
│   └── supplement_results.csv          # Supplementary protocol-shift evaluation
├── PREPROCESS_SPEC.md                  # Detailed mathematical specification of preprocessing
├── REPRODUCE.md                        # Step-by-step reproduction instructions
├── requirements.txt                    # Pinned Python package dependencies
├── LICENSE                             # MIT Open Source License
└── README.md                           # This document
```

---

## 📄 License & Ethics

- **Code:** Licensed under the [MIT License](LICENSE).
- **Dataset:** The underlying video data is sourced from the Dryad Digital Repository ([CC0 1.0 Universal Public Domain Dedication](https://creativecommons.org/publicdomain/zero/1.0/)).
- **Ethics Statement:** This research represents a secondary computational analysis of publicly available, de-identified skeletal coordinates. No patient-identifiable data or identifiable raw video files are redistributed.

---

## 📬 Contact

For questions or inquiries regarding the project:
- **Sireesha Puppala** (Project Lead): [sireesha@neuroparadigm.in](mailto:sireesha@neuroparadigm.in)  
  *Head of Research, NeuroParadigm Pvt. Ltd. | Department of CSE, Keshav Memorial Institute of Technology, Hyderabad, India.*
