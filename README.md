# PACE-ASD: Pose Analysis for Clinical Estimation — Autism Spectrum Disorder

<p align="center">
  <img src="MDPI_template_Chicago/pipeline.jpg" alt="PACE-ASD Pipeline" width="80%"/>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.8%2B-blue"/>
  <img alt="Framework" src="https://img.shields.io/badge/framework-PyTorch-orange"/>
  <img alt="Status" src="https://img.shields.io/badge/status-research--prototype-yellow"/>
  <img alt="AUC" src="https://img.shields.io/badge/AUC-0.9444-brightgreen"/>
</p>

---

> **⚠️ Research Prototype — Not for Clinical Use**
> This system is a feasibility demonstration. It has not been validated in a prospective clinical trial and must not be used as a diagnostic tool.

---

## Overview

PACE-ASD is a pose-driven, privacy-preserving framework for preliminary AI-assisted Autism Spectrum Disorder (ASD) screening from ordinary RGB video. It recovers full-body skeletal landmarks using MediaPipe Pose and processes them through a hybrid pipeline that combines kinematic feature extraction, sparse behavioral event selection, transformer-based temporal reasoning, and calibrated probability estimation.

The framework is designed to run on consumer-grade hardware and requires no specialized sensors or clinical equipment.

---

## Key Results

Evaluated on a completely held-out test set of **451 clips from 35 subjects**, withheld before any model development:

| Metric | Value |
|---|---|
| Accuracy | 85.71% |
| AUC | 0.9444 |
| F1 Score | 0.8649 |
| Sensitivity | 88.89% |
| Specificity | 82.35% |
| Expected Calibration Error (ECE) | 0.1092 |

On an **independent real-world external cohort** of clinically confirmed ASD subjects collected under unconstrained conditions, the model correctly identified approximately **80%** of cases without any retraining.

---

## Pipeline Architecture

The PACE-ASD pipeline runs in five stages:

1. **Skeletal Pose Extraction & Preprocessing** — MediaPipe Pose extracts 33 anatomical keypoints per frame; trajectories are smoothed with a Savitzky–Golay filter and normalized for body size and camera tilt.
2. **Kinematic Feature Extraction** — Per-joint velocity and acceleration are computed, yielding a 198-dimensional feature vector per frame encoding posture, motion, and dynamics.
3. **Sparse Behavioral Event Selection** — A multi-scale convolutional front-end followed by a learned saliency gate scores each frame for behavioral relevance; only the top-K frames are retained, reducing sequence length from T to K ≪ T.
4. **Temporal Context Aggregation** — Retained tokens are enriched with sinusoidal positional encodings and pose-quality confidence embeddings, then processed by a bidirectional multi-head self-attention transformer. Attention complexity drops from O(T²) to O(K²).
5. **Calibrated Risk Estimation** — A binary classifier produces an ASD-risk logit; Platt scaling calibrates this into an interpretable probability aligned with empirical classification frequencies.

---

## Datasets

| Dataset | Role | Subjects | Clips |
|---|---|---|---|
| [Dryad Digital Repository](https://datadryad.org/dataset/doi:10.5061/dryad.s7h44j150) | Primary training source | — | — |
| [Move4AS (Paulo et al., *Scientific Data* 2025)](https://doi.org/10.1038/s41597-025-05313-0) | Supplementary training source | 34 | — |
| In-house external cohort | External validation (ASD only) | — | — |
| **Combined corpus** | **Total** | **138** | **2,573** |

All partitioning was performed strictly at the **subject level** to prevent data leakage. A 25% subject-level holdout (35 subjects, 451 clips) was reserved before any model development.

---

## Repository Structure

```
PACE-ASD/
├── data/
│   └── README.md               # Instructions for downloading Dryad and Move4AS datasets
├── preprocessing/
│   ├── pose_extraction.py      # MediaPipe Pose landmark extraction
│   ├── normalization.py        # Hip-centering, scaling, rotation alignment
│   └── quality_masking.py      # Per-frame quality scoring and masking
├── model/
│   ├── encoder.py              # Residual MLP spatial encoder
│   ├── multiscale_conv.py      # Multi-scale 1D convolutional front-end
│   ├── saliency_gate.py        # Learned saliency scoring and top-K selection
│   ├── transformer.py          # Bidirectional multi-head self-attention module
│   └── pace_asd.py             # Full PACE-ASD model definition
├── training/
│   ├── train.py                # Training loop with subject-independent CV
│   ├── loss.py                 # Composite loss (WBCE + ranking + sensitivity@specificity)
│   └── calibration.py          # Platt scaling via L-BFGS
├── evaluation/
│   ├── evaluate.py             # Held-out test set evaluation
│   └── external_eval.py        # External cohort evaluation
├── collection_app/
│   └── README.md               # Documentation for the anonymized data collection app
├── MDPI_template_Chicago/      # LaTeX manuscript source
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/neuro-paradigm/PACE-ASD.git
cd PACE-ASD
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, PyTorch, MediaPipe, NumPy, scikit-learn, SciPy

---

## Usage

### 1. Pose Extraction

```bash
python preprocessing/pose_extraction.py \
  --input_dir /path/to/videos \
  --output_dir /path/to/landmarks
```

### 2. Training (Subject-Independent 3-Fold CV)

```bash
python training/train.py \
  --data_dir /path/to/landmarks \
  --labels /path/to/labels.csv \
  --output_dir ./checkpoints
```

### 3. Evaluation on Held-Out Test Set

```bash
python evaluation/evaluate.py \
  --checkpoint ./checkpoints/best_ensemble.pt \
  --test_dir /path/to/test_landmarks \
  --test_labels /path/to/test_labels.csv
```

---

## Ablation Summary

| Configuration | Accuracy | Δ vs Full |
|---|---|---|
| Full PACE-ASD | 85.71% | — |
| Without velocity & acceleration | 73.10% | −12.61 pp |
| Random event selection (no saliency) | 78.30% | −7.41 pp |
| Fixed positional encodings | 79.80% | −5.91 pp |
| Without Platt scaling | 85.71% | 0 pp (calibration ↑) |

---

## Acknowledgements

We thank the contributors of the [Dryad Digital Repository](https://datadryad.org/dataset/doi:10.5061/dryad.s7h44j150) dataset and the Move4AS team (Paulo et al., University of Coimbra) for making their data publicly available, and for their work in advancing open-access motor function research in autism.

---

## Disclaimer

This framework is a **research prototype** developed as a feasibility study. It is intended strictly for academic and research purposes. It has not undergone prospective clinical validation and **must not be used for diagnostic or clinical decision-making**. Any future clinical application would require rigorous prospective trials, regulatory review, and expert clinical oversight.