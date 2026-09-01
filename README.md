# PACE-ASD: Pose-Aware Contiguous Event Saliency Gate for ASD Screening

**Markerless video-based autism spectrum disorder (ASD) screening via skeleton pose sequences and a Block-Level Event Saliency Gate (Block-ESG).**

> This repository accompanies a manuscript under review at *Information Sciences with Applications* (Elsevier). Results are frozen pending re-training on the deduplicated cohort. See [Dataset](#dataset) for deduplication details.

---

## Overview

PACE-ASD classifies 2-D skeleton pose sequences extracted from standard RGB video into ASD / typically-developing (TD).
The core contribution is the **Block-ESG**: a lightweight differentiable gate that selects the *M = 8* most kinematically salient **contiguous 15-frame blocks** (120 / 300 frames total) before a single-layer, four-head Transformer encoder.

```
RGB video → MediaPipe pose → (300, 33, 2) keypoints
    → Spatial Encoder (128-d per frame)
        → Microkinetic Encoder (3× Conv1D → 96-d)
            → Block-ESG (300 → 120 tokens)
                → Temporal Transformer (1L / 4H)
                    → Classifier + Platt calibration
                        → P(ASD)
```

| Variant | Description |
|---|---|
| **A1** | Full PACE-ASD (Block-ESG + Transformer) |
| A2 | No gate — all 300 frames to Transformer |
| A3 | Frame-granularity gate (L=1, M=120) |
| A4 | No Transformer — linear head on gate output |

---

## Key Results (frozen, 20 seeds × 3-fold CV, n = 90)

| Model | AUC | Specificity | Collapse rate |
|---|---|---|---|
| **A1 (PACE-ASD)** | 0.858 | **0.783** | **0/20** |
| MTC-Former | 0.874 | 0.728 | 3/20 |

- Primary finding: A1 is **non-inferior** to MTC-Former on AUC (Δ = −0.016, p = 0.083, paired Wilcoxon).
- A1 **significantly outperforms** MTC-Former on specificity (Δ = +0.055, p = 0.004, Bonferroni-corrected).
- Gate ↔ Transformer coherence cross-check: Pearson r = 0.625, per-subject r = 0.783.

> ⚠️ These result numbers are from the pre-deduplication run (n=100). Re-training on the deduplicated n=90 cohort is required before final submission. Numbers will be updated here after re-training.

---

## Dataset

**Al-Jubouri, A. A.; Hadi, I.; Rajihy, Y. (2020).** *Three Dimensional Dataset Combining Gait and Full Body Movement of Children with Autism Spectrum Disorders Collected by Kinect v2 Camera.* Dryad. [doi:10.5061/dryad.s7h44j150](https://doi.org/10.5061/dryad.s7h44j150)

This study uses the **color video** component (Samsung Note 9 rear camera). Kinect v2 skeleton files were not used.

### Deduplication

An MD5 audit of all 110 processed feature files found **5 pairs of byte-identical TD recordings** in the Dryad deposit — duplicate source videos assigned to different subject IDs.
One from each pair was moved to `processed/removed_duplicates/`:

| Kept | Removed (duplicate of kept) |
|---|---|
| td_17 | td_39 |
| td_22 | td_5 |
| td_23 | td_4 |
| td_24 | td_7 |
| td_26 | td_50 |

After deduplication: **95 unique recordings (50 ASD + 45 TD)**.

### Cohort assignment (seed = 42)

| Partition | Subjects | n |
|---|---|---|
| **Main cohort** | 45 ASD + 45 TD | **90** |
| **Supplement** | 5 regular ASD + 9 severe-ASD | **14** |

Post-deduplication MD5 check: **zero train/test overlaps** ✅



## Reproduction

### 1. Environment

```bash
git clone https://github.com/neuro-paradigm/PACE-ASD.git
cd PACE-ASD
pip install -r requirements.txt
```

Python 3.11, CUDA 12.1, Windows 11 / Ubuntu 22.04. Full pinned versions in [`requirements.txt`](requirements.txt).

### 2. Obtain the dataset

Download `Dataset-2.rar` from [Dryad](https://doi.org/10.5061/dryad.s7h44j150) and extract to `D:/dryad/` (or edit `data.raw_dir` in `configs/config.yaml`).

### 3. Preprocess

```bash
python src/preprocess.py --raw_dir "D:/dryad" --out_dir processed
```

Produces `processed/features/*.npy` — shape `(300, 33, 2)` float32.

### 4. Run full ablation (A1–A4, 20 seeds × 3 folds)

```bash
python src/ablation.py --config configs/config.yaml
```

Results written to `results/ablation_results.csv`. See [`REPRODUCE.md`](REPRODUCE.md) for step-by-step instructions including single-fold sanity checks.

### 5. Verify frozen results

```bash
python "ISWA JOURNAL submission/compute_stats.py"
```

Reproduces all paired Wilcoxon statistics reported in the manuscript.

---

## Repository Structure

```
PACE-ASD/
├── configs/config.yaml             # Authoritative hyperparameters
├── src/
│   ├── model.py                    # ASDMotionModel, all variants
│   ├── train.py                    # Training loop, cross-validation
│   ├── preprocess.py               # MediaPipe pose extraction
│   ├── ablation.py                 # Runs all ablation variants
│   └── audit.py                    # Dataset audit
├── processed/
│   ├── features/                   # *.npy keypoint arrays (not tracked in git)
│   └── labels.csv                  # Subject-level metadata
├── splits/splits_dryad_only_v1.json # Frozen train/test/fold split
├── results/
│   ├── ablation_results.csv        # Main comparison table
│   └── supplement_results.csv      # Severe-ASD supplement
├── ISWA JOURNAL submission/        # LaTeX manuscript + supplementary
├── REPRODUCE.md                    # Detailed reproduction guide
├── PREPROCESS_SPEC.md              # Preprocessing specification
└── audit_report.txt                # Dataset audit log
```

---

## Model Size

With the current config (`spatial_dim=128`, `conv1d_channels=32`, `transformer_layers=1`):

| Variant | Parameters |
|---|---|
| A1 (full PACE-ASD) | 227,811 |
| A2 (no gate) | 218,402 |
| A3 (frame gate) | 227,811 |
| A4 (no Transformer) | 143,715 |

---

## Citation

> Puppala, S.; Mohan, K. V.; Tejesh, T. V. V. S. S.; Kota, P.; Dhanvanth, A. Y. (2026).
> *PACE-ASD: A Pose-Aware Contiguous Event Saliency Gate for Markerless Autism Screening.*
> *Information Sciences with Applications.* (under review)

---

## License

Code: MIT License. Dataset: CC0 1.0 Universal (Dryad, original depositors). See [LICENSE](LICENSE) for details.

---

## Contact

Sireesha Puppala — [sireesha@neuroparadigm.in](mailto:sireesha@neuroparadigm.in)  
NeuroParadigm Pvt. Ltd. / Keshav Memorial Institute of Technology, Hyderabad, India.
