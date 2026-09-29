# PACE-ASD — Software Comparison and Workflow Analysis

> **Assessment Basis:** Assessment by the authors from each project's public documentation, code repositories, and publications (accessed 20 September 2026):
> - `✓` Documented core capability
> - `~` Partial or indirect capability
> - `–` Not documented
> - `n/a` Not applicable
> *"Not documented" does not prove absence.*

---

## 1. Capability Comparison Matrix

This matrix matches Table 1 in the *BMC Medical Informatics and Decision Making* software article:

| Capability | MediaPipe Pose | OpenPose | DeepLabCut | PYSKL | SenseToKnow | PACE-ASD |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Markerless (camera-only) monocular input | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Whole-body 2D landmark extraction | ✓ | ✓ | ✓ | ~ | – | ✓ |
| Velocity and acceleration representations | – | – | – | ~ | – | ✓ |
| Learned contiguous temporal event selection | – | – | – | – | – | ✓ |
| ASD-vs-typical classification task | – | – | – | – | ✓ | ✓ |
| Post-hoc probability calibration (Platt scaling) | – | – | – | – | – | ✓ |
| Per-subject localized temporal evidence output | – | – | – | – | ~ | ✓ |
| Deduplicated, subject-level multi-seed benchmark scripts for public ASD walking data | n/a | n/a | n/a | – | – | ✓ |
| Prospective clinical validation reported | n/a | n/a | n/a | n/a | ✓ | – |
| Source code openly available | ✓ | ✓ | ✓ | ✓ | ~ | ✓ |
| Terms of use | Apache-2.0 | Free non-commercial | LGPL-3.0 | Apache-2.0 | Research paper code | Apache-2.0 |

---

## 2. Direct Workflow Comparison: PYSKL vs. PACE-ASD

To illustrate why a dedicated medical informatics framework is needed beyond generic computer vision toolboxes, we compare the end-to-end user experience of applying **PYSKL** (Duan et al., ACM MM 2022) versus **PACE-ASD** to custom clinical walking video.

### Step-by-Step Developer Journey

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ PYSKL WORKFLOW (Generic Action Recognition Tooling)                                   │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. Framework Setup: MMCV, MMAction2, and custom CUDA C++ extensions (`mmcv-full`)      │
│ 2. Landmark Pipeline: Run external pose script -> write pickle dictionary converter    │
│ 3. Data Ingestion: Write custom Dataset class & StratifiedGroupKFold split generator   │
│ 4. Deduplication: Requires external duplicate detection (no built-in audit)           │
│ 5. Model Execution: Train action-recognition model (e.g., MS-G3D or ST-GCN)          │
│ 6. Post-Processing: Implement custom script for Platt scaling & ECE calculation        │
│ 7. Clinical Outputs: Extracts uncalibrated task logits; no temporal event localization │
└────────────────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────────────────┐
│ PACE-ASD WORKFLOW (Integrated Decision-Support Research Framework)                     │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. Zero C++ Compilation: Standard pure Python & PyTorch dependencies (`pip install`)   │
│ 2. Automated Hygiene: Run `scripts/audit_dataset.py` to catch byte-duplicate files     │
│ 3. Single-Command End-to-End Inference:                                               │
│    python scripts/infer.py --input video.mp4 --checkpoint models/A1/fold1_seed42.pt   │
│    • Automatic MediaPipe 33-landmark extraction & hip-centering / shoulder scaling     │
│    • Run-time kinematic derivation (velocity x10, acceleration x5)                     │
│    • Stage-1 Block-ESG sparse event selection (20 blocks -> top-8 contiguous events)   │
│    • Stage-2 Temporal Transformer inference                                            │
│    • Applied fold-fitted Platt calibration (e^{θ} * logit + b)                         │
│ 4. Structured Multi-Modal Evidence Artifacts:                                          │
│    `result.json`, `kinematics.npz`, `selected_events.json`, `attribution.json`, PNGs   │
│ 5. Interactive Prototyping: `jupyter notebook examples/demo_inference.ipynb`          │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Detailed Aspect Comparison

| Aspect | PYSKL Workflow | PACE-ASD Workflow | Practical Impact |
|:---|:---|:---|:---|
| **Installation** | Requires building `mmcv-full` with matching CUDA toolkits; platform-dependent C++ build constraints. | Pure Python/PyTorch wheel installation via `requirements.txt`. | PACE-ASD installs in minutes on standard Windows/Linux PCs without GPU compilers. |
| **Input Flexibility** | Accepts custom pickle dictionaries formatted for action benchmarks. | Accepts raw video files (`.mp4`, `.avi`, `.mov`) or pre-extracted `.npy` landmarks. | Non-technical researchers can run inference directly without pose estimation libraries. |
| **Data Hygiene** | No duplicate detection; identical recordings can cross split boundaries unnoticed. | Built-in MD5 audit quarantines duplicate recordings before split creation. | Prevents silent train/test leakage and invalid test metrics. |
| **Clinical Calibration** | Outputs uncalibrated classification logits. | Integrates out-of-fold Platt calibration, optimizing ECE for clinical risk modeling. | Enables calibrated risk stratification rather than raw decision thresholds. |
| **Temporal Evidence** | Dense sequence processing without temporal event localization. | Block-ESG identifies contiguous 15-frame (500 ms) events and evaluates gate–attention coherence. | Clinicians and researchers can visually audit which sub-movements triggered the screen. |
| **Runtime Efficiency** | Heavy 3D graph convolutions require multi-GB GPU VRAM. | Compact 223K-parameter model runs in ~7.2 ms on CPU and ~3.7 ms on GPU (<14 MB VRAM). | Runs privately and securely on standard clinical workstation laptops. |

---

## 3. MS-G3D Baseline Implementation and Benchmark Reproduction

### Implementation Details
The **MS-G3D** (Liu et al., CVPR 2020) baseline evaluated in Section 3.2 and reported in Table 4 was **reimplemented natively in PyTorch within PACE-ASD** (`src/model.py`), rather than executed as an external PYSKL subprocess. 

This design choice was deliberate:
1. **Methodological Parity:** It ensures that MS-G3D receives the exact same frozen preprocessing inputs (hip-centered, shoulder-scaled coordinates), identical train/validation/test subject splits (`splits/splits_dryad_v2_dedup.json`), the identical AdamW optimizer schedule, and the same out-of-fold Platt calibration routine as all other models.
2. **Environment Uniformity:** It eliminates external C++ dependency discrepancies, allowing all 14 baselines to run deterministically from the same Python 3.11 environment.

### Steps and Time to Reproduce Benchmark Results

All benchmark numbers reported in the manuscript can be reproduced using the following steps:

1. **Dataset Audit and Preprocessing:**
   ```bash
   python scripts/audit_dataset.py --data_dir data/raw
   python scripts/preprocess.py --config configs/default.yaml
   ```
   *Measured execution time:* ~2 minutes on standard 16-thread CPU workstation.

2. **Evaluate Benchmark from Released Result Files:**
   ```bash
   python scripts/evaluate.py --results_dir results --output tables/table_main_results.csv
   ```
   *Measured execution time:* **<15 seconds** across all 14 models and 20 random seeds (generates Tables 4, S3, S4, and S5 directly from raw seed JSON records).

3. **Full Re-training from Scratch (Optional):**
   ```bash
   python scripts/train.py --config configs/default.yaml --model A1 --seeds 20
   ```
   *Measured execution time:* ~4.5 hours on an NVIDIA RTX 4050 GPU (6 GB VRAM) for 20 seeds across 3 folds (60 complete training runs); ~18 minutes for a single fold and seed.

4. **Single-Subject End-to-End Inference:**
   ```bash
   # From pre-extracted coordinates:
   python scripts/infer.py --input_npy processed/features/asd_45.npy --checkpoint models/A1/fold1_seed42.pt
   # Measured execution time: ~4.6 seconds (including figure rendering)
   
   # From raw MP4 video:
   python scripts/infer.py --input path/to/video.mp4 --checkpoint models/A1/fold1_seed42.pt
   # Measured execution time: ~10-15 seconds (MediaPipe pose extraction ~6-10s, model pass ~7ms, visualization export ~3-4s)
   ```

---

## 4. Basis of Advance: Software Integration vs. Discriminative Dominance

In our benchmark evaluation across 20 random seeds on the held-out test cohort:
- An **RBF-kernel Support Vector Machine** trained on static kinematic summary features achieved an AUC of **0.871**, exceeding the deep learning models in raw ranking.
- The unconstrained **frame-level gating ablation (A3)** achieved a nominal AUC of **0.857** ($p=0.091$ vs. A1), but its temporal selections decoupled from downstream Transformer attention ($r=0.054$).
- Performance differences between A1 and its deep architectural ablations (A2, A3, A4) were not statistically significant across seed initializations on this fixed partition.

Consequently, PACE-ASD's primary scientific advance over existing software does **not** rest on an assertion of superior raw discrimination over all possible algorithms. Rather, the advance lies in:
1. **End-to-end software integration** bridging computer vision pose estimators and clinical risk modeling.
2. **Strict data hygiene and auditability** preventing partition leakage.
3. **Calibrated probability outputs** suited for decision-support research.
4. **Exploratory contiguous temporal event evidence** preserving atomic movement trajectories for clinician inspection.

---

## References

1. **MediaPipe:** Lugaresi C, et al. MediaPipe: A framework for building perception pipelines. *arXiv:1906.08172*, 2019.
2. **OpenPose:** Cao Z, et al. Realtime multi-person 2D pose estimation using part affinity fields. *CVPR*, 2017.
3. **DeepLabCut:** Mathis A, et al. DeepLabCut: markerless pose estimation of user-defined body parts with deep learning. *Nat Neurosci*, 2018.
4. **PYSKL:** Duan H, et al. PYSKL: towards good practices for skeleton action recognition. *ACM MM*, 2022.
5. **SenseToKnow:** Perochon S, et al. Early detection of autism using digital behavioral phenotyping. *Nat Med*, 2023.
6. **MS-G3D:** Liu Z, et al. Disentangling and unifying graph convolutions for skeleton-based action recognition. *CVPR*, 2020.
