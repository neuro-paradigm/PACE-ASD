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
│ PYSKL WORKFLOW (Fragmented Tooling)                                                   │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. Dependency Hell: Install MMCV, MMAction2, MMPose + C++ CUDA extensions             │
│ 2. Custom Extraction: Run external pose script -> write pickle dictionary converter    │
│ 3. Data Ingestion: Write custom Dataset class & StratifiedGroupKFold split generator   │
│ 4. No Data Hygiene: Duplicate source files leak undetected across partitions          │
│ 5. Model Execution: Train action-recognition model (e.g., MS-G3D or ST-GCN)          │
│ 6. Manual Post-Processing: Write separate script for Platt scaling & ECE calculation  │
│ 7. Opaque Decisions: Extract uncalibrated task logits; no event-level localization    │
└────────────────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────────────────┐
│ PACE-ASD WORKFLOW (Integrated Decision-Support Research Framework)                     │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. Zero C++ Compilation: Standard pure Python & PyTorch dependencies (`pip install`)   │
│ 2. Automated Hygiene: Run `scripts/audit_dataset.py` to catch cryptographic duplicates  │
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
| **Installation** | Requires building `mmcv-full` with matching CUDA toolkits; prone to environment breaks. | Pure Python/PyTorch wheel installation via `requirements.txt`. | PACE-ASD installs in minutes on standard Windows/Linux PCs without GPU compilers. |
| **Input Flexibility** | Accepts custom pickle dictionaries formatted for action benchmarks. | Accepts raw video files (`.mp4`, `.avi`, `.mov`) or pre-extracted `.npy` landmarks. | Non-technical researchers can run inference directly without pose estimation libraries. |
| **Data Hygiene** | No duplicate detection; 5 identical pairs in Dryad pass unnoticed. | Built-in MD5 audit quarantines duplicate recordings before split creation. | Prevents silent train/test leakage and invalid test metrics. |
| **Clinical Calibration** | Outputs uncalibrated classification logits. | Integrates out-of-fold Platt calibration, optimizing ECE for clinical risk modeling. | Enables calibrated risk stratification rather than raw decision thresholds. |
| **Temporal Evidence** | Dense sequence processing without temporal event localization. | Block-ESG identifies contiguous 15-frame (500 ms) events and verifies attention coherence ($r=0.822$). | Clinicians and researchers can visually audit which sub-movements triggered the screen. |
| **Runtime Efficiency** | Heavy 3D graph convolutions require multi-GB GPU VRAM. | Compact 223K-parameter model runs in ~7.2 ms on CPU and ~3.7 ms on GPU (<14 MB VRAM). | Runs privately and securely on standard clinical workstation laptops. |

---

## 3. Basis of Advance: Software Integration vs. Discriminative Dominance

In our benchmark evaluation across 20 random seeds on the held-out test cohort:
- An **RBF-kernel Support Vector Machine** trained on static kinematic summary features achieved an AUC of **0.871**, exceeding the deep learning models in raw ranking.
- The unconstrained **frame-level gating ablation (A3)** achieved a nominal AUC of **0.857** ($p=0.091$ vs. A1), but its temporal selections decoupled from downstream Transformer attention ($r=0.054$).

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
