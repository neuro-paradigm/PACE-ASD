# PACE-ASD: Dryad-Only Retraining and Ablation Protocol

**Purpose:** Close the three MLWA desk-rejection findings by (1) retraining strictly on Dryad, with Move4AS removed entirely — not just excluded from training, but dropped from the study, since it's an adult cohort and therefore not a valid stress test for a pediatric model regardless of how it's used; (2) running a complete ablation matrix that includes two currently-missing models; (3) evaluating primarily on train/validation/test performance, with a small, explicitly-scoped supplementary note on the 9 differently-recorded subjects rather than a full OOD testing apparatus.

**Consequence worth flagging up front:** dropping Move4AS removes the age confound at its root, which is a clean fix for one of the editor's four flagged variables (age, sensor, protocol, environment all changing at once). But it also means the evidence base for any "shortcut-resilient" or "OOD-robust" headline claim shrinks to whatever the 9-subject note can support — and per the caveat below, that's sensitivity on a small, ASD-only group, not a full robustness demonstration. The abstract/conclusion's "shortcut-resilient" framing needs to scale down to match: something closer to "efficient, well-calibrated same-domain performance, with a preliminary device-generalization observation" rather than a headline robustness claim. Better to undersell this than repeat the original problem.

**Reproducibility applies to the whole project, not just the final numbers.** Rejection point 2 was a code/manuscript mismatch, not a total absence of a released artifact — the fix here has to be broader than a final regeneration check. Every stage below (the recording-source audit and split assignment in Section 1, the preprocessing spec in Section 2, every model in the ablation matrix in Section 4, and the supplementary note in Section 6) needs its own pinned code, fixed seeds, and documented commands from the start, not bolted on at the end. Section 5 operationalizes this as the final lock, but it isn't the only place reproducibility gets enforced — treat every section as something a third party should be able to rerun independently and get the same numbers, not just the headline table.

---

## 1. Data Partitioning (Dryad-Only)

No Move4AS anywhere in this document. Every split below is carved from the Dryad cohort alone.

### 1.1 Recording-source audit (required before any split is defined)

Per the dataset's own record (Al-Jubouri et al., 2020, Dryad DOI 10.5061/dryad.s7h44j150): the cohort is 50 ASD + 50 TD, each recorded on **both** a Kinect v2 (3D joints, skeleton video, and its own onboard RGB) **and** a Samsung Note 9 rear camera — plus a separate group of **9 severe-autism children with Samsung-only color video** (no usable Kinect skeletal tracking, and no TD counterpart). That's 109 documented participants, not the 110 the manuscript states — resolve against the raw folder count, don't assume either figure.

Two audit items, both blocking before Section 1.2:

1. **Resolve N.** Count actual subject folders in the raw release; reconcile against both the manuscript's 110 and the dataset record's 109.
2. **Resolve camera provenance for the 100 "regular" subjects.** Both a Kinect-onboard RGB stream and a Samsung Note 9 stream exist per subject. The manuscript's description doesn't establish which physical camera was actually used for MediaPipe extraction, or whether that choice was consistent across all 100. Audit this per-subject — an inconsistent mix here is an unquantified device confound sitting inside what's currently treated as a homogeneous training pool.

### 1.2 Split structure

| Split | Subjects | Role |
|---|---|---|
| Train/Val | Majority-camera-source subjects (the consistent Kinect-RGB or Samsung group, per 1.1), minus the 9 held out below | 3-fold `StratifiedGroupKFold` on `subject_id` |
| Test | Subject-disjoint held-out subset, **same** camera source as train/val | Standard generalization metric — this is now the primary and only headline evaluation |
| Supplementary device-shift note | The 9 severe-ASD, Samsung-only subjects (pending 1.1) — zero training exposure | Small, explicitly-scoped check (Section 6) — not a formal test tier, not part of the headline results |

- Enforce subject-level splitting throughout — no clip from a subject appears in more than one split.
- Freeze subject IDs into a versioned file (`splits_dryad_only_v1.json`) once 1.1 is resolved, so every model in Section 4 trains and evaluates on **identical** splits.
- Drop the Move4AS-specific subject-balanced clip-loss weighting — no longer applicable.

---

## 2. Preprocessing Pipeline (audit before locking)

**Single pipeline, no exceptions.** The centering/feature-dimensionality mismatch was exactly the kind of problem a second preprocessing path creates — a script that drifted from what actually produced the reported numbers. This project uses **one canonical preprocessing implementation**, called identically by every split (train/val/test/supplementary note) and every ablation model (A1–A7). No model gets a bespoke preprocessing variant, no notebook-only or exploratory version is allowed to diverge from what's used to produce reported numbers, and the camera-source resolution from Section 1.1 is enforced here, not re-decided per experiment: once that audit fixes which camera feed each subject's RGB stream comes from, this single script is the only place that decision is encoded.

1. MediaPipe 33-keypoint extraction from raw RGB video.
2. **Centering** — spec says mid-hip origin. **Audit:** confirm the released repo's preprocessing code actually centers on mid-hip, not another joint. Reconcile code and manuscript to agree — fix whichever is wrong, don't just relabel.
3. **Scale normalization** — dual-axis inter-shoulder normalization. **Audit:** confirm actual feature dimensionality at this stage matches the stated $D_c = 384$.
4. Fixed sequence length: pad/truncate to $T = 300$ frames (10 s @ 30 fps).
5. **Duplicate-implementation check.** Before anything in Section 4 is trained, search the full codebase (scripts, notebooks, per-experiment configs) for any second preprocessing path — a duplicated function, a notebook cell that re-implements centering or normalization, a per-model override. Consolidate everything found into the single canonical module referenced above. This is the direct, concrete fix for the specific discrepancy that triggered rejection point 2, not just a policy statement.
6. Once audited and consolidated, freeze this into `PREPROCESS_SPEC.md` with a pinned commit hash **before** any model below is trained. Every ablation arm — A1 through A7, and the Section 6 supplementary note — must call this same module and consume identically preprocessed inputs.

---

## 3. Core Architecture (unchanged, retrained Dryad-only)

Spatial residual MLP projection → multi-scale 1D temporal convolutions → **Block-ESG** (variable per ablation, see Section 4) → Transformer encoder, 2-layer/4-head (variable per ablation) → Platt probability calibration, fit exclusively on Dryad-only out-of-fold validation logits.

**Training protocol per model** (applies to every row in Section 4):
- 3-fold `StratifiedGroupKFold` CV → ensemble of fold checkpoints
- **20 independent random-seed reinitializations** of the full pipeline, matching the existing seed-stability protocol — required for every ablation arm, not just the main model
- Loss: standard BCE

---

## 4. Ablation Matrix — What to Train

| ID | Model | Change from full PACE-ASD | Why it's needed |
|---|---|---|---|
| **A1** | PACE-ASD (full, Dryad-only) | Baseline: $L=15$, $M=8$ | Reference point for every comparison below |
| **A2** | No-Block-ESG (dense) | Gate removed; Transformer sees all 300 frames | Isolates whether Block-ESG contributes anything beyond the rest of the pipeline on the primary train/test evaluation. Also run against the Section 6 supplementary note. |
| **A3** | Frame-granularity ablation | Same Gate MLP / top-$K$ logic, block length $L=1$, $M$ scaled to preserve a comparable selected-frame budget (e.g. $M=120$) | **Missing today.** Isolates the exact variable the editor questioned — contiguous blocks vs. individual frame/token selection — using your own codebase, so no external backbone or dataset confound is introduced. |
| **A4** | No-Transformer | Transformer encoder removed, classifier reads Block-ESG output directly | Exists already (Table `transformer-retrain`); retrain under the Dryad-only splits for consistency. |
| **A5** | Literature baselines | Stacked LSTM, Conv1D-BiLSTM-Attn, Kinematic CNN-LSTM, MS-G3D, MS-G3D+ConvNeXt, MediaPipe+{LR, SVM, RF, XGBoost} | Retrain under identical Dryad-only splits/seeds so the comparison set is matched to A1–A4. |
| A6 *(optional)* | Windowed-attention baseline (Swin/Longformer-style) | Restricted local attention window, all frames retained | Tests the "hard selection vs. windowed restriction" distinction the manuscript currently only asserts conceptually in Table 1. See Section 9, Tier B for the specific windowed-attention comparators this stands in for. |
| A7 *(exploratory, optional)* | Kinematic-boundary block gating | Same Gate MLP / top-$K$ logic, but block boundaries derived from velocity/acceleration zero-crossings in the pose stream instead of fixed $L=15$ windows | Tests the novelty direction in Section 9. **Not a substitute for A3** — reviewers will still expect the plain granularity ablation regardless of whether this direction is pursued. Requires its own validation (Section 9) before being treated as a headline result. |

Every row A1–A5 is evaluated on the standard train/test split (Section 1.2) as the primary result. A2 and A3 are additionally checked against the Section 6 supplementary note, since those two are the ones the rejection letter's novelty and causal-attribution concerns are actually about.

---

## 5. Reproducibility Lock (do this before any headline number is reported)

- Pin one commit/tag covering the entire ablation matrix.
- `REPRODUCE.md`: exact environment (framework/CUDA versions), and one command per table/figure to regenerate it from raw data.
- Confirm the released repo reproduces the Dryad-only-trained numbers exactly — do this as the final gate, after Sections 4 and 6 are complete.

---

## 6. Supplementary Note: Device-Shift Check (9 Subjects)

This is intentionally small in scope — a transparency check, not a robustness claim.

**Procedure:** evaluate A1–A5 (Section 4), trained with zero exposure to the 9 held-out subjects, on those 9 subjects.

**Report:**
- **Sensitivity only**, with the confidence interval explicit given n=9. Do not report AUC or specificity — the group is ASD-only, so neither is computable, and don't manufacture a synthetic TD comparison to force one.
- Whether sensitivity holds relative to that same model's sensitivity on the main test set, or drops — stated as an observation, not a statistical claim.

**How to write this up:** a short paragraph or small table in a limitations/supplementary section, explicitly labeled as an n=9, single-class, non-powered observation. Do not use it to support the abstract's headline claims, and do not run the full statistical battery (Wilcoxon, DeLong, Holm–Bonferroni) against it — that machinery implies a level of evidential weight this sample can't bear, which is the same overclaiming problem that produced the original rejection, just at a smaller scale.

---

## 7. Reporting Standards

- Headline numbers (Section 4, primary train/test results) = mean ± SD across the same 20 seeds, for every model.
- The Section 6 note is reported separately, clearly labeled as supplementary/exploratory, with no combined or averaged statistic that blends it into the headline results.

---

## 8. Related Architecture Landscape (for Table 1 Repositioning)

Purpose: avoid a repeat of the STTS problem — where the closest prior art wasn't in the comparison table — by cataloguing everything found that overlaps with Block-ESG's mechanism, ranked by how directly it threatens the novelty claim. All of these should be added to the manuscript's Table 1 (sparsification comparison), not just STTS.

### Tier A — Closest: skeleton/pose-specific temporal segmentation (same domain as Block-ESG)

| Architecture | Source | Mechanism | Why it's a risk |
|---|---|---|---|
| **SkelFormer** | Jan 2026, *PLOS ONE*, DOI 10.1371/journal.pone.0340390 | An "SKT Block" with a *Temporal Split* submodule that adaptively, hierarchically divides a skeleton sequence into learned temporal segments at multiple scales, explicitly to isolate meaningful motion phases | **Highest risk found.** Skeleton-specific (not generic video like STTS), adaptive rather than fixed-length, and very recent. Against this, "contiguous fixed-length blocks" as a distinguishing feature looks thinner than it did against STTS alone. |
| **MTC-Former** (Multi-Grained Temporal Clip Transformer) | 2025, MDPI *Applied Sciences* | Segments the sequence into $K$ clips at multiple granularities simultaneously, multi-branch, with learned attention weighting per granularity | Block/clip-based, skeleton-specific, learned — same family as Block-ESG. |
| **MTT** (Multi-Scale Temporal Transformer) | ~2022 | Segmental sampling + a skeleton-transformer module that automatically selects important joints | Selection mechanism over skeleton data; different axis (joints vs. frames) but same "learned selection" family. |
| **STAR** (Sparse Transformer-based Action Recognition) | 2021, arXiv:2107.07089 | Sparse attention + segmented linear attention over skeleton sequences | Segmentation + sparsity, same domain. |

### Tier B — Same neighborhood as STTS: generic video-transformer token/frame selection

| Architecture | Mechanism |
|---|---|
| SCSampler, AdaFrame, OcSampler | Lightweight policy networks that sample/select frames before the backbone runs |
| Token Turing Machines (Ryoo et al., 2023) / STTM (spatial-temporal token merger, ACM TOMM) | Compress tokens via a memory/merge mechanism rather than dropping them — a genuinely different mechanism (merging vs. hard selection) worth distinguishing Block-ESG from explicitly |
| Focal self-attention; Longformer | Windowed/local attention — the comparators A6 (Section 4) stands in for |

### Tier C — More distant mechanism, same idea: adaptive-length segmentation in time-series transformers

EntroPE (Abeywickrama et al., 2025), TimeMosaic (Ding et al., 2026), BSAT (arXiv:2601.00698), ReinPatch (Wu et al., 2026), HDMixer (Huang et al., 2024), PathFormer (Chen et al., 2024), MultiResFormer (arXiv:2311.18780).

Not pose-specific, but this is a whole active subfield built around replacing fixed-length patches with adaptively-placed boundaries (entropy-based, curvature-based, RL-learned). Relevant context for Section 9 below: if the novelty pitch ever becomes "adaptive block boundaries" in the abstract, this is the literature that already owns that idea generically.

### Tier D — Domain-specific, lower architectural overlap but relevant citations

- "Skeleton-Based Activity Recognition for Children with Autism Using Graph Convolutional Networks" (MDPI *Sensors*, 2026) — ProtoGCN backbone + margin regularizer, but for multi-class therapy-activity recognition, not ASD/TD screening, and no temporal-selection mechanism. Cite as domain-related work, not a direct architectural comparator.
- "Bio-Inspired Self-Supervised Learning for Wrist-worn Accelerometer Data" (arXiv:2603.10961) — defines token boundaries at acceleration zero-crossings ("type 2" movement segments) for wearable IMU data. Directly relevant to Section 9.
- "Using deep learning to classify developmental differences in reaching and placing movements in children with and without autism spectrum disorder" (*Scientific Reports*, Dec 2024) — velocity/acceleration/jerk zero-crossing "movement units" specifically in ASD-vs-TD kinematics, via IMU, as a feature not an architectural mechanism. Directly relevant to Section 9.

---

## 9. Novelty Positioning: Kinematic-Boundary Block Gating (Exploratory, A7)

**Why not pursue "adaptive block boundaries" generically:** Tier C above shows that's an occupied, actively-worked idea at the mechanism level, and SkelFormer (Tier A) makes it specifically risky in this domain — skeleton-specific, adaptive, hierarchical, and recent. Framing a resubmission's novelty around "we made the blocks adaptive" would likely trade one prior-art problem for a closer one.

**The narrower, currently-open combination:** no architecture found combines (a) block boundaries derived from kinematic events in the pose stream itself — velocity/acceleration zero-crossings, i.e. biomechanically meaningful movement-unit transitions — with (b) a differentiable saliency gate over those blocks, in (c) a pose-transformer, for (d) ASD screening. The individual pieces exist separately (Tier D: zero-crossing tokenization for wearable accelerometer data; zero-crossing "movement units" as an ASD-vs-TD kinematic feature) but not combined into an architecture, and not validated with clinical interpretability.

**What this would look like (A7 in Section 4):** replace fixed $L=15$ windows with blocks bounded at movement-unit transitions detected from the pose stream, keep the existing Top-K differentiable gate operating on these variable-length blocks, and validate with something the accelerometer/movement-unit literature doesn't offer — whether the gate's selected blocks align with clinically meaningful motor events, not just accuracy.

**Caveats before committing to this:**
- This is genuinely new work, not a repositioning of existing results — treat it as exploratory for a future iteration, not a requirement for closing the current three rejection points (Sections 1–8 already do that without it).
- Kinematic zero-crossing detection is materially noisier from 2D MediaPipe pose estimates than from a clean accelerometer signal — this needs its own validation (e.g., against manually-annotated movement transitions on a subset) before results from it are trusted, let alone reported.
- It does not replace A3 (Section 4) — the plain fixed-vs-frame-level granularity ablation is still the one reviewers will expect regardless of whether this direction is pursued.

---

## 10. Deliverables Checklist — mapped back to the desk-rejection letter

- [ ] **Recording-source audit (Section 1.1)** — resolve 109 vs. 110 subject count and per-subject camera provenance before any split is finalized; this gates everything else
- [ ] **A3 (granularity ablation)** — empirical evidence for the block-contiguity claim (rejection point 3)
- [ ] **Table 1 rewritten with the full Section 8 architecture landscape** (STTS, SkelFormer, MTC-Former, MTT, STAR, plus Tier B/C entries as relevant), citing the A3 result as the empirical basis for the contiguity distinction, not just a conceptual one (rejection point 3)
- [ ] **A2 (no-Block-ESG) result on the primary train/test split** — shows what Block-ESG actually contributes once the confounded external comparison is gone
- [ ] **Section 6 supplementary note**, clearly scoped as n=9 exploratory, not folded into headline claims
- [ ] **Code/manuscript audit** — centering method and $D_c$ reconciled between repo and text (rejection point 2)
- [ ] **Single preprocessing pipeline confirmed** — no duplicate/parallel implementation found across scripts, notebooks, or per-model configs; every split and every ablation arm (A1–A7, Section 6) verified to call the one canonical module (rejection point 2)
- [ ] **`REPRODUCE.md` + pinned commit**, verified to regenerate the Dryad-only-trained numbers exactly (rejection point 2)
- [ ] **Abstract/conclusion rewrite** — remove the Move4AS-based OOD numbers and the "shortcut-resilient" headline framing entirely; reposition the contribution around efficiency (parameter count, latency) and calibrated same-domain performance, with the device-shift note mentioned only as a limitations-section observation
- [ ] *(Optional, future work)* **A7 kinematic-boundary exploration (Section 9)** — only after Sections 1–7 are locked; not required to close the current rejection
