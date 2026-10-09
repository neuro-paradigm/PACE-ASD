# CHANGELOG

All notable changes to PACE (released as PACE-ASD before 1.2.0) are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.2.0] — 2026-10

### Third phase (October 2026)
- **Pre-specified criteria** for the third-phase analyses in
  `docs/validation_criteria_phase3.md`, committed before they were run, with one
  amendment (extended gain grid of the positive control) committed before any of
  its models was trained. The protocol's parts are renamed C1-C6 (Tier 1:
  recording audit, performance, recording controls, measurement checks; Tier 2:
  planted events and selection, attribution).
- **Calibration of the protocol**: semi-synthetic positive control
  (`scripts/positive_control.py`) and label-permutation null of the whole
  pipeline (`scripts/permutation_null.py`).
- **New controls and checks**: time of day, all controls together, tracking
  quality (second-difference noise, segment-length variability, MediaPipe
  visibility; `src/recording_features.py`), mean-pose classifier, alternative
  body scales (torso length, 1-s running median of the shoulder distance;
  `src/preprocess.normalise_scale_variant`), incremental value of movement
  scores, split-half reliability, viewpoint covariate, ages as in the original
  article, training within the cropped videos, frame-order shuffle, integrated
  gradients and temporal occlusion (`scripts/phase3_analyses.py`,
  `scripts/attribution_planted.py`), file and container metadata.
- **Pose quality control** (`scripts/pose_qc.py`, `scripts/overlay_sheets.py`)
  and independent code checks (`scripts/verify_analysis.py`).
- `scripts/feature_sets.py` holds the feature sets and one nested run of a
  feature classifier, shared by every script; `PACE_RF_JOBS` sets the threads
  of the random forests (results unchanged).
- Tests in `tests/test_phase3.py`.

### Second revision (October 2026)
- **Name.** The software is now called PACE (pose-aware contiguous-event
  modeling); the repository URL is unchanged. `PACEPredictor` replaces
  `PACEASDPredictor`, which remains as an alias. `PACE_RAW_DIR` is read before
  `PACE_ASD_RAW_DIR`. Inference is described as applying research models to new
  recordings, not as scoring them.
- **Pre-specified criteria.** `docs/validation_criteria.md` fixes the pass/fail
  criteria and settings of the second-revision analyses; it was committed before
  any of them was run. `scripts/evaluate_criteria.py` applies the criteria to
  the stored results (`results/criteria.json`).
- **Gate training modes** (`model.gate_train_mode`: `anneal`, `gumbel`,
  `perturbed`) and a motion-energy heuristic gate (`model.gate_scores: motion`);
  evaluation is unchanged (hard top-M), and the defaults reproduce earlier
  checkpoints. Tests in `tests/test_gate_training_modes.py`.
- **Planted events**: `flap` (oscillation without a raise) and `freeze` (arrest
  of movement keeping the tracking noise) event types; three reference
  detectors (hand height, motion energy, 6-Hz matched filter) with block-level
  localization; amplitudes chosen by a rule fixed in advance
  (`--reference-grid`, `results/synthetic/reference_grid.json`).
- **Controls**: image properties of the videos and landmark jitter
  (`src/recording_features.py`); depth-camera skeletons of the same walks
  (`src/depth_skeleton.py`, standard library only); confound regression of the
  gait descriptors on age and sex (`Gait-LR-adjusted`). New feature-model
  configurations `Scene`, `Jitter`, `Gait-LR-adjusted`, `Depth-LR`, `Depth-RF`,
  `Depth-recording` (`scripts/run_feature_models.py --raw_dir`).
- **Analyses** (`scripts/analyze_mechanisms.py`): `selection_nulls` (agreement of
  untrained networks, the motion heuristic and models of different
  repetitions), `movement` (descriptors within the cropped videos adjusted for
  age and sex, from smoothed and length-equalized trajectories; jitter; image
  properties; representativeness of the cropped autistic children), `depth`
  (depth descriptors, recording properties including time of day, agreement
  with the video descriptors); the subset analyses include cropped videos of
  boys in the shared age range.


### Changed
- **Coordinates in pixel units.** `src/preprocess.py` converts MediaPipe's
  image-normalised coordinates (x / width, y / height) to pixels before hip
  centring and shoulder scaling. Per-axis normalisation stretched body geometry
  by each video's aspect ratio, and in the Dryad deposit the frame size differs
  between groups (35/50 autistic videos uncropped 1080 x 1920, 0/45 typically
  developing videos), so earlier arrays carried the frame format. All arrays in
  `processed/features/` were regenerated from the videos; the raw keypoints
  (`processed/keypoints/`) and per-video metadata (`processed/video_metadata.csv`)
  are now kept. Applying the old normalisation to the new keypoints reproduces
  the old arrays (identical validity masks; largest difference 1e-4).
- **Sequence preparation** (`src/sequence.py`), shared by every model: frames
  with a normalised coordinate beyond 10 shoulder widths are treated as
  undetected, and sequences are shifted so the first detection is frame 0.
  `model.align_onset` does the same inside the network, which makes its output
  independent of undetected frames before the first and after the last
  detection.
- **Augmentation** (`src/dataset.py`) acts on detected frames only, keeps the
  onset and detection gaps, and never returns a near-empty sequence (previously
  it assumed detected frames start at frame 0).
- **Evaluation**: repeated nested subject-level cross-validation over all 95
  deduplicated children (`scripts/run_cv.py`), frozen in
  `splits/cv_partition.json`, storing every out-of-fold prediction.
- **Inference** (`src/inference_api.py`, `scripts/infer.py`): one code path for
  video and array input; ensembles of checkpoints; selected blocks reported in
  input frame numbers with the fraction of ensemble members selecting each block;
  timings and video metadata in `result.json`.
- `requirements.txt` pins `opencv-contrib-python==4.11.0.86` (the build MediaPipe
  installs) instead of `opencv-python`.

### Added
- Comparison architectures: BiGRU, TCN, ST-GCN, CTR-GCN, spatial-temporal
  Transformer (`src/comparison_models.py`); dilated temporal convolutions are
  computed exactly as undilated convolutions over interleaved subsequences.
- Gait and upper-body descriptors (`src/gait_features.py`) with logistic
  regression, SVM, random forest and gradient-boosted trees.
- Controls: clip duration, validity pattern, frame format, age and sex,
  validity-only network, test-time counterfactuals (`scripts/run_feature_models.py`,
  `scripts/analyze_mechanisms.py`).
- Planted-event benchmark for the selection mechanism and attributions
  (`scripts/synthetic_events.py`).
- `src/audit.py`: MD5 duplicate detection for videos and arrays.
- `processed/participant_metadata.csv`: sex and age from the deposit's readme.
- GitHub Actions workflow running the tests on Ubuntu, Windows and macOS.
- `scripts/train_release.py` and the released ensemble in `models/release/`.
- `scripts/check_reproducibility.py`: reruns one outer fold and compares its
  predictions with the stored ones.
- `scripts/measure_pose_timing.py`: pose-extraction and end-to-end timing.

## [1.1.0] — unreleased

### Fixed
- **Padding entered the model (v1.0 defect).** Padded frames acquired non-zero
  features in the microkinetic encoder (convolution bias + GroupNorm shift), so
  Block-ESG's empty-block test never fired, padding blocks could be selected as
  "events", and padded tokens entered attention and the final mean.
  New config key `model.mask_padding` (default `false`, so v1.0 checkpoints load
  and behave exactly as released). When `true`, the frame-validity mask is used
  for GroupNorm statistics, re-zeroing, block validity, block means, the
  Transformer key-padding mask and masked pooling. Predictions are then
  invariant to trailing padding (tests/test_padding_invariance.py).
  A sequence with no detected frame is processed unmasked (avoids undefined
  attention weights).
- **Stream attribution.** `compute_stream_attribution` multiplied the gradient
  w.r.t. *position* by the velocity/acceleration arrays; the result was driven
  by the x10/x50 scale factors. Replaced by `descriptor_attribution`
  (gradient x input on the 198-d descriptor at the first linear layer). The old
  function is kept as `compute_stream_attribution_v10` for comparison only.
- **Region attribution** in the CLI/API now uses the same descriptor-level
  computation, and is also reported over velocity + acceleration only
  (unaffected by mid-hip centring).
- **Gate–attention cross-check** used a block size of 15 for every arm; it now
  uses the model's `event_block_size`.
- `examples/demo_inference.ipynb` did not run (wrong constructor, checkpoint
  keys and methods; hard-coded attribution numbers). Rewritten against
  `PACEASDPredictor`; all cells verified by execution.

### Added
- `configs/config_v11.yaml` — v1.0 protocol with `mask_padding: true` and
  separate output directories.
- `scripts/run_protocol_v11.py` — resumable A1–A4 protocol that stores every
  subject-level test prediction. Run on `configs/config.yaml` it regenerates the
  released v1.0 checkpoints bit-for-bit on the same hardware.
- `scripts/analysis_v11.py` — seed-level and ensemble summaries, stratified
  subject bootstrap, length-only control, padding diagnostics, gate–attention
  agreement within selected blocks with a permutation null, selection
  stability vs chance, descriptor-level attribution and stream occlusion,
  regeneration check, runtime.
- `selected_events.json` / API events report the number of valid frames in each
  selected block.
- `codemeta.json` (schema.org SoftwareSourceCode metadata).
- Tests for padding invariance (all arms) and all-zero input (44 tests total).

### Known issues
- `augment_sequence` (src/dataset.py) assumes valid frames start at frame 0.
  For the 37/105 clips whose first frames have no detection, scaling, noise and
  time warping act on a shifted range; for two clips (asd_32, asd_33) time
  warping can return an all-zero sequence. Left unchanged in 1.1.0 so that
  v1.0 → v1.1 differs only in padding handling; to be fixed in 1.2.0.
- `requirements.txt` pins opencv-python 4.8.1.78; the environment used for the
  v1.1 results had 4.11.0 installed (used only for video decoding).

## [1.0.0] — 2026-09-21

### Added
- Initial public release for BMC Medical Informatics submission
- Complete PACE-ASD model: SpatialEncoder → MicrokineticEncoder → Block-ESG → TemporalEventTransformer
- Full training pipeline with 3-fold subject-level cross-validation, 20 seeds
- Platt scaling calibration (`src/calibration.py`)
- 8-stage interpretability framework (`src/interpretability.py`)
- 14 baseline model implementations (`src/baselines.py`)
- Single-video inference pipeline (`scripts/infer.py`) — accepts raw video OR .npy
- Evaluation reproducer (`scripts/evaluate.py`)
- Case study generator (`scripts/generate_case_study.py`)
- Bilateral asymmetry descriptive utility (`src/asymmetry.py`)
- Python inference API (`src/inference_api.py` — `PACEASDPredictor`)
- Test suite (`tests/`) — 8 test files, synthetic data only, no dataset required
- Documentation: `docs/installation.md`, `docs/usage.md`, `docs/architecture.md`,
  `docs/inference.md`, `docs/reproducibility.md`, `docs/software_comparison.md`
- `CITATION.cff` with all 5 authors and ORCID identifiers
- `LICENSE` (Apache-2.0)
- Inference config: `configs/inference.yaml`
- Evaluation config: `configs/evaluation.yaml`
- Dataset documentation: `data/README.md`
- Checkpoint documentation: `checkpoints/README.md`
- Reviewer package: `release/README_REVIEWER.md`

### Changed
- `configs/config.yaml`: removed hardcoded `D:/dryad` raw_dir path
- `.gitignore`: updated to track `*.pt` checkpoints, remove journal submission debris
- `README.md`: rewritten as BMC-ready researcher-facing documentation

### Security
- No API keys, passwords, or private credentials present
- Hardcoded absolute paths removed from config
