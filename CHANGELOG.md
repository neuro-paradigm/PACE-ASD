# CHANGELOG

All notable changes to PACE-ASD are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.2.0] — 2026-10

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
