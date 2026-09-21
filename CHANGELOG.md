# CHANGELOG

All notable changes to PACE-ASD are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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
- `LICENSE` (MIT)
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
