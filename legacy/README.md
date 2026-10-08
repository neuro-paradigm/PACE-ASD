# legacy/

Code, configuration, results and documents of the earlier fixed-split
evaluation of PACE-ASD (one held-out test set of 23 children, 20 seeds x 3
folds), kept for reference and for the history recorded in `CHANGELOG.md`.
None of it is used by the current pipeline, and it is not maintained: the
scripts expect to be run from the repository root with these files in their
former locations (`src/`, `scripts/`, `configs/`, `models/A1/`, `results/`).
The checkpoints in `models/A1/` were trained on landmark arrays normalised
per image axis (see `PREPROCESS_SPEC.md`); load them only for comparison.
