# Reproducing the reported analyses

All commands run from the repository root. Times are for the computer in the
article (Intel Core i5-13450HX, RTX 4050 Laptop GPU, Windows 11).

1. **Videos.** Download `Dataset-2.rar` from Dryad
   (doi:10.5061/dryad.s7h44j150) and extract it so that
   `data/raw/Dataset/Autism` and `data/raw/Dataset/Typical` exist.
2. **Landmarks** (about 14 min with six worker processes):
   `python src/preprocess.py --raw_dir data/raw/Dataset --out_dir processed --workers 6`
   Writes `processed/keypoints/`, `processed/features/` and
   `processed/video_metadata.csv`. The five duplicate recordings
   (`td_39`, `td_5`, `td_4`, `td_7`, `td_50`) are kept in
   `processed/removed_duplicates/` and are not part of the cohort.
3. **Duplicates**: `python src/audit.py --raw_dir data/raw/Dataset --arrays processed/features`
4. **Tests**: `python -m pytest tests -q`
5. **Feature classifiers and controls** (about 30 min; reads the videos and the depth-camera
   skeletons of the deposit): `python scripts/run_feature_models.py --raw_dir data/raw/Dataset`
6. **Networks** (several hours in total; all resumable):
   `python scripts/run_all_cv.py --stage pilot --jobs 3`,
   `python scripts/run_all_cv.py --stage pace --jobs 3`,
   `python scripts/run_all_cv.py --stage comparison --jobs 3`
7. **Planted events**: `python scripts/synthetic_events.py --reference-grid` (reference
   detectors and amplitude choice, no training), then
   `python scripts/synthetic_events.py --event raise --amplitudes 1.0 1.5 2.0 3.0 --arms blocks-M2 frames-K30 blocks-M8 nogate anneal-M2 gumbel-M2 perturbed-M2 motion-M2 --jobs 4`,
   `python scripts/synthetic_events.py --event flap --amplitudes 0.6 1.0 --arms blocks-M2 frames-K30 anneal-M2 motion-M2 nogate --jobs 4`,
   `python scripts/synthetic_events.py --event freeze --amplitudes 1.0 --arms blocks-M2 frames-K30 anneal-M2 motion-M2 nogate --jobs 4`,
   then `python scripts/synthetic_events.py --summarize`
8. **Released ensemble**: `python scripts/train_release.py`
9. **Summaries**: `python scripts/analyze_cv.py` and `python scripts/analyze_mechanisms.py`,
   then `python scripts/evaluate_criteria.py` (criteria of `docs/validation_criteria.md`)
10. **Timing**: `python scripts/measure_pose_timing.py --n 10 --device cpu`
11. **Rerun check**: `python scripts/check_reproducibility.py --arm PACE --repeat 0 --fold 0`
    reruns one outer fold into `results/repro/` and compares its predictions
    with the stored ones (`results/cv/reproducibility.json`).

The partition (`splits/cv_partition.json`) is part of the repository; deleting
it makes `run_cv.py` generate it again from `cv.seed` in `configs/config.yaml`,
which gives the same partition.

Every model is seeded and trained with deterministic GPU kernels, so a rerun on
the same hardware and library versions reproduces the stored predictions
(step 11 reproduced one outer fold of the reference network bit for bit);
other hardware reproduces the protocol, not necessarily every bit.
