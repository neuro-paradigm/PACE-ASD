# PACE-ASD Data Directory

This directory is used for raw dataset files during preprocessing.
**Raw participant videos are NOT distributed with this repository** due to data sharing restrictions.

## Dataset Source

PACE-ASD uses the publicly available **Dryad ASD Kinematic Dataset**:

> Aljubouri, A. A., Hadi, I., & Rajihy, Y. (2020). *Three Dimensional Dataset Combining Gait and Full Body Movement of Children with Autism Spectrum Disorders Collected by Kinect v2 Camera.* Dryad Digital Repository.  
> DOI: [10.5061/dryad.s7h44j150](https://doi.org/10.5061/dryad.s7h44j150)  
> License: CC0 1.0 Universal Public Domain Dedication

## Dataset Structure

After downloading from Dryad, place files at any local path (e.g., `D:/dryad` or `/data/dryad`).
The expected directory structure is:

```
<raw_dir>/
├── Autism/
│   ├── children with ASD/
│   │   ├── 1/
│   │   │   └── video/
│   │   │       └── video.avi
│   │   ├── 2/
│   │   │   └── video/
│   │   │       └── video.avi
│   │   └── ... (subjects 1–50)
│   └── Severe level of ASD/
│       ├── case1/
│       │   └── *.avi
│       └── ... (cases 1–9)
└── Typical/
    ├── 1/
    │   └── video/
    │       └── video.avi
    └── ... (subjects 1–50)
```

## Cohort Summary

| Group | Subjects | Videos | Label |
|---|---|---|---|
| ASD children (regular) | 50 | 50 | 1 (ASD) |
| Typical children (regular) | 50 | 50 | 0 (TD) |
| Severe ASD (supplement) | 9 | 10 | 1 (ASD) |

After deduplication audit (5 identical TD recordings removed), the main cohort is **N=90** (45 ASD + 45 TD).

## Duplicate Recordings

A forensic MD5 checksum audit revealed 5 pairs of byte-identical TD video files in the public deposit.
The following subjects are segregated to `processed/removed_duplicates/`:

| Removed | Duplicate of |
|---|---|
| td_39 | td_17 |
| td_5 | td_22 |
| td_4 | td_23 |
| td_7 | td_24 |
| td_50 | td_26 |

See `src/audit.py` for the MD5-based verification script.

## Preprocessing

Once raw videos are downloaded:

```bash
# Dry run (no files written — shows what would be processed)
python src/preprocess.py --raw_dir /path/to/dryad --out_dir processed --dry_run

# Full preprocessing
python src/preprocess.py --raw_dir /path/to/dryad --out_dir processed
```

Preprocessed features are stored in `processed/features/*.npy` — shape `(300, 33, 2)` float32.
**Pre-processed features are already included in the repository** and do not require re-extraction.

## Note for Reviewers

The `processed/features/` directory contains all 105 preprocessed numpy arrays (anonymised landmark sequences, no video data). Raw video files must be obtained separately from Dryad for video-based inference. All training and evaluation functionality works directly with the pre-processed `.npy` files.
