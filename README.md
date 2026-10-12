# PACE: cross-validated modeling of whole-body movement in short videos

[![tests](https://github.com/neuro-paradigm/PACE-ASD/actions/workflows/tests.yml/badge.svg)](https://github.com/neuro-paradigm/PACE-ASD/actions/workflows/tests.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1.2-orange.svg)](https://pytorch.org/)
[![Dataset: Dryad CC0](https://img.shields.io/badge/Dataset-Dryad%20CC0-green.svg)](https://doi.org/10.5061/dryad.s7h44j150)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-lightgrey.svg)](LICENSE)

PACE (pose-aware contiguous-event modeling) takes short videos of one person
moving to cross-validated, recalibrated predictions, and makes the recording
properties a model could exploit instead of movement visible and testable. It
contains:

* **preprocessing**: MediaPipe Pose landmarks converted to pixel units (so that
  geometry does not depend on how a video was cropped), hip-centred and
  shoulder-scaled; raw keypoints and per-video metadata are kept;
* **sequence preparation** shared by every model: a plausibility check and
  onset alignment (`src/sequence.py`);
* **a reference network** with contiguous event selection and frame-validity
  masking throughout (`src/model.py`), **five comparison architectures**
  (BiGRU, TCN, ST-GCN, CTR-GCN, spatial–temporal Transformer;
  `src/comparison_models.py`) and **four classifiers of hand-crafted gait
  descriptors** (`src/gait_features.py`);
* **repeated nested subject-level cross-validation** that stores every
  out-of-fold prediction (`scripts/run_cv.py`, `scripts/run_feature_models.py`);
* **controls and checks**: clip duration, validity pattern, frame format,
  image properties (scene and camera), landmark jitter and demographics
  controls; a reader for depth-camera skeletons of the same recordings; a
  validity-only network; test-time counterfactuals; a planted-event benchmark
  with three event types and reference detectors for the selection mechanism
  and attributions; invariance tests;
* **calibration of the checks**: planted movement differences of graded size
  at several sample sizes, sanity checks with planted recording differences
  (image, framing, video only) and a label-permutation null, with exact
  intervals (`scripts/phase4_calibration.py`, `scripts/phase5_analyses.py`,
  `scripts/permutation_null.py`); incremental value of a movement score by
  nested stacking with a Firth second level (`scripts/phase4_common.py`);
  second-sensor comparison matched for projection, frame rate and duration;
* **pass/fail criteria, most written before their analyses**
  (`docs/validation_criteria.md`, `_phase3.md` to `_phase5.md`; commits, not
  independent time stamps), applied to the stored results by
  `scripts/evaluate_criteria.py`;
* **inference** for research: applying trained models to a video or array with
  one checkpoint or an ensemble (`scripts/infer.py`, `src/inference_api.py`).

**Research software, not a medical device.** The demonstration uses one public
dataset of children walking towards a camera. Nothing here establishes
diagnostic or clinical validity; nothing in the code is specific to autism.

---

## Quick start

```bash
git clone https://github.com/neuro-paradigm/PACE-ASD.git
cd PACE-ASD
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m pytest tests -q            # synthetic data only, a few seconds
```

Apply the released research models to a stored landmark array or a video
(for method development and reanalysis; the output is a model score, not a
diagnostic or screening result):

```bash
python scripts/infer.py --input_npy processed/features/asd_1.npy --checkpoint models/release --output outputs/asd_1
python scripts/infer.py --input walk.mp4 --checkpoint models/release --output outputs/walk
```

```python
import sys; sys.path.insert(0, "src")
from inference_api import PACEPredictor
pred = PACEPredictor("models/release")
r = pred.predict_npy("processed/features/asd_1.npy")
print(r["probability"], r["probability_sd"], r["selection"]["selection_is_trivial"])
```

The released models were trained on every child of the demonstration cohort,
whose frame format differs between groups (see below); their outputs on new
recordings inherit that limitation and must not be used to make decisions
about individuals. Do not use them to estimate performance on this cohort.

---

## Reproducing the analyses

| Step | Command | Output |
|---|---|---|
| 1. Landmarks from the Dryad videos (download separately) | `python src/preprocess.py --raw_dir data/raw/Dataset --out_dir processed --workers 6` | `processed/keypoints/`, `processed/features/`, `processed/video_metadata.csv` |
| 2. Inventory and duplicate check | `python src/audit.py --raw_dir data/raw/Dataset --arrays processed/features` | report, duplicate groups |
| 3. Tests | `python -m pytest tests -q` | |
| 4. Feature classifiers and controls (recording, image, jitter, depth skeletons) | `python scripts/run_feature_models.py --raw_dir data/raw/Dataset` | `results/cv/runs/<model>/` |
| 5. Networks (pilot, PACE family, comparison architectures) | `python scripts/run_all_cv.py --stage pilot`, then `--stage pace`, then `--stage comparison` | `results/cv/runs/<arm>/`, `models/cv/` |
| 6. Planted-event benchmark | `python scripts/synthetic_events.py --reference-grid`, the runs in `docs/reproducibility.md`, then `--summarize` | `results/synthetic/` |
| 7. Summaries and criteria | `python scripts/analyze_cv.py`, `python scripts/analyze_mechanisms.py`, `python scripts/evaluate_criteria.py` | `results/cv/summary.json`, `results/cv/mechanisms.json`, `results/criteria.json` |
| 8. Released ensemble | `python scripts/train_release.py` | `models/release/` |
| 9. Rerun check | `python scripts/check_reproducibility.py` | `results/cv/reproducibility.json` |

All settings are in `configs/config.yaml`; the partition is frozen in
`splits/cv_partition.json`. Every run is seeded and uses deterministic GPU
kernels; runs resume at the level of an outer fold.

<!-- RESULTS_TABLE -->
### Results on the demonstration cohort

Out-of-fold AUC (95% interval over children), 3 x 5-fold nested cross-validation over 95 children; 'cropped' = the 15 autistic and 45 typically developing children whose videos were cropped (the only frame format both groups share). Source: `results/cv/summary.json`, `results/cv/mechanisms.json`.

| Configuration | AUC, all children | AUC, cropped videos |
|---|---|---|
| controls: clip duration | 0.635 (0.52-0.74) | 0.493 (0.33-0.70) |
| controls: validity pattern | 0.634 (0.52-0.74) | 0.566 (0.41-0.73) |
| controls: age and sex | 0.701 (0.58-0.81) | 0.655 (0.51-0.79) |
| controls: frame format | 0.901 (0.84-0.95) | 0.671 (0.54-0.80) |
| controls: format, validity and demographics | 0.891 (0.83-0.94) | 0.661 (0.52-0.79) |
| controls: scene and camera (image properties) | 0.967 (0.94-0.99) | 0.941 (0.86-1.00) |
| controls: high-frequency landmark motion | 0.590 (0.49-0.69) | 0.605 (0.44-0.75) |
| controls: tracking quality | 0.569 (0.47-0.67) | 0.601 (0.46-0.72) |
| controls: time of day | 0.853 (0.77-0.93) | 0.951 (0.89-1.00) |
| controls: depth-recording properties | 0.897 (0.83-0.95) | 0.961 (0.92-0.99) |
| controls: all controls together | 0.970 (0.93-0.99) | 0.922 (0.81-0.99) |
| controls: validity-only network | 0.626 (0.52-0.72) | 0.522 (0.37-0.67) |
| posture and gait descriptors: mean pose (posture only) | 0.819 (0.74-0.89) | 0.834 (0.64-0.97) |
| posture and gait descriptors: logistic regression | 0.858 (0.78-0.92) | 0.962 (0.92-0.99) |
| posture and gait descriptors: support vector machine | 0.872 (0.80-0.93) | 0.944 (0.89-0.98) |
| posture and gait descriptors: random forest | 0.886 (0.81-0.95) | 0.973 (0.93-1.00) |
| posture and gait descriptors: gradient-boosted trees | 0.872 (0.80-0.93) | 0.950 (0.90-0.99) |
| posture and gait descriptors: logistic regression, age and sex removed | 0.802 (0.72-0.88) | 0.939 (0.88-0.98) |
| posture and gait descriptors: logistic regression, image-normalized | 0.837 (0.76-0.91) | 0.951 (0.90-0.99) |
| posture and gait descriptors: logistic regression, torso scale | 0.901 (0.83-0.95) | 0.972 (0.94-0.99) |
| posture and gait descriptors: random forest, torso scale | 0.878 (0.80-0.94) | 0.957 (0.91-0.99) |
| posture and gait descriptors: logistic regression, 1-s median scale | 0.873 (0.80-0.93) | 0.967 (0.93-0.99) |
| posture and gait descriptors: random forest, 1-s median scale | 0.893 (0.82-0.95) | 0.964 (0.92-1.00) |
| depth-camera skeletons of the same walks: descriptors, logistic regression | 0.697 (0.60-0.79) | 0.720 (0.55-0.86) |
| depth-camera skeletons of the same walks: descriptors, random forest | 0.833 (0.75-0.91) | 0.849 (0.72-0.94) |
| comparison networks: BiGRU | 0.891 (0.82-0.94) | 0.932 (0.88-0.98) |
| comparison networks: TCN | 0.866 (0.79-0.93) | 0.940 (0.88-0.99) |
| comparison networks: ST-GCN | 0.855 (0.78-0.92) | 0.970 (0.94-0.99) |
| comparison networks: CTR-GCN | 0.885 (0.81-0.94) | 0.969 (0.93-0.99) |
| comparison networks: ST-Transformer | 0.884 (0.81-0.95) | 0.909 (0.82-0.98) |
| test-case network and ablations: test-case network (w=15, M=8) | 0.900 (0.83-0.96) | 0.957 (0.91-0.99) |
| test-case network and ablations: no gate | 0.892 (0.82-0.95) | 0.952 (0.91-0.99) |
| test-case network and ablations: frame selection (w=1, M=120) | 0.902 (0.83-0.96) | 0.956 (0.91-0.99) |
| test-case network and ablations: budget M=4 | 0.907 (0.84-0.96) | 0.962 (0.92-0.99) |
| test-case network and ablations: averaging instead of attention | 0.879 (0.80-0.94) | 0.953 (0.90-0.99) |
| test-case network and ablations: no validity masking | 0.894 (0.83-0.95) | 0.948 (0.90-0.98) |
| test-case network and ablations: no onset alignment | 0.897 (0.83-0.96) | 0.950 (0.90-0.99) |
| test-case network and ablations: image-normalized coordinates | 0.871 (0.80-0.93) | 0.905 (0.83-0.96) |
| test-case network and ablations: epoch selection by validation loss | 0.905 (0.84-0.96) | 0.965 (0.93-0.99) |

Frame format alone separates the groups in this dataset; within the cropped videos image properties still do; according to the file dates in the deposit the two groups were recorded months apart, and many autistic children's videos do not show the walking task (an unblinded, AI-assisted annotation also found no setting shared). None of the discrimination above can therefore be attributed to movement; movement differences are neither established nor excluded. Nothing here establishes clinical validity.
<!-- /RESULTS_TABLE -->

---

## Data

The demonstration uses the colour videos of the Dryad deposit
[doi:10.5061/dryad.s7h44j150](https://doi.org/10.5061/dryad.s7h44j150) (CC0),
described in Al-Jubouri, Ali and Rajihy (2021), *J. Phys.: Conf. Ser.* 1818, 012201.
Raw videos are not redistributed. The repository contains, for every recording,
the MediaPipe keypoints (`processed/keypoints/`), the prepared arrays
(`processed/features/`, `(300, 33, 2)` float32), video metadata
(`processed/video_metadata.csv`) and, from the deposit's readme, each child's
sex and age (`processed/participant_metadata.csv`; see
`processed/participant_metadata_README.md` for caveats).

Properties of this dataset that any analysis must handle:

* five pairs of byte-identical videos among the typically developing children
  (one of each pair is in `processed/removed_duplicates/` and not analysed);
* 35 of 50 autistic children's videos are uncropped 1080 x 1920 frames, whereas
  all 45 typically developing children's videos were cropped to individual
  sizes; frame format alone separates the groups;
* the groups differ in sex and age;
* recordings differ in length and in when the child is first detected.

---

## Repository layout

```
configs/config.yaml        all settings
src/                       preprocess, audit, sequence, model, comparison_models,
                           gait_features, protocol, calibration, interpretability,
                           inference_api, dataset (augmentation)
scripts/                   run_cv, run_feature_models, run_all_cv, synthetic_events,
                           analyze_cv, analyze_mechanisms, train_release, infer
tests/                     pytest suite (synthetic data only)
processed/                 keypoints, features, metadata
splits/cv_partition.json   frozen repeated nested partition
results/cv/runs/           every out-of-fold prediction
models/release/            released ensemble
```

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

Apache License 2.0. Research use; not a medical device.
