# PACE-ASD: cross-validated modeling of whole-body movement in short videos

[![tests](https://github.com/neuro-paradigm/PACE-ASD/actions/workflows/tests.yml/badge.svg)](https://github.com/neuro-paradigm/PACE-ASD/actions/workflows/tests.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1.2-orange.svg)](https://pytorch.org/)
[![Dataset: Dryad CC0](https://img.shields.io/badge/Dataset-Dryad%20CC0-green.svg)](https://doi.org/10.5061/dryad.s7h44j150)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-lightgrey.svg)](LICENSE)

PACE-ASD (pose-aware contiguous-event modeling) takes short videos of one person
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
* **controls and checks**: clip duration, validity pattern, frame format and
  demographics controls; a validity-only network; test-time counterfactuals;
  a planted-event benchmark for the selection mechanism and attributions;
  invariance tests;
* **inference** on a video or array with one checkpoint or an ensemble
  (`scripts/infer.py`, `src/inference_api.py`).

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
python -m pytest tests -q            # synthetic data only, about 15 s
```

Score a stored landmark array or a video with the released ensemble:

```bash
python scripts/infer.py --input_npy processed/features/asd_1.npy --checkpoint models/release --output outputs/asd_1
python scripts/infer.py --input walk.mp4 --checkpoint models/release --output outputs/walk
```

```python
import sys; sys.path.insert(0, "src")
from inference_api import PACEASDPredictor
pred = PACEASDPredictor("models/release")
r = pred.predict_npy("processed/features/asd_1.npy")
print(r["probability"], r["probability_sd"], r["selection"]["selection_is_trivial"])
```

The released models were trained on every child of the demonstration cohort;
use them to score new recordings, not to estimate performance on this cohort.

---

## Reproducing the analyses

| Step | Command | Output |
|---|---|---|
| 1. Landmarks from the Dryad videos (download separately) | `python src/preprocess.py --raw_dir data/raw/Dataset --out_dir processed --workers 6` | `processed/keypoints/`, `processed/features/`, `processed/video_metadata.csv` |
| 2. Inventory and duplicate check | `python src/audit.py --raw_dir data/raw/Dataset --arrays processed/features` | report, duplicate groups |
| 3. Tests | `python -m pytest tests -q` | |
| 4. Feature classifiers and controls | `python scripts/run_feature_models.py` | `results/cv/runs/<model>/` |
| 5. Networks (pilot, PACE-ASD family, comparison architectures) | `python scripts/run_all_cv.py --stage pilot`, then `--stage pace`, then `--stage comparison` | `results/cv/runs/<arm>/`, `models/cv/` |
| 6. Planted-event benchmark | `python scripts/synthetic_events.py --jobs 2` then `--summarize` | `results/synthetic/` |
| 7. Summaries | `python scripts/analyze_cv.py` and `python scripts/analyze_mechanisms.py` | `results/cv/summary.json`, `results/cv/mechanisms.json` |
| 8. Released ensemble | `python scripts/train_release.py` | `models/release/` |
| 9. Rerun check | `python scripts/check_reproducibility.py` | `results/cv/reproducibility.json` |

All settings are in `configs/config.yaml`; the partition is frozen in
`splits/cv_partition.json`. Every run is seeded and uses deterministic GPU
kernels; runs resume at the level of an outer fold.

<!-- RESULTS_TABLE -->
### Results on the demonstration cohort

Out-of-fold AUC (95% interval over children), 1 x 5-fold nested cross-validation over 95 children; 'cropped' = the 15 autistic and 45 typically developing children whose videos were cropped (the only frame format both groups share). Source: `results/cv/summary.json`, `results/cv/mechanisms.json`.

| Configuration | AUC, all children | AUC, cropped videos |
|---|---|---|
| controls: clip duration | 0.644 (0.52-0.75) | 0.493 (0.33-0.70) |
| controls: validity pattern | 0.648 (0.53-0.75) | 0.566 (0.41-0.73) |
| controls: age and sex | 0.699 (0.58-0.80) | 0.655 (0.51-0.79) |
| controls: frame format | 0.884 (0.80-0.95) | 0.671 (0.54-0.80) |
| controls: all of the above | 0.905 (0.84-0.96) | - |
| controls: validity-only network | 0.639 (0.53-0.74) | - |
| gait-descriptor classifiers: logistic regression | 0.843 (0.75-0.92) | 0.962 (0.92-0.99) |
| gait-descriptor classifiers: support vector machine | 0.848 (0.76-0.92) | 0.944 (0.89-0.98) |
| gait-descriptor classifiers: random forest | 0.884 (0.81-0.95) | 0.973 (0.93-1.00) |
| gait-descriptor classifiers: gradient-boosted trees | 0.870 (0.79-0.94) | 0.950 (0.90-0.99) |
| gait-descriptor classifiers: logistic regression, image-normalized | 0.787 (0.69-0.88) | 0.951 (0.90-0.99) |
| comparison networks: BiGRU | 0.864 (0.78-0.93) | - |
| reference network and ablations: reference (L=15, M=8) | 0.892 (0.82-0.96) | 0.957 (0.91-0.99) |
| reference network and ablations: no gate | 0.873 (0.79-0.95) | 0.952 (0.91-0.99) |
| reference network and ablations: frame selection (L=1, M=120) | 0.899 (0.83-0.96) | 0.956 (0.91-0.99) |
| reference network and ablations: budget M=4 | 0.897 (0.82-0.96) | 0.962 (0.92-0.99) |
| reference network and ablations: averaging instead of attention | 0.887 (0.81-0.95) | 0.953 (0.90-0.99) |
| reference network and ablations: no validity masking | 0.878 (0.80-0.95) | 0.948 (0.90-0.98) |
| reference network and ablations: no onset alignment | 0.880 (0.80-0.95) | - |
| reference network and ablations: epoch selection by validation loss | 0.890 (0.81-0.95) | - |

Frame format alone separates the groups in this dataset; evidence about movement comes from the cropped videos. Nothing here establishes clinical validity.
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
