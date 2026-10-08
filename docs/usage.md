# Usage

## Scoring recordings

```bash
python scripts/infer.py --input walk.mp4 --checkpoint models/release --output outputs/walk
python scripts/infer.py --input_npy processed/features/asd_1.npy --checkpoint models/release
```

```python
import sys; sys.path.insert(0, "src")
from inference_api import PACEASDPredictor

pred = PACEASDPredictor("models/release", device="cpu")   # directory, file or list of files
res = pred.predict_video("walk.mp4")                      # or predict_npy / predict_array
print(res["probability"], res["probability_sd"])
print(res["video"]["fps"], res["video"]["pose_seconds"])
for blk in res["selection"]["blocks"]:
    print(blk["input_frames"], blk["selected_by_fraction_of_models"])
```

See `docs/inference.md` for every output field.

## Preparing a dataset

```bash
python src/preprocess.py --raw_dir data/raw/Dataset --out_dir processed --workers 6
python src/audit.py --raw_dir data/raw/Dataset --arrays processed/features --json processed/duplicates.json
```

For another dataset, adapt `build_video_catalogue` in `src/preprocess.py` (one
entry per video with `clip_id`, `subject_id`, `label`, `group`, `video_path`)
and write `processed/labels.csv`. `scripts/run_cv.py`'s `cohort()` defines which
recordings form the evaluation cohort.

## Evaluating a configuration

```bash
python scripts/run_cv.py --arm PACE --save_checkpoints               # one arm
python scripts/run_cv.py --arm PACE --workers 3 --worker_id 0         # split across processes
python scripts/run_cv.py --arm TCN --set training.lr=0.001 --tag TCN@1e-3
python scripts/run_feature_models.py
python scripts/analyze_cv.py
python scripts/analyze_mechanisms.py --sections shortcuts confounds selection
```

Arms are defined in `src/protocol.py` (`ARMS`). Each outer fold is written as
`results/cv/runs/<tag>/r<repeat>_k<fold>.json` and contains the test children,
their labels, each inner model's recalibrated probabilities and logits for the
original input and the three test-time variants, and training details. Runs
skip folds whose file exists, so an interrupted run is resumed by repeating the
command.

## Checking a selection mechanism

```bash
python scripts/synthetic_events.py --amplitudes 0.05 0.1 --jobs 3
python scripts/synthetic_events.py --summarize
```

Copies of every recording with and without a planted event are classified in
cross-validation grouped by recording; the summary reports detection AUC, how
much of the event the selected blocks cover against random selection, whether
the top block hits the event, and attribution shares, for trained and randomly
initialised networks.
