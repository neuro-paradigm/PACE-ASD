# PREPROCESS_SPEC.md — Preprocessing Specification Lock

**Status:** Locked before any model training.  
**Purpose:** Ensures every ablation arm (A1–A5) consumes identically preprocessed inputs (Protocol Section 2).

---

## 1. Input Videos

| Group | Source path | N videos |
|---|---|---|
| ASD children | `D:/dryad/Autism/children with ASD/{1..50}/video/video.avi` | 50 |
| Typical children | `D:/dryad/Typical/{1..50}/video/video.avi` | 50 |
| Severe ASD (supplement) | `D:/dryad/Autism/Severe level of ASD/case{1..9}/*.avi` | 10 |

Augmentation clips (`augmentation/`) are **not used**.

---

## 2. MediaPipe Configuration

| Parameter | Value |
|---|---|
| `model_complexity` | 2 (full accuracy) |
| `static_image_mode` | False (video mode, smoothed tracking) |
| `smooth_landmarks` | True |
| `min_detection_confidence` | 0.5 |
| `min_tracking_confidence` | 0.5 |
| Package | `mediapipe==0.10.14` |

Landmark set: **MediaPipe Pose** — 33 landmarks, normalized (x, y) ∈ [0, 1].  
Z-coordinate and visibility score are **discarded**.

---

## 3. Per-Frame Normalization

Applied identically to every frame where at least one landmark is non-zero.

### 3.1 Centering
Subtract the **mid-hip origin** from every landmark:

```
mid_hip = (landmark[23] + landmark[24]) / 2
x_centred[t, j] = x[t, j] - mid_hip[t]
```

Landmarks 23 = left_hip, 24 = right_hip.

> **Note:** The previous implementation centred on landmark 0 (nose). This has been corrected to mid-hip as specified in the manuscript.

### 3.2 Scale normalization
Divide by the inter-shoulder distance (clamped ≥ 1e-5):

```
shoulder_dist = ‖landmark[11] - landmark[12]‖₂
x_norm[t] = x_centred[t] / max(shoulder_dist, 1e-5)
```

Landmarks 11 = left_shoulder, 12 = right_shoulder.

---

## 4. Sequence Length

- Target length: **T = 300 frames** (10 seconds at 30 fps)
- Frames beyond 300: **truncated** (keep first 300)
- Sequences shorter than 300: **zero-padded** at the end

Frames where MediaPipe fails to detect a pose are stored as all-zeros.

---

## 5. Output Format

- File: `processed/features/{clip_id}.npy`
- Shape: `(300, 33, 2)` — float32
- Content: normalized (x, y) coordinates, mid-hip centred, inter-shoulder scaled
- Zeros: padding frames and failed-detection frames

### Feature dimensionality
Velocity and acceleration are computed **at runtime** inside `SpatialEncoder`:

```
per-frame descriptor = [position(66) | velocity(66) | acceleration(66)]
D_c = 33 × 2 × 3 = 198
```

This matches the manuscript's stated D_c = 198. ✓

---

## 6. Labels File

`processed/labels.csv` — columns:

| Column | Type | Description |
|---|---|---|
| `clip_id` | str | Unique clip identifier (e.g. `asd_1`, `td_23`, `severe_case2_v1`) |
| `subject_id` | str | Subject identifier (e.g. `asd_1`, `severe_case2`) |
| `label` | int | 1 = ASD, 0 = TD |
| `group` | str | `regular` (train/val/test pool) or `supplement` (9 severe-ASD) |

---

## 7. Clip ID Convention

| Source | clip_id | subject_id |
|---|---|---|
| ASD subject N, main video | `asd_N` | `asd_N` |
| TD subject N, main video | `td_N` | `td_N` |
| Severe case N, video i | `severe_caseN_vi` | `severe_caseN` |

---

## 8. Reproducibility

Preprocessing is deterministic given fixed MediaPipe version and input video.  
Run `python src/preprocess.py --dry_run` to list all clips before processing.  
Outputs are cached — re-running skips already-processed clips.
