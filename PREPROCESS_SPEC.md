# Preprocessing specification

Every model in this repository consumes the same arrays, produced as follows
(`src/preprocess.py`, then `src/sequence.py` at load time).

## 1. Input videos

| Group | Path under the Dryad root | Videos |
|---|---|---|
| Autistic children | `Autism/children with ASD/{1..50}/video/video.avi` | 50 |
| Typically developing children | `Typical/{1..50}/video/video.avi` (`video1.avi` for child 2) | 50 |
| Severe autism (not analysed) | `Autism/Severe level of ASD/case{1..9}/*.avi` | 10 |

Skeleton and trajectory videos (`Svideo.avi`, `Tvideo.avi`) and the augmented
depth-camera data are not used. All analysed videos are 60 frames per second,
portrait orientation; their frame size varies (see section 6).

## 2. Pose estimation

MediaPipe Pose 0.10.14, `model_complexity=2`, `smooth_landmarks=True`,
`min_detection_confidence=0.5`, `min_tracking_confidence=0.5`, video mode.
For each frame the 33 landmarks' image-normalised (x, y) are kept (depth and
visibility discarded); frames without a detection are zero. Stored unchanged
in `processed/keypoints/<clip_id>.npy` with shape `(n_frames, 33, 2)`.

## 3. Normalisation (per frame with a detection)

1. Pixel units: `x *= width`, `y *= height`.
2. Centre on the hip midpoint (landmarks 23, 24).
3. Divide by the distance between the shoulder landmarks (11, 12), bounded
   below by 1e-5.

Step 1 matters: MediaPipe divides the two axes by different lengths, so
without it the geometry depends on the frame's aspect ratio.

## 4. Length

Truncate or zero-pad at the end to T = 300 frames (5 s at 60 frames per
second). Stored as `processed/features/<clip_id>.npy`, `(300, 33, 2)` float32.

## 5. Preparation at load time (`src/sequence.py`)

1. Plausibility check: a frame with any coordinate beyond 10 shoulder widths is
   set to zero (treated as undetected).
2. Onset alignment: the sequence is shifted so that its first detected frame is
   frame 0.

A frame is detected (valid) when the sum of the absolute values of its
coordinates exceeds 1e-4.

## 6. Metadata

`processed/video_metadata.csv`: frame rate, width, height, frames read, frames
with a detection, and extraction time for each video.
`processed/participant_metadata.csv`: sex and age from the deposit's readme.
`processed/labels.csv`: clip and subject identifiers and labels.

## 7. Reproducibility

Pose estimation is deterministic for a given MediaPipe version and video.
Re-extracting the videos and applying the earlier per-axis normalisation
reproduces the arrays used before version 1.2 (identical validity masks; largest
coordinate difference 1e-4 shoulder widths; `results/regeneration_check.csv`).
