# Inference outputs

```bash
python scripts/infer.py --input VIDEO | --input_npy ARRAY --checkpoint CKPT [CKPT ...] | DIR \
       [--output DIR] [--device auto|cpu|cuda] [--no_attribution] [--no_figures]
```

`--checkpoint` takes one `.pt` file, several, or a directory (all `.pt` files in
it are scored as an ensemble). The input goes through exactly the preparation
used in training: for video, MediaPipe Pose and pixel-unit normalisation
(`src/preprocess.py`); then the plausibility check and onset alignment
(`src/sequence.py`). Frame numbers in the outputs refer to the input.

## result.json

| Key | Meaning |
|---|---|
| `probability` | mean recalibrated probability over ensemble members |
| `probability_sd`, `member_probabilities`, `member_logits` | spread and individual outputs of the members |
| `threshold`, `above_threshold` | decision threshold (default 0.5) and whether `probability` reaches it |
| `n_detected_frames` | frames of the input with a detected pose |
| `n_frames_rejected_implausible` | detected frames set to undetected by the plausibility check |
| `onset_frame` | first detected frame of the input; outputs are shifted back by this amount |
| `model_seconds` | time for the networks, preparation included |
| `video` | for video input: frame rate, width, height, frames read and detected, pose-extraction seconds, frames truncated beyond 300 |
| `checkpoints` | files used |

## selected_events.json

| Key | Meaning |
|---|---|
| `block_size`, `budget_blocks` | L and M of the network |
| `n_valid_blocks` | blocks of the prepared sequence containing a detected frame |
| `selection_is_trivial` | true when `n_valid_blocks <= budget_blocks`: every valid block is kept and the gate does not select |
| `blocks[]` | per block: `input_frames` (first and last frame in the input), `valid_frames`, `selected_by_fraction_of_models`, `mean_gate_score` |
| `per_model` | indices of the blocks each member selected |

A low `selected_by_fraction_of_models` across members means the selection is
not stable for this recording; selected blocks should not be read as
behavioral events without the checks described in the article
(`scripts/synthetic_events.py`, `scripts/analyze_mechanisms.py --sections selection`).

## attribution.json

Descriptor-level gradient x input, averaged over members: `stream_share`
(position, velocity, acceleration), `region_share_velocity_acceleration`
(head, arms and hands, hips, legs and feet) and `landmark_share` (33 values).
These describe the networks' sensitivity for this recording, not the movement.

## kinematics.npz

`positions`, `velocities`, `accelerations` (300, 33, 2) exactly as the network
computes them (velocity = 10 x first difference, acceleration = 5 x difference
of velocity, zero across undetected frames), `valid` (300,), and `input_frame`
(300,), the input frame number of each row.

## visualizations/

`kinematics.png` (wrist and ankle traces) and `selection.png` (fraction of
members selecting each block).
