# PACE-ASD — Inference & Output Documentation

## Running Inference

```bash
# Pre-extracted .npy (preferred — no MediaPipe required)
python scripts/infer.py \
  --input_npy processed/features/asd_1.npy \
  --checkpoint models/A1/fold1_seed42.pt \
  --config configs/inference.yaml \
  --output outputs/asd1_result

# Raw video
python scripts/infer.py \
  --input path/to/video.mp4 \
  --checkpoint models/A1/fold1_seed42.pt \
  --config configs/inference.yaml \
  --output outputs/video_result
```

## Output Directory Structure

```
outputs/example/
├── result.json              # Primary prediction output
├── kinematics.npz           # Kinematic feature arrays
├── selected_events.json     # Block-ESG selected events
├── attribution.json         # Body-region and stream attribution
└── visualizations/
    ├── kinematics.png         # Position/velocity/acceleration plots
    ├── event_saliency.png     # Block saliency bar chart
    └── evidence_summary.png   # Multi-panel evidence summary
```

## Output File Reference

### result.json

Primary prediction output. All fields:

| Field | Type | Description |
|---|---|---|
| `clip_id` | str | Identifier derived from input filename |
| `input_source` | str | Path to input file |
| `checkpoint` | str | Absolute path to checkpoint used |
| `prediction` | str | Binary prediction: `"ASD"` or `"TD"` |
| `raw_probability` | float | Uncalibrated sigmoid probability P(ASD) ∈ [0,1] |
| `calibrated_probability` | float | Platt-calibrated probability P(ASD) ∈ [0,1] |
| `threshold` | float | Decision threshold (default: 0.5) |
| `has_platt_scaler` | bool | Whether Platt calibration was applied |
| `body_region_attribution` | dict | Gradient×Input attribution per body region (fraction, sums to 1) |
| `kinematic_stream_attribution` | dict | Attribution split across position/velocity/acceleration |
| `bilateral_asymmetry` | dict | Descriptive left-right movement difference (not an ASD biomarker) |
| `selected_events_summary` | dict | Summary of Block-ESG event selection |
| `runtime_seconds` | float | Total wall-clock time for this inference run |

**body_region_attribution** subfields:
- `head`: attribution fraction for landmarks 0–10 (nose, eyes, ears, mouth)
- `arms`: attribution fraction for landmarks 11–22 (shoulders, elbows, wrists, hands)
- `torso`: attribution fraction for landmarks 23–24 (hips)
- `legs`: attribution fraction for landmarks 25–32 (knees, ankles, feet)

**bilateral_asymmetry** subfields:
- `pairs`: per joint-pair mean absolute left-right difference (shoulder, elbow, wrist, hip, knee, ankle)
- `mean_asymmetry`: mean across all pairs
- `note`: disclaimer that this is descriptive and not a validated ASD biomarker

### kinematics.npz

NumPy compressed archive. Load with `np.load('kinematics.npz')`.

| Array | Shape | Description |
|---|---|---|
| `positions` | (300, 33, 2) float32 | Normalised landmark positions (hip-centred, shoulder-scaled) |
| `velocities` | (300, 33, 2) float32 | Frame-to-frame velocity (×10 scaling, 0 for invalid frames) |
| `accelerations` | (300, 33, 2) float32 | Frame-to-frame acceleration (×5 scaling, 0 for invalid frames) |
| `frame_indices` | (300,) int | Frame indices [0, 1, 2, ..., 299] |

**Coordinate system:** hip-centred (left+right hip midpoint = origin), inter-shoulder distance = 1.0.
**Padding:** frames where MediaPipe failed to detect a pose are stored as all-zeros.

### selected_events.json

Block-ESG temporal event selection.

| Field | Type | Description |
|---|---|---|
| `block_size_frames` | int | Number of frames per block (default: 15) |
| `num_selected_blocks` | int | Number of blocks selected (default: 8) |
| `selected_frame_indices` | list[int] | All frame indices in selected blocks |
| `selected_block_starts` | list[int] | Start frame of each selected block |
| `selected_block_ranges` | list[[start, end]] | [start, end] for each selected block |
| `all_block_saliency_scores` | list[float] | Raw gate score for every candidate block |

### attribution.json

Gradient×Input attribution decomposition.

| Field | Type | Description |
|---|---|---|
| `body_region_attribution` | dict | Per-region importance fraction |
| `kinematic_stream_attribution` | dict | Per-stream (position/velocity/acceleration) importance fraction |
| `joint_attribution` | list[float] | Per-joint importance fraction (33 values, sum = 1) |

**Important:** Attribution values are computed via Gradient×Input with respect to the raw logit. They reflect which landmarks and kinematic streams had the largest gradient-weighted activation for this specific input. They are NOT guaranteed to be globally consistent across subjects.

## Limitations

1. **Single subject, single video:** PACE-ASD operates on one video at a time. Subject-level aggregation (mean logit across multiple clips) is applied only during training evaluation.
2. **Pre-trained checkpoint required:** The model must be trained first or downloaded from the repository releases.
3. **T_MAX=300 frames:** Videos longer than 10 seconds at 30fps are truncated. Shorter videos are zero-padded.
4. **No ground-truth label:** Inference outputs a research probability score; clinical diagnosis requires a licensed professional.
5. **Calibration validity:** Platt calibration is fitted on the out-of-fold validation set (~22 subjects). Calibration quality may vary for out-of-distribution recordings.
