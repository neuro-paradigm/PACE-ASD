# Troubleshooting

| Symptom | Cause | Remedy |
|---|---|---|
| `ModuleNotFoundError: cv2` or `cv2.__version__` missing after installing | both `opencv-python` and `opencv-contrib-python` installed, or one removed after the other | `pip uninstall -y opencv-python opencv-contrib-python` then `pip install opencv-contrib-python==4.11.0.86` |
| MediaPipe downloads a model on first use | `model_complexity: 2` uses `pose_landmark_heavy.tflite`, which is not in the wheel | allow the download once, or use a machine with network access to populate the package directory |
| Training of TCN or CTR-GCN is very slow on the GPU | deterministic cuDNN kernels for dilated convolutions | already handled: `DilatedTemporalConv` computes dilated convolutions as undilated ones over interleaved subsequences |
| `RuntimeError: Attempting to deserialize object on CUDA device` | checkpoint saved on GPU, loaded on a machine without one | pass `device="cpu"` (`--device cpu` in `scripts/infer.py`) |
| A run stopped part-way | interruption or a locked file on Windows | rerun the same command; finished outer folds (`results/cv/runs/<arm>/r*_k*.json`) are skipped |
| `selection_is_trivial: true` in `selected_events.json` | the recording has no more valid blocks than the budget | expected for short recordings; nothing is selected, every valid block is used |
| Many frames reported as rejected | normalised coordinates beyond 10 shoulder widths (shoulders nearly coincide in the image, e.g. the person turned sideways) | inspect the video; the frames are treated as undetected |
| Groups differ in frame size, frame rate, duration or onset | recording or preparation differs between groups | tabulate `processed/video_metadata.csv` by group before training; see the controls in `scripts/run_feature_models.py` |
