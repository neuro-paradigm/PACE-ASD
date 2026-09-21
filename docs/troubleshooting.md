# PACE-ASD — Troubleshooting Guide

## Common Issues & Solutions

### 1. `ModuleNotFoundError: No module named 'src'` or `No module named 'model'`

**Cause:** Python cannot find the `src/` directory.
**Solution:** Run scripts from the repository root, or ensure `src/` is in `PYTHONPATH`:
```bash
# Windows
set PYTHONPATH=%PYTHONPATH%;%CD%\src

# Linux/macOS
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
```
All scripts in `scripts/` automatically add `src/` to `sys.path`.

---

### 2. MediaPipe Fails or Video Decoding Errors

**Symptom:** `RuntimeError: Cannot open video` or MediaPipe import error.
**Cause:** OpenCV cannot read the video codec, or MediaPipe binary issue.
**Solution:**
- If you have pre-extracted `.npy` files, use `--input_npy` instead:
  ```bash
  python scripts/infer.py --input_npy processed/features/asd_1.npy --checkpoint ...
  ```
- Ensure video is in standard H.264 or MPEG-4 format.
- For MediaPipe on Windows, ensure Microsoft Visual C++ Redistributable 2015–2022 is installed.

---

### 3. Out of Memory (OOM) During Training

**Cause:** Batch size too large for available GPU RAM.
**Solution:** Reduce `batch_size` in `configs/config.yaml`:
```yaml
training:
  batch_size: 4  # was 8
```

---

### 4. Checkpoint Loading Errors

**Symptom:** `KeyError: 'state_dict'` or architecture mismatch.
**Cause:** The checkpoint was saved with a different model variant configuration.
**Solution:** Ensure the model variant matches:
- Full PACE-ASD: `use_gate=True, use_transformer=True` (A1)
- No-Gate: `use_gate=False, use_transformer=True` (A2)
- Frame-Gate: `event_block_size=1, event_top_m=120` (A3)
- No-Transformer: `use_gate=True, use_transformer=False` (A4)

---

### 5. `weights_only` Security Warning in PyTorch 2.4+

**Symptom:** `FutureWarning: You are using `torch.load` with `weights_only=False`...`
**Explanation:** Checkpoints contain the pickled `PlattScaler` object along with the model `state_dict`. `weights_only=False` is required to deserialize the calibration scaler. Only load checkpoints from trusted sources.

---

### 6. Tests Fail Due to Missing Modules

**Solution:** Ensure all requirements are installed:
```bash
pip install -r requirements.txt
pip install pytest pytest-cov
```
