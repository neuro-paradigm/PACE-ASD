# ASDMotion — ASD Risk Prediction from Video Motion Analysis

Deep learning pipeline for predicting Autism Spectrum Disorder (ASD) risk from video-based skeletal motion analysis.

## Architecture

```text
Raw Video → MediaPipe Pose Extraction → Spatial Reconstruction → Savitzky-Golay Smoothing
         → CNN Spatial Encoder (ResNet18, 256-dim)
         → Microkinetic Encoder (Conv1D k=1,3,5)
         → Saliency Gate (Sparse Top-K)
         → Temporal Event Transformer (Masked Self-Attention)
         → Calibration Layer (Temperature Scaling)
         → ASD Risk Prediction
```

---

## 1. Exact Environment Reproduction

For precise reproducibility, all experiments were conducted in the following hardware and software environment. 

**Hardware & Software:**
- **Python:** 3.12
- **PyTorch:** 2.5.1
- **CUDA:** 12.1
- **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU

**Installation:**
```bash
pip install -r requirements.txt
```

---

## 2. Fixed Random Seeds

To ensure completely deterministic training and evaluation runs, all experiments were run with **fixed seed 42**. The following seed enforcement strategy is strictly applied before model initialization:

```python
import torch
import random
import numpy as np

seed = 42

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
*All experiments reported in our findings were run with fixed seed 42.*

---

## 3. Subject-wise Split Transparency

Ensuring no data leakage is critical for healthcare AI. We strictly enforce **subject-level stratified splitting** to isolate external datasets and guarantee **no subject overlap** between the training, validation, and test sets.

To maximize transparency and guarantee reproducibility of our 3-fold cross-validation results, we provide the exact, static split files used during training. These files define the exact subject IDs allocated to each partition.

**Split Files Provided:**
```text
splits/
  ├── fold1_train.txt
  ├── fold1_val.txt
  ├── fold1_test.txt
  ├── fold2_train.txt
  ...
  └── test_subjects.txt
```

---

## 4. Config-Based Training

Hyperparameters are decoupled from the code and managed via configuration files, ensuring transparency in training parameters.

**To run the exact training pipeline used in our study:**
```bash
python train_reconstruct.py
```
*(This script bypasses dynamic split generation and loads the static partitions directly from the `splits/` directory.)*

**To run the test set evaluation on the finalized models:**
```bash
python evaluate.py
```

**To run batch inference on an external, hold-out dataset (e.g., Supadata):**
```bash
python predict_batch.py --models-dir reconstruct --output_csv supadata_predictions.csv
```

---

## Configuration

All hyperparameters are documented in `configs/config.yaml`. Key settings include:

| Parameter | Value | Description |
|---|---|---|
| `batch_size` | 8 | Minibatch size |
| `learning_rate` | 0.0001 | Initial learning rate |
| `training.epochs` | 100 | Max training epochs |
| `training.early_stopping_patience` | 15 | Early stop patience |
| `training.n_folds` | 3 | Cross-validation folds |
| `model.top_k` | 32 | Saliency gate sparse tokens |
| `data.max_frames` | 300 | Sequence length (10s @ 30fps) |
