# PACE-ASD Checkpoints

Pre-trained model checkpoints for PACE-ASD and ablation variants are stored in `models/`.

## Checkpoint Locations

Checkpoints are stored at:
```
models/
├── A1/    # Full PACE-ASD (fold1-3 × seed42-61 = 60 checkpoints)
├── A2/    # No-Block-ESG ablation
├── A3/    # Frame-granularity gate ablation
├── A4/    # No-transformer ablation
└── A5_*/  # 14 baseline model checkpoints
```

## Checkpoint Format

Each `.pt` file is a PyTorch checkpoint saved with `torch.save()` containing:

```python
{
    "epoch":      int,          # Best epoch (early stopping)
    "state_dict": dict,         # Model weights (OrderedDict)
    "metrics":    dict,         # Best validation metrics at save time
    "config":     dict,         # Training configuration
    "scaler":     bytes,        # Pickle-serialized PlattScaler object
    "threshold":  float,        # Decision threshold (0.5)
}
```

## Loading a Checkpoint

```python
import torch, pickle, sys, yaml
sys.path.insert(0, 'src')
from model import ASDMotionModel
from calibration import PlattScaler

# Load
checkpoint = torch.load('models/A1/fold1_seed42.pt', map_location='cpu', weights_only=False)
config = checkpoint['config']

# Build model
model = ASDMotionModel(config, use_gate=True, use_transformer=True)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Load Platt scaler
scaler = pickle.loads(checkpoint['scaler'])
```

## Manuscript Checkpoint

The checkpoints in `models/A1/` correspond to the Full PACE-ASD model (A1) reported in the manuscript.

- Architecture: `use_gate=True, use_transformer=True`
- Configuration: `configs/config.yaml`
- Training: 3-fold × 20 seeds = 60 independent runs
- Split file: `splits/splits_dryad_v2_dedup.json`

For single-video inference, any of the A1 checkpoints can be used. `fold1_seed42.pt` is recommended as the reference checkpoint.

## Retraining from Scratch

If checkpoints are unavailable (e.g., you need to verify from scratch):

```bash
# Single fold, single seed (quick sanity check, ~5-10 min on GPU)
python src/train.py --config configs/config.yaml --model_id A1 --seed 42 --fold 0

# Full 20-seed ablation (all A1-A4 variants, ~8-16 hours on GPU)
python src/ablation.py --config configs/config.yaml --models A1
```

See `docs/reproducibility.md` for complete retraining instructions.
