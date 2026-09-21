"""
PACE-ASD — Test fixtures and shared utilities.
All tests use synthetic data only. No dataset files required.
"""

import os
import sys
import pytest
import numpy as np
import torch

# Ensure src/ is importable from tests/
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)


@pytest.fixture
def minimal_config():
    """Minimal config dict matching the model architecture from configs/config.yaml."""
    return {
        "model": {
            "spatial_dim": 64,      # Smaller than production for fast tests
            "conv1d_channels": 16,
            "dropout": 0.0,
            "event_block_size": 15,
            "event_top_m": 4,
            "transformer_heads": 2,
            "transformer_layers": 1,
        },
        "training": {
            "batch_size": 2,
            "num_workers": 0,
        },
        "calibration": {
            "lr": 0.01,
            "max_iter": 10,
        },
        "output": {
            "models_dir": "models",
            "results_dir": "results",
            "splits_file": "splits/splits_dryad_v2_dedup.json",
        },
        "data": {
            "processed_dir": "processed",
        }
    }


@pytest.fixture
def synthetic_pose_batch():
    """Synthetic pose batch: (2, 300, 33, 2) — batch of 2 with realistic values."""
    torch.manual_seed(0)
    batch = torch.zeros(2, 300, 33, 2)
    # Only first 150 frames are 'valid' (non-zero)
    batch[:, :150, :, :] = torch.randn(2, 150, 33, 2) * 0.5
    return batch


@pytest.fixture
def synthetic_pose_numpy():
    """Synthetic pose array: (300, 33, 2) float32 — single subject."""
    np.random.seed(42)
    arr = np.zeros((300, 33, 2), dtype=np.float32)
    arr[:150] = np.random.randn(150, 33, 2).astype(np.float32) * 0.5
    return arr


@pytest.fixture
def device():
    return torch.device("cpu")  # CPU only for tests
