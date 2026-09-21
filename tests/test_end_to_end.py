"""
End-to-end inference test using synthetic data only.
Does NOT require real dataset files or a trained checkpoint.
"""

import os
import sys
import json
import pickle
import tempfile

import numpy as np
import pytest
import torch

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)


@pytest.fixture
def minimal_config():
    return {
        "model": {
            "spatial_dim": 64,
            "conv1d_channels": 16,
            "dropout": 0.0,
            "event_block_size": 15,
            "event_top_m": 4,
            "transformer_heads": 2,
            "transformer_layers": 1,
        },
        "inference": {
            "use_gate": True,
            "use_transformer": True,
            "device": "cpu",
            "threshold": 0.5,
            "calibrate": True,
            "save_kinematics": True,
            "save_events": True,
            "save_attribution": True,
            "save_visualizations": False,  # skip viz in CI
        },
        "training": {"batch_size": 2, "num_workers": 0},
        "calibration": {"lr": 0.01, "max_iter": 10},
        "data": {"processed_dir": "processed"},
        "output": {
            "models_dir": "models",
            "results_dir": "results",
            "splits_file": "splits/splits_dryad_v2_dedup.json",
        },
        "preprocessing": {
            "mediapipe_model_complexity": 2,
            "mediapipe_smooth_landmarks": True,
            "mediapipe_min_detection_confidence": 0.5,
            "mediapipe_min_tracking_confidence": 0.5,
        },
    }


def make_synthetic_checkpoint(cfg, tmp_dir):
    """Create a minimal valid checkpoint file with synthetic weights."""
    from model import ASDMotionModel
    from calibration import PlattScaler
    model = ASDMotionModel(cfg, use_gate=True, use_transformer=True)
    model.eval()
    scaler = PlattScaler()
    scaler.fit(
        np.array([-1.0, -0.5, 0.5, 1.0], dtype=np.float32),
        np.array([0, 0, 1, 1], dtype=np.float32),
    )
    ckpt = {
        "epoch":      1,
        "state_dict": model.state_dict(),
        "metrics":    {"accuracy": 0.7},
        "config":     cfg,
        "scaler":     pickle.dumps(scaler),
        "threshold":  0.5,
    }
    ckpt_path = os.path.join(tmp_dir, "test_checkpoint.pt")
    torch.save(ckpt, ckpt_path)
    return ckpt_path


def test_inference_api_predict_npy(minimal_config, tmp_path):
    """PACEASDPredictor.predict_npy() returns expected keys."""
    from calibration import PlattScaler
    from inference_api import PACEASDPredictor
    import yaml

    # Write config to temp file
    cfg_path = str(tmp_path / "inference.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(minimal_config, f)

    # Write synthetic checkpoint
    ckpt_path = make_synthetic_checkpoint(minimal_config, str(tmp_path))

    # Write synthetic .npy
    npy_path = str(tmp_path / "synthetic_subject.npy")
    arr = np.zeros((300, 33, 2), dtype=np.float32)
    arr[:150] = np.random.randn(150, 33, 2).astype(np.float32) * 0.5
    np.save(npy_path, arr)

    predictor = PACEASDPredictor(checkpoint=ckpt_path, config=cfg_path, device="cpu")
    result = predictor.predict_npy(npy_path)

    # Check required output keys
    assert "calibrated_probability" in result
    assert "raw_probability" in result
    assert "prediction" in result
    assert "selected_events" in result
    assert "bilateral_asymmetry" in result
    assert result["prediction"] in ("ASD", "TD")
    assert 0.0 <= result["calibrated_probability"] <= 1.0
    assert 0.0 <= result["raw_probability"] <= 1.0


def test_inference_produces_result_json(minimal_config, tmp_path):
    """infer.py --input_npy saves result.json with required fields."""
    import yaml
    import importlib.util

    cfg_path = str(tmp_path / "inference.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(minimal_config, f)

    ckpt_path = make_synthetic_checkpoint(minimal_config, str(tmp_path))

    npy_path = str(tmp_path / "synthetic.npy")
    arr = np.zeros((300, 33, 2), dtype=np.float32)
    arr[:120] = np.random.randn(120, 33, 2).astype(np.float32) * 0.5
    np.save(npy_path, arr)

    out_dir = str(tmp_path / "outputs")

    # Load and run infer.py
    SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
    spec = importlib.util.spec_from_file_location("infer", os.path.join(SCRIPTS_DIR, "infer.py"))
    infer_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(infer_mod)

    class FakeArgs:
        input      = None
        input_npy  = npy_path
        checkpoint = ckpt_path
        config     = cfg_path
        output     = out_dir

    result = infer_mod.run_inference(FakeArgs())

    # Check result.json was written
    result_path = os.path.join(out_dir, "result.json")
    assert os.path.isfile(result_path), "result.json not found"

    with open(result_path) as f:
        saved = json.load(f)

    assert "calibrated_probability" in saved
    assert "raw_probability" in saved
    assert "prediction" in saved
    assert "bilateral_asymmetry" in saved
    assert 0.0 <= saved["calibrated_probability"] <= 1.0


def test_kinematics_npz_saved(minimal_config, tmp_path):
    """infer.py saves kinematics.npz with correct arrays."""
    import yaml
    import importlib.util

    cfg_path = str(tmp_path / "inference.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(minimal_config, f)

    ckpt_path = make_synthetic_checkpoint(minimal_config, str(tmp_path))

    npy_path = str(tmp_path / "synthetic2.npy")
    arr = np.random.randn(300, 33, 2).astype(np.float32) * 0.5
    np.save(npy_path, arr)

    out_dir = str(tmp_path / "out2")

    SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
    spec = importlib.util.spec_from_file_location("infer", os.path.join(SCRIPTS_DIR, "infer.py"))
    infer_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(infer_mod)

    class FakeArgs:
        input = None; input_npy = npy_path
        checkpoint = ckpt_path; config = cfg_path; output = out_dir

    infer_mod.run_inference(FakeArgs())

    kinem_path = os.path.join(out_dir, "kinematics.npz")
    assert os.path.isfile(kinem_path)
    data = np.load(kinem_path)
    assert "positions" in data
    assert "velocities" in data
    assert "accelerations" in data
    assert data["positions"].shape == (300, 33, 2)
    assert data["velocities"].shape == (300, 33, 2)
    assert data["accelerations"].shape == (300, 33, 2)
