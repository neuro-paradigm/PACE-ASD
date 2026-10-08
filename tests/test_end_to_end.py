"""
End-to-end inference with synthetic data and synthetic checkpoints only:
the Python API, ensembles, the command-line tool and its output files.
"""

import argparse
import copy
import importlib.util
import json
import os
import pickle
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


@pytest.fixture
def cfg():
    return {
        "model": {"spatial_dim": 64, "conv1d_channels": 16, "dropout": 0.0,
                  "event_block_size": 15, "event_top_m": 4, "transformer_heads": 2,
                  "transformer_layers": 1, "mask_padding": True, "align_onset": True},
        "data": {"max_abs_coord": 10.0, "align_onset": True},
    }


def make_checkpoint(cfg, path, seed=0, pickled_scaler=False):
    from calibration import PlattScaler
    from model import ASDMotionModel
    torch.manual_seed(seed)
    model = ASDMotionModel(cfg).eval()
    scaler = PlattScaler()
    scaler.fit(np.array([-1.0, -0.5, 0.5, 1.0], np.float32), np.array([0, 0, 1, 1], np.float32))
    sc = pickle.dumps(scaler) if pickled_scaler else {
        "temperature": float(scaler.temperature), "bias": float(scaler.bias)}
    torch.save({"state_dict": model.state_dict(), "config": cfg, "arm": "PACE", "scaler": sc}, path)
    return path


def clip(n=120, lead=0, seed=1):
    rng = np.random.default_rng(seed)
    a = np.zeros((300, 33, 2), np.float32)
    a[lead:lead + n] = rng.normal(0, 0.5, (n, 33, 2))
    return a


def test_api_single_and_ensemble(cfg, tmp_path):
    from inference_api import PACEASDPredictor
    p1 = make_checkpoint(cfg, str(tmp_path / "m1.pt"), 0)
    p2 = make_checkpoint(cfg, str(tmp_path / "m2.pt"), 1, pickled_scaler=True)
    npy = str(tmp_path / "c.npy"); np.save(npy, clip())
    single = PACEASDPredictor(p1, device="cpu").predict_npy(npy)
    ens = PACEASDPredictor(str(tmp_path), device="cpu").predict_npy(npy)
    assert ens["n_models"] == 2 and single["n_models"] == 1
    assert 0 <= ens["probability"] <= 1
    assert np.isclose(ens["probability"], np.mean(ens["member_probabilities"]), atol=1e-6)
    assert np.isclose(ens["member_probabilities"][0], single["probability"], atol=1e-6)
    sel = ens["selection"]
    assert sel["n_valid_blocks"] == 8 and not sel["selection_is_trivial"]
    assert all(0 <= b["selected_by_fraction_of_models"] <= 1 for b in sel["blocks"])
    assert set(ens["attribution"]["stream_share"]) == {"position", "velocity", "acceleration"}


def test_api_reports_onset_and_is_invariant_to_it(cfg, tmp_path):
    from inference_api import PACEASDPredictor
    pred = PACEASDPredictor(make_checkpoint(cfg, str(tmp_path / "m.pt")), device="cpu")
    a = pred.predict_array(clip(lead=0), attribution=False)
    b = pred.predict_array(clip(lead=40), attribution=False)
    assert b["onset_frame"] == 40 and a["onset_frame"] == 0
    assert np.isclose(a["probability"], b["probability"], atol=1e-6)
    first_a = a["selection"]["blocks"][0]["input_frames"][0]
    first_b = b["selection"]["blocks"][0]["input_frames"][0]
    assert first_b - first_a == 40


def test_api_rejects_implausible_frames(cfg, tmp_path):
    from inference_api import PACEASDPredictor
    pred = PACEASDPredictor(make_checkpoint(cfg, str(tmp_path / "m.pt")), device="cpu")
    a = clip()
    a[10, 0, 0] = 50.0
    r = pred.predict_array(a, attribution=False)
    assert r["n_frames_rejected_implausible"] == 1


def _run_cli(argv):
    spec = importlib.util.spec_from_file_location("infer", os.path.join(ROOT, "scripts", "infer.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ns = argparse.Namespace(input=None, input_npy=None, checkpoint=None, config=None,
                            output=None, device="cpu", no_attribution=False, no_figures=True)
    for k, v in argv.items():
        setattr(ns, k, v)
    return mod.run(ns)


def test_cli_writes_all_outputs(cfg, tmp_path):
    ck = make_checkpoint(cfg, str(tmp_path / "m.pt"))
    npy = str(tmp_path / "c.npy"); np.save(npy, clip(lead=12))
    out = str(tmp_path / "out")
    _run_cli({"input_npy": npy, "checkpoint": [ck], "output": out})
    saved = json.load(open(os.path.join(out, "result.json")))
    assert 0 <= saved["probability"] <= 1 and saved["onset_frame"] == 12
    ev = json.load(open(os.path.join(out, "selected_events.json")))
    assert ev["block_size"] == 15 and ev["blocks"]
    att = json.load(open(os.path.join(out, "attribution.json")))
    assert abs(sum(att["stream_share"].values()) - 1) < 1e-3
    k = np.load(os.path.join(out, "kinematics.npz"))
    assert k["positions"].shape == k["velocities"].shape == k["accelerations"].shape == (300, 33, 2)
    assert int(k["input_frame"][0]) == 12


def test_legacy_checkpoint_without_validity_keys_loads(cfg, tmp_path):
    """Checkpoints whose configuration predates the validity keys load and run."""
    from inference_api import PACEASDPredictor
    legacy = copy.deepcopy(cfg)
    legacy["model"].pop("mask_padding"); legacy["model"].pop("align_onset"); legacy.pop("data")
    r = PACEASDPredictor(make_checkpoint(legacy, str(tmp_path / "old.pt"), pickled_scaler=True),
                         device="cpu").predict_array(clip(), attribution=False)
    assert 0 <= r["probability"] <= 1 and r["onset_frame"] == 0


def test_display_path_on_another_drive():
    """Checkpoints on another drive than the working directory (as on Windows
    CI runners) must not make inference fail."""
    from inference_api import display_path
    other = "Z:\models\m.pt" if os.name == "nt" else "/tmp/m.pt"
    assert display_path(other)
