"""Tests for full model forward-pass shapes for all ablation variants."""

import pytest
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


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
        }
    }


@pytest.mark.parametrize("variant,kwargs", [
    ("A1", {"use_gate": True,  "use_transformer": True}),
    ("A2", {"use_gate": False, "use_transformer": True}),
    ("A3", {"use_gate": True,  "use_transformer": True}),
    ("A4", {"use_gate": True,  "use_transformer": False}),
])
def test_model_forward_shape(minimal_config, variant, kwargs):
    """Model forward pass returns (B,) for probs and logits."""
    from model import ASDMotionModel
    B, T = 2, 120
    cfg = dict(minimal_config)
    if variant == "A3":
        cfg["model"] = dict(cfg["model"])
        cfg["model"]["event_block_size"] = 1
        cfg["model"]["event_top_m"] = 60
    m = ASDMotionModel(cfg, **kwargs)
    m.eval()
    x = torch.randn(B, T, 33, 2)
    with torch.no_grad():
        probs, logits = m(x)
    assert probs.shape  == (B,), f"{variant}: expected probs shape ({B},), got {probs.shape}"
    assert logits.shape == (B,), f"{variant}: expected logits shape ({B},), got {logits.shape}"
    assert (probs >= 0).all() and (probs <= 1).all(), f"{variant}: probs out of [0,1]"


def test_model_get_attention_maps(minimal_config):
    """get_attention_maps() returns valid tensors for A1."""
    from model import ASDMotionModel
    B, T = 1, 90
    m = ASDMotionModel(minimal_config, use_gate=True, use_transformer=True)
    m.eval()
    x = torch.randn(B, T, 33, 2)
    attn, indices, block_scores = m.get_attention_maps(x)
    assert attn is not None
    assert indices is not None
    assert block_scores is not None
    # attn should be 2D: (K, K) where K = M*L
    assert attn.ndim == 3  # (B, K, K)
    assert block_scores.ndim == 2  # (B, N_blocks)


def test_model_calibration_flag(minimal_config):
    """calibrate=True divides logits by temperature; doesn't change probs ordering."""
    from model import ASDMotionModel
    m = ASDMotionModel(minimal_config)
    m.eval()
    x = torch.randn(2, 60, 33, 2)
    with torch.no_grad():
        probs_uncal, logits_uncal = m(x, calibrate=False)
        probs_cal,   logits_cal   = m(x, calibrate=True)
    # With temperature=1.0 (init), calibrated == uncalibrated
    assert torch.allclose(probs_uncal, probs_cal, atol=1e-5)


def test_model_no_nan_output(minimal_config):
    """Model should not produce NaN output on valid input."""
    from model import ASDMotionModel
    m = ASDMotionModel(minimal_config)
    m.eval()
    x = torch.randn(2, 60, 33, 2)
    with torch.no_grad():
        probs, logits = m(x)
    assert not torch.isnan(probs).any(),  "NaN in probs"
    assert not torch.isnan(logits).any(), "NaN in logits"
