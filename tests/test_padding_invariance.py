"""Frame-validity handling (model.align_onset = model.mask_padding = True).

The logit of a clip must not depend on how many undetected (all-zero) frames
precede its first detection or follow its last one, and Block-ESG must never
select a block without a valid frame while an unselected valid block exists.
"""

import copy
import sys, os

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

VARIANTS = [
    ("A1", {"use_gate": True,  "use_transformer": True},  {}),
    ("A2", {"use_gate": False, "use_transformer": True},  {}),
    ("A3", {"use_gate": True,  "use_transformer": True},  {"event_block_size": 1, "event_top_m": 60}),
    ("A4", {"use_gate": True,  "use_transformer": False}, {}),
]


def _clip(n_valid: int, total: int, seed: int = 0, lead: int = 0, gap=None) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(total, 33, 2)
    x[lead:lead + n_valid] = torch.randn(n_valid, 33, 2, generator=g) * 0.5
    if gap is not None:
        x[lead + gap[0]: lead + gap[1]] = 0.0
    return x


def _model(cfg, kwargs, patch):
    from model import ASDMotionModel
    cfg = copy.deepcopy(cfg)
    cfg["model"].update(patch)
    cfg["model"]["mask_padding"] = True
    cfg["model"]["align_onset"] = True
    torch.manual_seed(0)
    return ASDMotionModel(cfg, **kwargs).eval()


@pytest.mark.parametrize("variant,kwargs,patch", VARIANTS)
def test_prediction_invariant_to_trailing_padding(minimal_config, variant, kwargs, patch):
    m = _model(minimal_config, kwargs, patch)
    short = _clip(70, 120, gap=(20, 26))
    long = torch.cat([short, torch.zeros(180, 33, 2)])
    with torch.no_grad():
        _, l_short = m(short[None])
        _, l_long = m(long[None])
    assert torch.allclose(l_short, l_long, atol=1e-5), (
        f"{variant}: logit changed with trailing padding ({l_short.item()} vs {l_long.item()})")


@pytest.mark.parametrize("variant,kwargs,patch", VARIANTS)
def test_prediction_invariant_to_leading_undetected_frames(minimal_config, variant, kwargs, patch):
    m = _model(minimal_config, kwargs, patch)
    base = _clip(70, 300, gap=(20, 26))
    for lead in (1, 17, 45):
        shifted = _clip(70, 300, lead=lead, gap=(20, 26))
        with torch.no_grad():
            _, l0 = m(base[None])
            _, l1 = m(shifted[None])
        assert torch.allclose(l0, l1, atol=1e-5), (
            f"{variant}: logit changed with {lead} leading frames ({l0.item()} vs {l1.item()})")


def test_batch_with_different_onsets_matches_single(minimal_config):
    m = _model(minimal_config, {"use_gate": True, "use_transformer": True}, {})
    a, b = _clip(60, 300, seed=1, lead=0), _clip(80, 300, seed=2, lead=33)
    with torch.no_grad():
        _, batch = m(torch.stack([a, b]))
        _, la = m(a[None])
        _, lb = m(b[None])
    assert torch.allclose(batch, torch.cat([la, lb]), atol=1e-5)


def test_gate_prefers_valid_blocks_when_masked(minimal_config):
    """Blocks without valid frames are selected only after every valid block."""
    m = _model(minimal_config, {"use_gate": True, "use_transformer": True}, {})   # L=15, M=4
    x = _clip(50, 300, lead=40)        # 4 valid blocks after alignment
    with torch.no_grad():
        m(x[None])
    gate = m.saliency_gate
    assert int(gate.get_valid_blocks()[0].sum()) == 4
    # all 50 valid frames are retained; the remaining 10 tokens are masked
    assert int(gate.get_token_mask()[0].sum()) == 50


def test_flags_default_to_off(minimal_config):
    """Checkpoints whose stored configuration lacks the keys keep their behaviour."""
    from model import ASDMotionModel
    torch.manual_seed(0)
    m = ASDMotionModel(minimal_config).eval()
    assert m.mask_padding is False and m.align_onset is False
    with torch.no_grad():
        m(_clip(50, 120)[None])
    assert m.saliency_gate.get_token_mask() is None


def test_all_zero_sequence_is_finite(minimal_config):
    """A sequence with no detected frame must not produce NaN."""
    for kwargs in ({"use_gate": True, "use_transformer": True},
                   {"use_gate": False, "use_transformer": True},
                   {"use_gate": True, "use_transformer": False}):
        m = _model(minimal_config, kwargs, {})
        x = torch.stack([torch.zeros(120, 33, 2), _clip(40, 120)])
        with torch.no_grad():
            _, logits = m(x)
        assert torch.isfinite(logits).all(), kwargs
