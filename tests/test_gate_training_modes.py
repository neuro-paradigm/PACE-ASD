"""Gate training modes (model.gate_train_mode) and the motion-energy heuristic
gate (model.gate_scores = "motion").

Every training mode must leave evaluation unchanged (deterministic hard
top-M), the perturbed relaxation must pass gradients to blocks that the hard
gate does not select, the annealed budget must follow its schedule, and the
heuristic gate must select the blocks with the most movement.
"""

import copy
import sys, os

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def _cfg(base, **patch):
    cfg = copy.deepcopy(base)
    cfg["model"].update({"mask_padding": True, "align_onset": True, **patch})
    return cfg


def _clip(seed=0, n_valid=150, total=300):
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(total, 33, 2)
    x[:n_valid] = torch.randn(n_valid, 33, 2, generator=g) * 0.5
    return x


@pytest.mark.parametrize("mode", ["anneal", "gumbel", "perturbed"])
def test_training_modes_do_not_change_evaluation(minimal_config, mode):
    from model import ASDMotionModel
    torch.manual_seed(0)
    hard = ASDMotionModel(_cfg(minimal_config)).eval()
    torch.manual_seed(0)
    other = ASDMotionModel(_cfg(minimal_config, gate_train_mode=mode)).eval()
    other.set_epoch(1)
    x = torch.stack([_clip(0), _clip(1, n_valid=90)])
    with torch.no_grad():
        assert torch.allclose(hard(x)[1], other(x)[1], atol=1e-6)


def test_perturbed_mode_passes_gradients_to_unselected_blocks():
    from model import EventSaliencyGate
    torch.manual_seed(0)
    L, M, D = 15, 1, 16
    x = torch.randn(1, 6 * L, D, requires_grad=True)
    for mode, expect_grad in (("hard", False), ("perturbed", True)):
        gate = EventSaliencyGate(D, block_size=L, top_m=M, train_mode=mode).train()
        torch.manual_seed(1)
        tokens, _ = gate(x)
        x.grad = None
        tokens.sum().backward()
        scores = gate.get_block_scores()[0]
        chosen = int(scores.argmax())
        others = [b for b in range(6) if b != chosen]
        g = torch.stack([x.grad[0, b * L:(b + 1) * L].abs().sum() for b in others])
        assert bool((g > 0).any()) == expect_grad, mode


def test_anneal_schedule():
    from model import EventSaliencyGate
    gate = EventSaliencyGate(8, block_size=15, top_m=2, train_mode="anneal", anneal_epochs=20)
    gate.train()
    budgets = []
    for e in (1, 10, 19, 20, 50):
        gate.epoch = e
        budgets.append(gate.training_budget(20))
    assert budgets[0] == 20 and budgets[-1] == 2 and budgets[-2] == 2
    assert budgets[0] > budgets[1] > budgets[2] >= 2
    assert gate.training_budget(12) <= 12           # never more than the blocks present
    gate.eval()
    gate.epoch = 1
    assert gate.training_budget(20) == 2             # evaluation uses the fixed budget


def test_motion_gate_selects_moving_block(minimal_config):
    from model import ASDMotionModel
    torch.manual_seed(0)
    m = ASDMotionModel(_cfg(minimal_config, gate_scores="motion", event_top_m=1)).eval()
    base = torch.randn(33, 2) * 0.5
    x = base.expand(150, 33, 2).clone()
    t = torch.arange(15, dtype=torch.float32)
    x[45:60, 15:23, 1] += 0.5 * torch.sin(2 * torch.pi * t / 5)[:, None]   # block 3 moves
    x = torch.cat([x, torch.zeros(150, 33, 2)])[None]
    _, idx, scores = m.get_attention_maps(x)
    assert set((idx[0] // 15).tolist()) == {3}
    assert int(scores[0, :10].argmax()) == 3
