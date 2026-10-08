"""
PACE-ASD — training and evaluation of one model under the nested protocol.

`fit_model` trains one network on the inner-training children of an outer
fold, selects the epoch on the inner-validation children, fits the logistic
recalibration on the same validation children and returns everything needed
to score the outer-test children. The recipe is identical for PACE-ASD, its
ablation arms and the comparison architectures:

  AdamW, cosine decay over at most `epochs`, batch size `batch_size`,
  gradient-norm clipping at 1, binary cross-entropy with label smoothing 0.04,
  mixup on 35% of batches (Beta(0.2, 0.2)), every training child drawn
  `clips_per_subject` times per epoch with independent augmentation, early
  stopping after `early_stopping_patience` epochs without improvement.

Epoch selection uses either the composite validation score of the original
protocol (`composite`) or the validation loss alone (`val_loss`).
"""

import copy
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW

from calibration import PlattScaler
from comparison_models import COMPARISON_MODELS
from dataset import augment_sequence
from metrics import compute_ece
from model import ASDMotionModel
from sequence import fill_internal_gaps, valid_mask

LABEL_SMOOTH = 0.04
MIXUP_P = 0.35
MIXUP_ALPHA = 0.2


# ── arms ──────────────────────────────────────────────────────────────────────

def _pace(gate=True, transformer=True, **patch):
    return {"family": "pace", "kwargs": {"use_gate": gate, "use_transformer": transformer},
            "patch": patch}


ARMS = {
    # PACE-ASD and its ablations
    "PACE":            _pace(),
    "PACE-nogate":     _pace(gate=False),
    "PACE-frames":     _pace(**{"model.event_block_size": 1, "model.event_top_m": 120}),
    "PACE-noattn":     _pace(transformer=False),
    "PACE-M4":         _pace(**{"model.event_top_m": 4}),
    "PACE-nomask":     _pace(**{"model.mask_padding": False}),
    "PACE-noalign":    _pace(**{"model.align_onset": False, "data.align_onset": False}),
    "PACE-valloss":    _pace(**{"training.selection": "val_loss"}),
    "PACE-maskonly":   _pace(**{"data.input": "mask_only"}),
    "PACE-imagenorm":  _pace(**{"data.coordinates": "image"}),
    # comparison architectures
    **{name: {"family": "comparison", "kwargs": {}, "patch": {}} for name in COMPARISON_MODELS},
}


def apply_patch(config: dict, patch: dict) -> dict:
    cfg = copy.deepcopy(config)
    for key, val in patch.items():
        d = cfg
        *head, last = key.split(".")
        for p in head:
            d = d.setdefault(p, {})
        d[last] = val
    return cfg


def arm_config(arm: str, config: dict, overrides: dict | None = None) -> dict:
    cfg = apply_patch(config, ARMS[arm]["patch"])
    return apply_patch(cfg, overrides or {})


def build_model(arm: str, cfg: dict) -> nn.Module:
    spec = ARMS[arm]
    if spec["family"] == "pace":
        return ASDMotionModel(cfg, **spec["kwargs"])
    return COMPARISON_MODELS[arm](dropout=cfg["model"]["dropout"])


# ── reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── input handling ────────────────────────────────────────────────────────────

MASK_ONLY_VALUE = 0.5


def to_mask_only(seq: np.ndarray) -> np.ndarray:
    """Replace every detected frame by the same constant pose, so that only
    the pattern of detected frames remains (negative control)."""
    out = np.zeros_like(seq)
    out[valid_mask(seq)] = MASK_ONLY_VALUE
    return out


def crop_batch(x: torch.Tensor, multiple: int = 60) -> torch.Tensor:
    """Drop trailing frames that are undetected in every sequence of the
    batch. The models are invariant to trailing undetected frames, so this
    only saves computation."""
    valid = x.abs().sum(dim=(-2, -1)) > 1e-4
    if not valid.any():
        return x
    last = int(torch.nonzero(valid.any(dim=0)).max()) + 1
    T = min(x.shape[1], int(math.ceil(last / multiple) * multiple))
    return x[:, :T]


def _batches(arrays, labels, batch_size, shuffle):
    idx = np.arange(len(arrays))
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        j = idx[i:i + batch_size]
        yield np.stack([arrays[k] for k in j]), np.asarray(labels, np.float32)[j]


# ── training ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_logits(model, arrays, device, batch_size=32, mask_only=False) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(arrays), batch_size):
        xb = np.stack(arrays[i:i + batch_size])
        if mask_only:
            xb = np.stack([to_mask_only(s) for s in xb])
        x = crop_batch(torch.from_numpy(xb).to(device))
        out.append(model(x)[1].float().cpu().numpy())
    return np.concatenate(out)


def _selection_score(rule, auc, ece, vl, tl, history):
    if rule == "val_loss":
        return -vl
    prev = history[-2] if len(history) >= 2 else history[-1]
    svl = 0.6 * vl + 0.4 * prev["val_loss"]
    stl = 0.6 * tl + 0.4 * prev["train_loss"]
    return auc - 0.5 * svl - 0.4 * ece - 0.3 * max(0.0, svl - stl)


def fit_model(arm: str, cfg: dict, train_x, train_y, val_x, val_y, seed: int,
              device, verbose: bool = False):
    """Train one model. train_x / val_x are lists of prepared (T, 33, 2)
    arrays. Returns (model, scaler, info)."""
    tc = cfg["training"]
    mask_only = cfg.get("data", {}).get("input") == "mask_only"
    rule = tc.get("selection", "composite")
    set_seed(seed)
    model = build_model(arm, cfg).to(device)
    opt = AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc["epochs"], eta_min=1e-6)
    crit = nn.BCEWithLogitsLoss()
    reps = tc.get("clips_per_subject", 1)
    ep_x = [a for a in train_x for _ in range(reps)]
    ep_y = [y for y in train_y for _ in range(reps)]
    vy = np.asarray(val_y, int)

    best, best_state, best_epoch, wait, history = -np.inf, None, 0, 0, []
    t0 = time.time()
    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        tot, nb = 0.0, 0
        for xb, yb in _batches(ep_x, ep_y, tc["batch_size"], shuffle=True):
            xb = np.stack([augment_sequence(s) for s in xb])
            if mask_only:
                xb = np.stack([to_mask_only(s) for s in xb])
            x = crop_batch(torch.from_numpy(xb).to(device))
            y = torch.from_numpy(yb).to(device)
            if len(y) > 1 and np.random.rand() < MIXUP_P:
                lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
                perm = torch.randperm(len(y))
                x = lam * x + (1 - lam) * x[perm]
                y = lam * y + (1 - lam) * y[perm]
            _, logits = model(x)
            loss = crit(logits, y * (1 - LABEL_SMOOTH) + 0.5 * LABEL_SMOOTH)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        tl = tot / max(nb, 1)

        vlog = predict_logits(model, val_x, device, mask_only=mask_only)
        vl = float(crit(torch.from_numpy(vlog), torch.from_numpy(vy.astype(np.float32))))
        vp = 1 / (1 + np.exp(-vlog))
        auc = float(roc_auc_score(vy, vp)) if len(set(vy)) > 1 else 0.5
        ece = float(compute_ece(vy, vp))
        sens = float(((vp >= 0.5) & (vy == 1)).sum() / max((vy == 1).sum(), 1))
        spec = float(((vp < 0.5) & (vy == 0)).sum() / max((vy == 0).sum(), 1))
        history.append({"epoch": epoch, "train_loss": tl, "val_loss": vl, "val_auc": auc,
                        "val_ece": ece})
        score = _selection_score(rule, auc, ece, vl, tl, history)
        collapsed = rule == "composite" and (sens <= 0.05 or spec <= 0.05 or not np.isfinite(vl))
        if score > best and not collapsed:
            best, best_epoch, wait = score, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if verbose:
            print(f"  e{epoch:3d} tl={tl:.3f} vl={vl:.3f} auc={auc:.3f}{' *' if wait == 0 else ''}")
        if wait >= tc["early_stopping_patience"]:
            break
    if best_state is None:                       # every epoch collapsed: keep the last
        best_state, best_epoch = model.state_dict(), epoch
    model.load_state_dict(best_state)

    vlog = predict_logits(model, val_x, device, mask_only=mask_only)
    scaler = PlattScaler()
    scaler.fit(vlog, vy.astype(np.float32), lr=cfg["calibration"]["lr"],
               max_iter=cfg["calibration"]["max_iter"])
    vp = scaler.calibrate(vlog)
    info = {"best_epoch": best_epoch, "epochs_run": epoch,
            "seconds": round(time.time() - t0, 1),
            "val_auc": float(roc_auc_score(vy, vp)) if len(set(vy)) > 1 else None,
            "platt_temperature": float(scaler.temperature),
            "platt_bias": float(scaler.bias),
            "n_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad))}
    return model, scaler, info


# ── test-time input variants used by the shortcut analyses ────────────────────

def lead_shift(seq: np.ndarray, k: int) -> np.ndarray:
    """Prepend k undetected frames (content beyond the array end is cut)."""
    out = np.zeros_like(seq)
    out[k:] = seq[: len(seq) - k]
    return out


def crop_valid(seq: np.ndarray, n: int) -> np.ndarray:
    """Keep only the first n frames from the first detection onwards."""
    vm = valid_mask(seq)
    if not vm.any():
        return seq
    first = int(np.argmax(vm))
    out = np.zeros_like(seq)
    out[first: first + n] = seq[first: first + n]
    return out


TEST_VARIANTS = {
    "original": lambda s: s,
    "gaps_filled": fill_internal_gaps,
    "lead30": lambda s: lead_shift(s, 30),
    "crop90": lambda s: crop_valid(s, 90),
}
