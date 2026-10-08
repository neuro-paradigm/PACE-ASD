"""
PACE-ASD — descriptor-level gradient x input attribution.

The 198-dimensional per-frame descriptor [position | velocity | acceleration]
enters the spatial encoder's first linear layer. Gradient x input is taken at
that layer's input, so the shares do not depend on the fixed scale factors of
the velocity (x10) and acceleration (x5) streams. Only frames with a detected
pose are counted; the sign of the target does not change |gradient x input|.

Attribution describes what a trained network is sensitive to for one input;
it is not a measurement of the movement and should be checked (planted events,
randomly initialised networks) before it is interpreted.
"""

import numpy as np
import torch

STREAMS = {"position": slice(0, 66), "velocity": slice(66, 132),
           "acceleration": slice(132, 198)}
REGIONS = {"head": list(range(0, 11)), "arms_hands": list(range(11, 23)),
           "hips": [23, 24], "legs_feet": list(range(25, 33))}


def descriptor_gradxinput(model, sequence: torch.Tensor) -> np.ndarray:
    """|gradient x input| of the logit with respect to the 198-d per-frame
    descriptor, shape (B, T, 198). Frame positions are those the encoder sees
    (after onset alignment when the model aligns)."""
    model.eval()
    layer = model.spatial_encoder.input_proj[0]
    store = {}

    def _pre(mod, args):
        f = args[0].detach().requires_grad_(True)
        store["f"] = f
        return (f,)

    handle = layer.register_forward_pre_hook(_pre)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            _, logits = model(sequence)
            logits.sum().backward()
        f = store["f"]
        gx = (f.grad * f).detach().abs()
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
    B, T = sequence.shape[:2]
    return gx.reshape(B, T, -1).cpu().numpy()


def shares(gx: np.ndarray, valid: np.ndarray) -> dict:
    """Stream shares, region shares over velocity and acceleration (which are
    unaffected by hip centring) and landmark shares for one sequence.
    gx: (T, 198); valid: (T,) bool."""
    g = gx[valid]
    total = g.sum() + 1e-12
    per_lm = g.reshape(-1, 3, 33, 2).sum(axis=(0, 3))           # (3, 33)
    motion = per_lm[1:].sum(0)
    return {
        "stream_share": {k: float(g[:, s].sum() / total) for k, s in STREAMS.items()},
        "region_share_velocity_acceleration": {
            k: float(motion[j].sum() / (motion.sum() + 1e-12)) for k, j in REGIONS.items()},
        "landmark_share": (per_lm.sum(0) / (per_lm.sum() + 1e-12)).tolist(),
    }
