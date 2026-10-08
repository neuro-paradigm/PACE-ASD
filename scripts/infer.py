"""
PACE-ASD — command-line inference on one recording.

    python scripts/infer.py --input walk.mp4 --checkpoint models/release --output outputs/walk
    python scripts/infer.py --input_npy processed/features/asd_1.npy --checkpoint models/release

--checkpoint accepts a single .pt file, a directory of checkpoints (scored as
an ensemble) or several files. Outputs written to --output:

    result.json            probability (ensemble mean), spread across members,
                           detection counts, onset frame, timings, checkpoints
    selected_events.json   blocks selected by each member, in input frame numbers,
                           with the fraction of members selecting each block
    attribution.json       descriptor-level gradient x input shares
    kinematics.npz         prepared positions, velocities, accelerations, validity
    visualizations/        kinematics.png, selection.png
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from inference_api import PACEASDPredictor          # noqa: E402
from sequence import valid_mask                     # noqa: E402


def kinematics(seq: np.ndarray):
    """Velocity and acceleration exactly as the network computes them."""
    vm = valid_mask(seq)
    vel = np.zeros_like(seq); acc = np.zeros_like(seq)
    vel[1:] = (seq[1:] - seq[:-1]) * 10.0
    vel[1:][~(vm[1:] & vm[:-1])] = 0
    acc[2:] = (vel[2:] - vel[1:-1]) * 5.0
    acc[2:][~(vm[2:] & vm[1:-1] & vm[:-2])] = 0
    return vel, acc, vm


def save_figures(out_dir, seq, vel, acc, vm, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    viz = os.path.join(out_dir, "visualizations")
    os.makedirs(viz, exist_ok=True)
    onset = result["onset_frame"]
    last = int(np.flatnonzero(vm).max()) + 1 if vm.any() else len(vm)
    t = np.arange(last) + onset
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for ax, arr, name in zip(axes, (seq, vel, acc), ("position", "velocity", "acceleration")):
        for j, lab in ((15, "left wrist"), (27, "left ankle")):
            y = np.where(vm[:last], arr[:last, j, 0], np.nan)
            ax.plot(t, y, lw=1.2, label=lab)
        ax.set_ylabel(f"x {name}")
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("frame (input numbering)")
    fig.tight_layout(); fig.savefig(os.path.join(viz, "kinematics.png"), dpi=150); plt.close(fig)

    sel = result.get("selection")
    if sel:
        b = [blk for blk in sel["blocks"]]
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.bar([blk["input_frames"][0] for blk in b],
               [blk["selected_by_fraction_of_models"] for blk in b],
               width=sel["block_size"] * 0.9, align="edge",
               color=["#4C72B0" if blk["valid_frames"] else "#BBBBBB" for blk in b])
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("frame (input numbering)")
        ax.set_ylabel("fraction of models\nselecting the block")
        ax.set_title(f"{sel['n_valid_blocks']} valid blocks, budget {sel['budget_blocks']}"
                     + (" (every valid block is kept)" if sel["selection_is_trivial"] else ""),
                     fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(viz, "selection.png"), dpi=150); plt.close(fig)


def run(a) -> dict:
    ck = a.checkpoint[0] if len(a.checkpoint) == 1 else a.checkpoint
    pred = PACEASDPredictor(ck, a.config, device=a.device)
    res = (pred.predict_video(a.input, attribution=not a.no_attribution) if a.input
           else pred.predict_npy(a.input_npy, attribution=not a.no_attribution))
    seq = res.pop("_prepared")
    os.makedirs(a.output, exist_ok=True)
    vel, acc, vm = kinematics(seq)
    np.savez_compressed(os.path.join(a.output, "kinematics.npz"), positions=seq,
                        velocities=vel, accelerations=acc, valid=vm,
                        input_frame=np.arange(len(seq)) + res["onset_frame"])
    if "selection" in res:
        json.dump(res["selection"], open(os.path.join(a.output, "selected_events.json"), "w"), indent=1)
    if "attribution" in res:
        json.dump(res["attribution"], open(os.path.join(a.output, "attribution.json"), "w"), indent=1)
    summary = {k: v for k, v in res.items() if k not in ("selection", "attribution")}
    json.dump(summary, open(os.path.join(a.output, "result.json"), "w"), indent=1)
    if not a.no_figures:
        save_figures(a.output, seq, vel, acc, vm, res)
    return res


def main():
    ap = argparse.ArgumentParser(description="PACE-ASD inference on one recording")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="video file (.mp4, .avi, ...)")
    src.add_argument("--input_npy", help="stored landmark array (300, 33, 2)")
    ap.add_argument("--checkpoint", nargs="+", required=True,
                    help=".pt file(s) or a directory of .pt files (ensemble)")
    ap.add_argument("--config", default=None, help="optional inference YAML")
    ap.add_argument("--output", default="outputs/result")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--no_attribution", action="store_true")
    ap.add_argument("--no_figures", action="store_true")
    res = run(ap.parse_args())
    print(f"{res['clip_id']}: probability {res['probability']:.3f} "
          f"(SD across {res['n_models']} models {res['probability_sd']:.3f})")


if __name__ == "__main__":
    main()
