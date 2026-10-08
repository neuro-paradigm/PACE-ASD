"""
PACE-ASD — end-to-end timing from video to probability on the processor.

Scores N videos of the cohort with the released ensemble through
PACEASDPredictor.predict_video (MediaPipe Pose, normalisation, preparation,
ensemble, block agreement and attribution), one video at a time in a single
process, and records the time spent on pose extraction and on everything
after it. Writes results/pose_timing.json.

Usage:
    python scripts/measure_pose_timing.py --raw_dir data/raw/Dataset --n 10 --device cpu
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

from inference_api import PACEASDPredictor          # noqa: E402
from preprocess import build_video_catalogue        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw/Dataset")
    ap.add_argument("--checkpoints", default="models/release")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count() // 2)
    cat = [e for e in build_video_catalogue(a.raw_dir) if e["group"] == "regular"]
    asd = [e for e in cat if e["label"] == 1][: a.n // 2]
    td = [e for e in cat if e["label"] == 0][: a.n - a.n // 2]
    pred = PACEASDPredictor(a.checkpoints, device=a.device)
    pred.predict_npy(f"processed/features/{asd[0]['clip_id']}.npy")     # warm-up
    rows = []
    for e in asd + td:
        t0 = time.perf_counter()
        r = pred.predict_video(e["video_path"])
        total = time.perf_counter() - t0
        v = r["video"]
        rows.append({"clip_id": e["clip_id"], "frames": v["frames_read"], "width": v["width"],
                     "height": v["height"], "pose_seconds": v["pose_seconds"],
                     "model_seconds": r["model_seconds"], "total_seconds": round(total, 3),
                     "probability": r["probability"]})
        print(rows[-1], flush=True)
    ms = [1000 * x["pose_seconds"] / x["frames"] for x in rows]
    out = {"n_videos": len(rows), "device": a.device, "n_models": r["n_models"],
           "threads": torch.get_num_threads(),
           "ms_per_frame_median": float(np.median(ms)),
           "seconds_per_video_median": float(np.median([x["pose_seconds"] for x in rows])),
           "frames_per_video_median": float(np.median([x["frames"] for x in rows])),
           "model_seconds_median": float(np.median([x["model_seconds"] for x in rows])),
           "end_to_end_seconds_median": float(np.median([x["total_seconds"] for x in rows])),
           "rows": rows}
    json.dump(out, open("results/pose_timing.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
