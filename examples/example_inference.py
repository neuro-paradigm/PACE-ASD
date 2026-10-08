"""
PACE-ASD — worked example: score one stored recording with the released
ensemble and print what each output means.

    python examples/example_inference.py [clip_id]

Writes the same files as scripts/infer.py to examples/expected_output/<clip_id>/.
The released models were trained on every child of the demonstration cohort,
so the probability printed here is not an estimate of performance; it shows
what the outputs look like.
"""

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

from inference_api import PACEASDPredictor   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip_id", nargs="?", default="asd_1")
    a = ap.parse_args()
    npy = f"processed/features/{a.clip_id}.npy"
    pred = PACEASDPredictor("models/release", device="cpu")
    r = pred.predict_npy(npy)

    print(f"recording {a.clip_id}: {r['n_detected_frames']} detected frames, "
          f"first detection at frame {r['onset_frame']}, "
          f"{r['n_frames_rejected_implausible']} frames rejected by the plausibility check")
    print(f"probability {r['probability']:.3f} (mean of {r['n_models']} models; "
          f"standard deviation {r['probability_sd']:.3f})")
    sel = r["selection"]
    print(f"{sel['n_valid_blocks']} valid blocks of {sel['block_size']} frames; budget {sel['budget_blocks']}"
          + (" -> every valid block is kept, nothing is selected" if sel["selection_is_trivial"] else ""))
    for blk in sel["blocks"]:
        if blk["valid_frames"]:
            bar = "#" * round(10 * blk["selected_by_fraction_of_models"])
            print(f"  frames {blk['input_frames'][0]:3d}-{blk['input_frames'][1]:3d}  "
                  f"selected by {blk['selected_by_fraction_of_models']:.2f} of models  {bar}")
    print("attribution by stream:", r["attribution"]["stream_share"])
    print("note:", r["attribution"]["note"])

    out = os.path.join("examples", "expected_output", a.clip_id)
    subprocess.run([sys.executable, "scripts/infer.py", "--input_npy", npy, "--checkpoint",
                    "models/release", "--device", "cpu", "--output", out], check=True)
    print(f"files written to {out}")


if __name__ == "__main__":
    main()
