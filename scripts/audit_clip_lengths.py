"""
scripts/audit_clip_lengths.py
Audits the distribution of valid (non-padded) frame lengths across all processed
clips in processed/features/*.npy, and reports what fraction fall at or below
the ESG token budget used in A1 (M=8 blocks x L=15 frames = 120) and A3 (K=120).

Usage:
    python scripts/audit_clip_lengths.py --features_dir processed/features
    python scripts/audit_clip_lengths.py --features_dir processed/features --budget 120
"""

import argparse
import os
import sys

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features_dir", default="processed/features",
                   help="Directory with *.npy clip feature files (shape T x J x C)")
    p.add_argument("--budget", type=int, default=120,
                   help="ESG token budget to audit against (default: 120 = M*L for A1/A3)")
    p.add_argument("--threshold", type=float, default=1e-4,
                   help="Threshold for non-zero frame detection (default: 1e-4)")
    return p.parse_args()


def valid_length(arr, threshold=1e-4):
    """Number of frames where at least one joint has a non-zero coordinate."""
    # arr: (T, J, C) or (T, D) where D=J*C
    if arr.ndim == 3:
        per_frame = np.abs(arr).sum(axis=(-2, -1))
    else:
        per_frame = np.abs(arr).sum(axis=-1)
    valid = per_frame > threshold
    # Clip length = position of the last valid frame + 1
    nonzero_idx = np.where(valid)[0]
    if len(nonzero_idx) == 0:
        return 0
    return int(nonzero_idx[-1]) + 1


def main():
    args = parse_args()
    feat_dir = args.features_dir
    budget   = args.budget

    if not os.path.isdir(feat_dir):
        sys.exit(f"[ERROR] features_dir not found: {feat_dir}")

    npy_files = sorted([f for f in os.listdir(feat_dir) if f.endswith(".npy")])
    if not npy_files:
        sys.exit(f"[ERROR] No .npy files found in {feat_dir}")

    lengths = []
    errors  = []
    for fname in npy_files:
        path = os.path.join(feat_dir, fname)
        try:
            arr = np.load(path, allow_pickle=False)
            lengths.append(valid_length(arr, threshold=args.threshold))
        except Exception as e:
            errors.append((fname, str(e)))

    if errors:
        print(f"\n[WARN] Failed to load {len(errors)} files:")
        for fname, err in errors[:5]:
            print(f"  {fname}: {err}")

    lengths = np.array(lengths)
    n = len(lengths)

    pct_at_or_below = 100.0 * (lengths <= budget).sum() / n
    pct_below       = 100.0 * (lengths < budget).sum() / n
    pct_above       = 100.0 * (lengths > budget).sum() / n

    print(f"\n{'='*60}")
    print(f"  Clip-length audit   (n={n} clips, budget={budget} frames)")
    print(f"{'='*60}")
    print(f"  Min valid length        : {lengths.min()}")
    print(f"  Max valid length        : {lengths.max()}")
    print(f"  Mean ± SD               : {lengths.mean():.1f} ± {lengths.std():.1f}")
    print(f"  Median                  : {np.median(lengths):.0f}")
    print()
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for pct in percentiles:
        print(f"  P{pct:<4}                   : {np.percentile(lengths, pct):.0f}")
    print()
    print(f"  Clips <= budget ({budget}f)  : {(lengths<=budget).sum():4d} / {n}  ({pct_at_or_below:.1f}%)")
    print(f"  Clips <  budget ({budget}f)  : {(lengths< budget).sum():4d} / {n}  ({pct_below:.1f}%)")
    print(f"  Clips >  budget ({budget}f)  : {(lengths> budget).sum():4d} / {n}  ({pct_above:.1f}%)")
    print()

    if pct_at_or_below > 50:
        print(f"  [!] Over half the cohort fits within the ESG token budget.")
        print(f"      The gate may retain nearly ALL frames for most subjects,")
        print(f"      making 'sparse event selection' an overstatement for the majority.")
    elif pct_at_or_below > 25:
        print(f"  [~] A substantial minority ({pct_at_or_below:.0f}%) fits within the budget.")
        print(f"      Sparsity claim is valid for most clips but hedging is warranted.")
    else:
        print(f"  [OK] Most clips exceed the budget ({pct_above:.0f}% > {budget}f).")
        print(f"       ESG is genuinely selecting a sub-portion for the majority of the cohort.")

    # Histogram (ASCII)
    print(f"\n  ASCII histogram of valid lengths (bin width ~15 frames):")
    bins = np.arange(0, lengths.max() + 16, 15)
    counts, edges = np.histogram(lengths, bins=bins)
    max_bar = 40
    for i, (lo, hi, c) in enumerate(zip(edges[:-1], edges[1:], counts)):
        bar = "#" * int(round(c / counts.max() * max_bar)) if counts.max() > 0 else ""
        print(f"  {int(lo):4d}-{int(hi):4d}  {bar:<40}  {c}")

    print()


if __name__ == "__main__":
    main()
