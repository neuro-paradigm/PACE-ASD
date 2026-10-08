"""
scripts/wilcoxon_test.py
Paired Wilcoxon signed-rank tests across all per-seed JSON files in results/.

Compares every PACE arm (A1-A4) against every A5 baseline on AUC, accuracy, F1,
sensitivity, specificity, and ECE (test split), with Bonferroni correction over
the number of arm x baseline pairs x metrics tested.

Usage:
    python scripts/wilcoxon_test.py --results_dir results
    python scripts/wilcoxon_test.py --results_dir results --arm A1 --baseline A5_mtcformer
    python scripts/wilcoxon_test.py --results_dir results --split val
"""

import argparse
import json
import os
import sys
from itertools import product

import numpy as np
from scipy.stats import wilcoxon

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Paired Wilcoxon tests on per-seed results")
    p.add_argument("--results_dir", default="results",
                   help="Directory containing *_per_seed.json files")
    p.add_argument("--split",   default="test", choices=["val", "test"],
                   help="Which split to compare (default: test)")
    p.add_argument("--arm",      default=None,
                   help="If set, compare only this PACE arm (e.g. A1)")
    p.add_argument("--baseline", default=None,
                   help="If set, compare only this baseline (e.g. A5_mtcformer)")
    p.add_argument("--alpha",    type=float, default=0.05,
                   help="Significance threshold before Bonferroni correction")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

METRICS = ["auc", "accuracy", "f1", "sensitivity", "specificity", "ece"]

def load_per_seed(results_dir, model_id, split):
    """Return {metric: [seed0_val, seed1_val, ...]} for one arm/baseline."""
    path = os.path.join(results_dir, f"{model_id}_per_seed.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)[split]  # list of dicts, one per seed
    out = {}
    for m in METRICS:
        vals = [d[m] for d in data if m in d and not (isinstance(d[m], float) and d[m] != d[m])]
        if vals:
            out[m] = vals
    return out


def discover_ids(results_dir):
    """Auto-discover PACE arms (A1-A4) and A5 baselines from JSON files."""
    pace_arms, baselines = [], []
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith("_per_seed.json"):
            continue
        mid = fname.replace("_per_seed.json", "")
        if mid in ("A1", "A2", "A3", "A4"):
            pace_arms.append(mid)
        else:
            baselines.append(mid)
    return pace_arms, baselines


def cohens_d(a, b):
    diff = np.array(a) - np.array(b)
    sd   = diff.std(ddof=1)
    return diff.mean() / sd if sd > 0 else float("nan")


def ci_mean_diff(a, b, alpha=0.05, n_boot=2000, rng=None):
    """Bootstrap 95% CI on mean(a) - mean(b)."""
    rng   = rng or np.random.default_rng(42)
    diffs = np.array(a) - np.array(b)
    n     = len(diffs)
    boots = [rng.choice(diffs, size=n, replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return lo, hi


# ── Main ─────────────────────────────────────────────────────────────────────

def run_comparison(results_dir, arm_id, bl_id, split, alpha):
    arm_data = load_per_seed(results_dir, arm_id, split)
    bl_data  = load_per_seed(results_dir, bl_id,  split)

    if not arm_data:
        print(f"  [WARN] No per-seed data found for {arm_id} in {results_dir}/")
        return
    if not bl_data:
        print(f"  [WARN] No per-seed data found for {bl_id} in {results_dir}/")
        return

    n_tests = sum(
        1 for m in METRICS
        if m in arm_data and m in bl_data and
        len(arm_data[m]) == len(bl_data[m]) and len(arm_data[m]) >= 5
    )
    bonf = alpha / max(n_tests, 1)

    print(f"\n{'='*80}")
    print(f"  {arm_id} vs {bl_id}  |  split={split}  |  Bonferroni a={bonf:.4f}  (n_tests={n_tests})")
    print(f"{'='*80}")
    print(f"  {'Metric':<14}  {'PACE':>8}  {'Baseline':>10}  {'Diff':>8}  {'95% CI':>22}  {'p-val':>8}  {'d':>6}  Sig?  Direction")
    print(f"  {'-'*108}")

    rng = np.random.default_rng(42)
    for m in METRICS:
        if m not in arm_data or m not in bl_data:
            print(f"  {m:<14}  (missing data)")
            continue
        a_vals = arm_data[m]
        b_vals = bl_data[m]
        if len(a_vals) != len(b_vals):
            print(f"  {m:<14}  (seed-count mismatch: {len(a_vals)} vs {len(b_vals)})")
            continue
        if len(a_vals) < 5:
            print(f"  {m:<14}  (too few seeds: {len(a_vals)})")
            continue

        a_mean = float(np.mean(a_vals))
        b_mean = float(np.mean(b_vals))
        diff   = a_mean - b_mean
        lo, hi = ci_mean_diff(a_vals, b_vals, rng=rng)
        d      = cohens_d(a_vals, b_vals)

        try:
            _, p = wilcoxon(a_vals, b_vals, alternative="two-sided",
                            zero_method="wilcox", correction=True)
        except ValueError as e:
            print(f"  {m:<14}  Wilcoxon error: {e}")
            continue

        sig_marker = "**" if p < bonf else ("*" if p < alpha else "  ")
        if m == "ece":
            direction = "PACE better" if diff < 0 else ("Baseline better" if diff > 0 else "tie")
        else:
            direction = "PACE better" if diff > 0 else ("Baseline better" if diff < 0 else "tie")

        print(
            f"  {m:<14}  {a_mean:>8.4f}  {b_mean:>10.4f}  {diff:>+8.4f}"
            f"  [{lo:+.4f}, {hi:+.4f}]  {p:>8.4f}  {d:>+6.3f}  {sig_marker}    {direction}"
        )

    print()
    print(f"  ** = Bonferroni-corrected significant (a={alpha})")
    print(f"  *  = Nominally significant at a={alpha} (not Bonferroni-corrected)")


def main():
    args = parse_args()
    results_dir = args.results_dir

    if not os.path.isdir(results_dir):
        sys.exit(f"[ERROR] results_dir not found: {results_dir}")

    pace_arms, baselines = discover_ids(results_dir)

    if args.arm:
        pace_arms = [a for a in pace_arms if a == args.arm]
        if not pace_arms:
            sys.exit(f"[ERROR] arm '{args.arm}' not found in {results_dir}/")
    if args.baseline:
        baselines = [b for b in baselines if b == args.baseline]
        if not baselines:
            sys.exit(f"[ERROR] baseline '{args.baseline}' not found in {results_dir}/")

    if not pace_arms:
        sys.exit(f"[ERROR] No A1-A4 per-seed JSON files found in {results_dir}/")
    if not baselines:
        sys.exit(f"[ERROR] No baseline per-seed JSON files found in {results_dir}/")

    print(f"\nPACE arms : {pace_arms}")
    print(f"Baselines : {baselines}")
    print(f"Split     : {args.split}")

    for arm_id, bl_id in product(pace_arms, baselines):
        run_comparison(results_dir, arm_id, bl_id, args.split, args.alpha)


if __name__ == "__main__":
    main()
