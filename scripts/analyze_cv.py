"""
PACE-ASD — every number reported from the nested cross-validation.

Reads results/cv/runs/<model>/r*_k*.json (written by run_cv.py and
run_feature_models.py) and writes results/cv/summary.json.

Estimand. For each repetition r, every child has exactly one out-of-fold
probability (for the neural networks, the mean of the three inner models'
recalibrated probabilities). A metric is computed over all children of the
repetition and averaged over repetitions. Its 95% interval comes from a
stratified bootstrap of children (the same resampled children in every
repetition), so it describes uncertainty due to the sampling of children but
not due to the sampling of training sets (Bates et al., 2024). Paired
differences between models use the same resamples; for the differences in
area under the curve we also report the corrected resampled t interval of
Nadeau and Bengio (2003) over the R x K outer folds.

Usage:
    python scripts/analyze_cv.py
"""

import glob
import json
import os
import sys

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

RUNS = "results/cv/runs"
OUT = "results/cv/summary.json"
B = 2000
RNG = 20261010


def fast_auc(y, p):
    """Area under the ROC curve via the Mann-Whitney statistic (ties averaged);
    identical to sklearn's roc_auc_score, several times faster."""
    from scipy.stats import rankdata
    y = np.asarray(y); r = rankdata(p)
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ── loading ───────────────────────────────────────────────────────────────────

def load_model_runs(name, R=None, variant="original"):
    """-> dict r -> {subject: prob}, list of per-fold records."""
    files = sorted(glob.glob(os.path.join(RUNS, name, "r*_k*.json")))
    recs = [json.load(open(p)) for p in files]
    by_r = {}
    for rec in recs:
        if R is not None and rec["repeat"] >= R:
            continue
        if variant == "original" or "inner" not in rec or not isinstance(rec["inner"], list) \
                or not rec["inner"] or "test" not in rec["inner"][0]:
            probs = rec["probs"]
        else:
            probs = np.mean([inf["test"][variant]["probs"] for inf in rec["inner"]], 0)
        d = by_r.setdefault(rec["repeat"], {})
        d.update(zip(rec["test_subject_ids"], map(float, probs)))
    return by_r, recs


def complete(by_r, n_subj):
    return {r: d for r, d in by_r.items() if len(d) == n_subj}


# ── metrics ───────────────────────────────────────────────────────────────────

def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibration_slope_intercept(y, p):
    """Logistic recalibration y ~ a + b * logit(p) (slope b) and
    calibration-in-the-large a' with b fixed at 1 (Van Calster et al.)."""
    z = _logit(p)

    def nll(w, off=None):
        a, b = (w[0], w[1]) if off is None else (w[0], 1.0)
        eta = a + b * z
        return np.sum(np.logaddexp(0, eta) - y * eta)
    slope = minimize(nll, [0.0, 1.0], method="BFGS").x[1]
    citl = minimize(lambda w: nll(w, off=True), [0.0], method="BFGS").x[0]
    return float(slope), float(citl)


def ece(y, p, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    tot = 0.0
    for i in range(n_bins):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < n_bins - 1 else (p <= 1))
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(tot)


def metrics(y, p, full=True):
    yhat = p >= 0.5
    out = {"auc": fast_auc(y, p),
           "sensitivity": float(yhat[y == 1].mean()),
           "specificity": float((~yhat)[y == 0].mean())}
    out["balanced_accuracy"] = (out["sensitivity"] + out["specificity"]) / 2
    if full:
        out["brier"] = float(np.mean((p - y) ** 2))
        out["ece10"] = ece(y, p, 10)
        out["ece5"] = ece(y, p, 5)
        out["cal_slope"], out["cal_intercept"] = calibration_slope_intercept(y, p)
    return out


def matrix(by_r, subjects):
    R = sorted(by_r)
    return np.array([[by_r[r][s] for s in subjects] for r in R])        # (R, N)


def repeat_mean(y, P, full=True):
    ms = [metrics(y, p, full) for p in P]
    return {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}, ms


def bootstrap(y, mats: dict, pairs=(), full=False, B=B, seed=RNG):
    """mats: name -> (R, N) probability matrix. Returns per-model CIs of
    repeat-averaged metrics and CIs of paired differences."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    keys = ["auc", "sensitivity", "specificity", "balanced_accuracy"] + \
           (["brier", "cal_slope", "cal_intercept"] if full else [])
    draws = {n: {k: [] for k in keys} for n in mats}
    for _ in range(B):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        cur = {}
        for n, P in mats.items():
            m, _ = repeat_mean(y[idx], P[:, idx], full)
            cur[n] = m
            for k in keys:
                draws[n][k].append(m[k])
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    out = {n: {k: ci(v) for k, v in d.items()} for n, d in draws.items()}
    diffs = {}
    for a, b in pairs:
        diffs[f"{a} - {b}"] = {k: ci(np.array(draws[a][k]) - np.array(draws[b][k]))
                               for k in ("auc", "balanced_accuracy")}
    return out, diffs


def nadeau_bengio(recs_a, recs_b, n_train, n_test):
    """Corrected resampled t interval for the mean difference in per-fold AUC."""
    fa = {(r["repeat"], r["outer_fold"]): roc_auc_score(r["test_labels"], r["probs"]) for r in recs_a}
    fb = {(r["repeat"], r["outer_fold"]): roc_auc_score(r["test_labels"], r["probs"]) for r in recs_b}
    keys = sorted(set(fa) & set(fb))
    d = np.array([fa[k] - fb[k] for k in keys])
    J = len(d)
    var = (1 / J + n_test / n_train) * d.var(ddof=1)
    t = stats.t.ppf(0.975, J - 1)
    return {"mean": float(d.mean()), "ci95": [float(d.mean() - t * np.sqrt(var)),
                                             float(d.mean() + t * np.sqrt(var))],
            "n_folds": J}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    part = json.load(open("splits/cv_partition.json"))
    S = part["subjects"]
    y = np.array([part["labels"][s] for s in S])
    N = len(S)
    names = sorted(n for n in os.listdir(RUNS) if n != "pilot" and os.path.isdir(os.path.join(RUNS, n)))
    summary = {"n_children": N, "n_autistic": int(y.sum()), "n_td": int((1 - y).sum())}
    mats, recs, perf = {}, {}, {}
    R_common = None
    for n in names:
        by_r, rr = load_model_runs(n)
        by_r = complete(by_r, N)
        if not by_r:
            continue
        mats[n], recs[n] = matrix(by_r, S), rr
        R_common = len(by_r) if R_common is None else min(R_common, len(by_r))
    for n in mats:
        mats[n] = mats[n][:R_common]
        recs[n] = [r for r in recs[n] if r["repeat"] < R_common]
        mean, per = repeat_mean(y, mats[n])
        perf[n] = {"mean": mean,
                   "sd": {k: float(np.std([m[k] for m in per], ddof=1)) if len(per) > 1 else 0.0
                          for k in mean},
                   "per_repeat_auc": [m["auc"] for m in per]}
    summary["n_repeats"] = R_common
    summary["performance"] = perf

    ref = "PACE"
    pairs = [(ref, n) for n in mats if n != ref] + \
            [(n, "Duration") for n in mats if n not in (ref, "Duration")]
    cis, diffs = bootstrap(y, mats, pairs)
    for n in mats:
        perf[n]["ci95"] = cis[n]
    summary["paired_differences_bootstrap"] = diffs
    n_test = N / part["n_outer"]
    summary["paired_differences_nadeau_bengio"] = {
        f"{a} - {b}": nadeau_bengio(recs[a], recs[b], N - n_test, n_test) for a, b in pairs}
    # calibration intervals for every model, from the same resamples of children
    cal, _ = bootstrap(y, mats, full=True)
    for n in mats:
        perf[n]["ci95_calibration"] = cal[n]
    if ref in mats:
        # pooled reliability data for the figure
        summary["reliability"] = {"probs": mats[ref].mean(0).tolist(), "labels": y.tolist(),
                                  "subjects": S}

    json.dump(summary, open(OUT, "w"), indent=1)
    print(f"wrote {OUT} ({len(mats)} models, {R_common} repetitions)")
    for n in sorted(perf, key=lambda k: -perf[k]["mean"]["auc"]):
        m, c = perf[n]["mean"], perf[n]["ci95"]["auc"]
        print(f"{n:16s} AUC {m['auc']:.3f} [{c[0]:.3f}, {c[1]:.3f}]  sens {m['sensitivity']:.2f} "
              f"spec {m['specificity']:.2f}  brier {m['brier']:.3f}")


if __name__ == "__main__":
    main()
