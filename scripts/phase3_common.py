"""
PACE — helpers shared by the third-phase analyses (docs/validation_criteria_phase3.md).

  permuted_partition   label permutation i of the cohort with its own stratified
                       nested partition (D2)
  pseudo_partition     random pseudo-groups of the typically developing children
                       with their own nested partition (D1)
  plant_arm_swing      the arm-swing gain of the positive control, for a prepared
                       video sequence (D1)
  plant_arm_swing_depth  the same for a depth-camera skeleton (D1)
  run_network_folds    train the test-case network on a given partition and
                       arrays, one JSON per outer fold
"""

import json
import os
import sys

import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from recording_features import _runs           # noqa: E402
from sequence import valid_mask                 # noqa: E402

OUT = os.path.join(ROOT, "results", "phase3")
PERM_SEED = 20261201
PSEUDO_SEED = 20261101

# MediaPipe: shoulder, elbow, wrist and hand landmarks of each side
ARM_SIDES = {"left": (11, 13, [15, 17, 19, 21]), "right": (12, 14, [16, 18, 20, 22])}
# Kinect v2 joints (depth_skeleton.JOINTS) of each side
DEPTH_SIDES = {"Left": ("ShoulderLeft", "ElbowLeft", ["WristLeft", "HandLeft", "HandTipLeft", "ThumbLeft"]),
               "Right": ("ShoulderRight", "ElbowRight", ["WristRight", "HandRight", "HandTipRight", "ThumbRight"])}


def nested_partition(subjects, labels, seed, n_outer=5, n_inner=3):
    """One repetition of stratified n_outer-fold cross-validation with
    n_inner stratified inner folds, in the format of splits/cv_partition.json."""
    subj = np.asarray(subjects)
    y = np.asarray(labels)
    outer = StratifiedKFold(n_outer, shuffle=True, random_state=seed)
    folds = []
    for k, (tr, te) in enumerate(outer.split(subj, y)):
        inner = StratifiedKFold(n_inner, shuffle=True, random_state=seed + 1000 + k)
        inn = [{"train": sorted(subj[tr][a].tolist()), "val": sorted(subj[tr][b].tolist())}
               for a, b in inner.split(subj[tr], y[tr])]
        folds.append({"test": sorted(subj[te].tolist()), "inner": inn})
    return {"seed": int(seed), "n_outer": n_outer, "n_inner": n_inner,
            "subjects": subj.tolist(), "labels": {s: int(l) for s, l in zip(subj, y)},
            "repeats": [folds]}


def permuted_partition(i, subjects, labels):
    rng = np.random.default_rng(PERM_SEED + i)
    return nested_partition(subjects, rng.permutation(np.asarray(labels)), PERM_SEED + i)


def pseudo_labels(j, td_subjects):
    """22 children in pseudo-group 1, 23 in pseudo-group 0 (split j)."""
    td = sorted(td_subjects)
    perm = np.random.default_rng(PSEUDO_SEED + j).permutation(len(td))
    lab = np.zeros(len(td), int)
    lab[perm[:22]] = 1
    return td, lab


def pseudo_partition(j, td_subjects):
    td, lab = pseudo_labels(j, td_subjects)
    return nested_partition(td, lab, PSEUDO_SEED + j)


def _gain(rel, runs, gain, window):
    """Scale the deviation of the low-frequency part of `rel` (T, d) from its
    mean over `runs` by `gain`, keeping the high-frequency residual."""
    low = np.zeros_like(rel)
    for a, b in runs:
        low[a:b] = uniform_filter1d(rel[a:b], size=min(window, b - a), axis=0, mode="nearest")
    idx = np.concatenate([np.arange(a, b) for a, b in runs])
    m = low[idx].mean(0)
    out = rel.copy()
    out[idx] = m + gain * (low[idx] - m) + (rel[idx] - low[idx])
    return out


def plant_arm_swing(seq, gain, window=15):
    """Positive control, video: amplify the low-frequency movement of each
    arm's wrist and hand landmarks relative to the shoulder by `gain` (elbow
    by (1 + gain) / 2). Undetected frames stay zero."""
    if gain == 1:
        return seq
    out = seq.copy()
    runs = _runs(valid_mask(seq), 1)
    if not runs:
        return out
    for sh, el, hand in ARM_SIDES.values():
        for lm, g in [(el, (1 + gain) / 2)] + [(h, gain) for h in hand]:
            rel = seq[:, lm] - seq[:, sh]
            out[:, lm] = seq[:, sh] + _gain(rel, runs, g, window)
    out[~valid_mask(seq)] = 0.0
    return out


def plant_arm_swing_depth(X, gain, window=7):
    """Positive control, depth skeleton (T, 25, 3) in metres."""
    import depth_skeleton as ds
    if gain == 1:
        return X
    out = X.copy()
    runs = [(0, len(X))]
    for sh, el, hand in DEPTH_SIDES.values():
        s = ds.J[sh]
        for name, g in [(el, (1 + gain) / 2)] + [(h, gain) for h in hand]:
            j = ds.J[name]
            out[:, j] = X[:, s] + _gain(X[:, j] - X[:, s], runs, g, window)
    return out


def fast_auc(y, p):
    from scipy.stats import rankdata
    y = np.asarray(y); r = rankdata(p)
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logreg(X, y, lam=0.1, iters=50):
    """Logistic regression by Newton-Raphson with a weak ridge penalty `lam`
    on the standardized predictors (intercept unpenalized), so that the
    second-level models of D3 stay finite when the groups are nearly
    separable. Returns (coefficients incl. intercept, mean, sd, log-likelihood)."""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.zeros(Z.shape[1])
    P = np.eye(Z.shape[1]) * lam; P[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        g = Z.T @ (p - y) + P @ w
        H = (Z * (p * (1 - p))[:, None]).T @ Z + P
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    eta = Z @ w
    ll = float(np.sum(y * eta - np.logaddexp(0, eta)))
    return w, mu, sd, ll


def _predict(fit, X):
    w, mu, sd, _ = fit
    return np.c_[np.ones(len(X)), (X - mu) / sd] @ w


def _loo(X, y, ids):
    """Leave-one-child-out linear predictors (all copies of a child left out)."""
    out = np.zeros(len(y))
    for c in np.unique(ids):
        te = ids == c
        out[te] = _predict(logreg(X[~te], y[~te]), X[te])
    return out


def incremental_value(y, controls, movement, B=2000, seed=20261105):
    """D3: does `movement` (n,) add to the control scores `controls` (n, c)?
    Likelihood-ratio test of the nested second-level models, leave-one-out
    AUC of both and the stratified bootstrap interval of their difference
    (leave-one-out refitted within every resample)."""
    from scipy import stats
    y = np.asarray(y, float)
    Xa, Xb = np.asarray(controls, float), np.c_[controls, movement]
    fa, fb = logreg(Xa, y), logreg(Xb, y)
    lr = 2 * (fb[3] - fa[3])
    ids = np.arange(len(y))
    auc_a, auc_b = fast_auc(y, _loo(Xa, y, ids)), fast_auc(y, _loo(Xb, y, ids))
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    d = []
    for _ in range(B):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        yy = y[i]
        d.append(fast_auc(yy, _loo(Xb[i], yy, i)) - fast_auc(yy, _loo(Xa[i], yy, i)))
    ci = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    p = float(stats.chi2.sf(max(lr, 0.0), 1))
    return {"lr_stat": float(lr), "lr_p": p, "loo_auc_controls": auc_a, "loo_auc_with_movement": auc_b,
            "loo_auc_diff": auc_b - auc_a, "loo_auc_diff_ci95": ci,
            "adds": bool(p < 0.05 and ci[0] > 0)}


def paired_auc_bootstrap(y, a, b, B=2000, seed=20261106):
    """AUC of a and b, and the stratified bootstrap interval of a - b."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    da, db, dd = [], [], []
    for _ in range(B):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        x1, x2 = fast_auc(y[i], a[i]), fast_auc(y[i], b[i])
        da.append(x1); db.append(x2); dd.append(x1 - x2)
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"auc_a": fast_auc(y, a), "ci_a": ci(da), "auc_b": fast_auc(y, b), "ci_b": ci(db),
            "diff": fast_auc(y, a) - fast_auc(y, b), "diff_ci95": ci(dd)}


def run_network_folds(part, arrays, out_dir, jobs, arm="PACE", device=None, config_path=None):
    """Train the test-case network for the outer folds in `jobs` (list of k)
    of a one-repetition partition; one JSON per fold in out_dir."""
    import torch
    import yaml
    from protocol import arm_config
    from run_cv import run_outer_fold
    cfg = arm_config(arm, yaml.safe_load(open(config_path or os.path.join(ROOT, "configs", "config.yaml"))))
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)
    for k in jobs:
        path = os.path.join(out_dir, f"r0_k{k}.json")
        if os.path.isfile(path):
            continue
        rec = run_outer_fold(arm, cfg, part, arrays, 0, k, device, list(range(part["n_inner"])), None)
        json.dump(rec, open(path, "w"))
        from sklearn.metrics import roc_auc_score
        print(f"[{out_dir}] k{k} auc={roc_auc_score(rec['test_labels'], rec['probs']):.3f}", flush=True)
