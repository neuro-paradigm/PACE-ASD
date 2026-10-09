"""
PACE — helpers of the fourth-phase analyses (docs/validation_criteria_phase4.md).

  firth_fit, firth_predict    Firth-penalized logistic regression (E2)
  fit_predict_nested          one nested cross-validation run of a feature
                              classifier that also returns, for every outer
                              fold, inner-fold predictions of its training
                              children (E2)
  inner_from_stored           the same inner-fold predictions for a stored
                              run of the frozen partition (best
                              hyperparameters read from the run files)
  inner_from_checkpoints      inner-fold predictions of the test-case network
                              from its stored inner-model checkpoints
  nested_incremental          incremental value of a movement score beyond
                              the control scores by cross-fitted (nested)
                              stacking with a Firth second level (E2)
"""

import glob
import json
import os
import sys

import numpy as np
from scipy.special import expit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

OUT4 = os.path.join(ROOT, "results", "phase4")
STACK_CONTROLS = ["Scene", "Time-of-day", "Frame-format", "Demographics"]


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fast_auc(y, s):
    from scipy.stats import rankdata
    y = np.asarray(y); r = rankdata(s)
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ── Firth-penalized logistic regression ───────────────────────────────────────

def _pen_loglik(Z, y, w):
    eta = Z @ w
    p = expit(eta)
    info = (Z * (p * (1 - p))[:, None]).T @ Z
    return float(np.sum(y * eta - np.logaddexp(0, eta)) + 0.5 * np.linalg.slogdet(info)[1])


def firth_fit(X, y, max_iter=200, tol=1e-9, max_step=5.0):
    """Logistic regression with Firth's penalty (Jeffreys prior; Firth, 1993):
    maximizes log L(b) + 0.5 log det I(b). The estimates stay finite when the
    groups are separable. Predictors are standardized on the fitting data
    (the penalty is invariant to that). Newton-Raphson on the modified score
    U*(b) = Z'(y - p + h (1/2 - p)), h the diagonal of the hat matrix, with
    step halving and a bounded step as in logistf. Returns (w, mu, sd)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.zeros(Z.shape[1])
    ll = _pen_loglik(Z, y, w)
    for _ in range(max_iter):
        p = expit(Z @ w)
        W = p * (1 - p)
        info = (Z * W[:, None]).T @ Z
        inv = np.linalg.inv(info)
        h = W * np.einsum("ij,jk,ik->i", Z, inv, Z)
        step = inv @ (Z.T @ (y - p + h * (0.5 - p)))
        big = np.abs(step).max()
        if big > max_step:
            step *= max_step / big
        t = 1.0
        while True:
            cand = w + t * step
            ll_c = _pen_loglik(Z, y, cand)
            if ll_c >= ll - 1e-12 or t < 1e-6:
                break
            t /= 2
        w, ll = cand, ll_c
        if np.abs(t * step).max() < tol:
            break
    return w, mu, sd


def firth_predict(fit, X):
    """Linear predictor (logit) of a Firth fit."""
    w, mu, sd = fit
    return np.c_[np.ones(len(X)), (np.asarray(X, float) - mu) / sd] @ w


# ── inner-fold predictions ────────────────────────────────────────────────────

def _inner_folds(fold, train):
    pos = {s: i for i, s in enumerate(train)}
    return [([pos[s] for s in f["train"]], [pos[s] for s in f["val"]]) for f in fold["inner"]]


def fit_predict_nested(name, F, part, r=0, subjects=None):
    """As feature_sets.fit_predict (identical outer predictions), plus, for
    every outer fold k, the inner-fold predictions of the outer-training
    children: the classifier with the hyperparameters selected for fold k is
    refitted on each inner training set and predicts that inner fold's
    validation children (each training child exactly once).
    Returns (outer {child: p}, inner {k: {child: p}}, params {k: {...}})."""
    from sklearn.base import clone
    from sklearn.model_selection import GridSearchCV
    from run_feature_models import model_seed, models
    S = subjects or part["subjects"]
    lab = part["labels"]
    outer, inner, params = {}, {}, {}
    for k in range(part["n_outer"]):
        fold = part["repeats"][r][k]
        test = fold["test"]
        train = sorted(set(S) - set(test))
        folds = _inner_folds(fold, train)
        est, grid, kind = models(model_seed(r, k, 0))[name]
        Xtr, Xte = F.matrix(kind, train), F.matrix(kind, test)
        ytr = np.array([lab[s] for s in train])
        gs = GridSearchCV(est, grid, scoring="roc_auc", cv=folds, refit=True).fit(Xtr, ytr)
        outer.update(zip(test, map(float, gs.predict_proba(Xte)[:, 1])))
        inn = {}
        for tr_i, va_i in folds:
            e = clone(gs.best_estimator_).fit(Xtr[tr_i], ytr[tr_i])
            inn.update(zip([train[i] for i in va_i], map(float, e.predict_proba(Xtr[va_i])[:, 1])))
        inner[k] = inn
        params[k] = {kk: (vv if isinstance(vv, (int, float, str)) or vv is None else str(vv))
                     for kk, vv in gs.best_params_.items()}
    return outer, inner, params


def inner_from_stored(name, F, part, R, runs_dir=None):
    """Inner-fold predictions {r: {k: {child: p}}} for the stored runs of
    classifier `name` under the frozen partition: the estimator of
    run_feature_models.models with the stored best hyperparameters of each
    outer fold is refitted on every inner training set."""
    from sklearn.base import clone
    from run_feature_models import model_seed, models
    runs_dir = runs_dir or os.path.join(ROOT, "results", "cv", "runs")
    lab = part["labels"]
    S = part["subjects"]
    out = {}
    for r in range(R):
        out[r] = {}
        for k in range(part["n_outer"]):
            rec = json.load(open(os.path.join(runs_dir, name, f"r{r}_k{k}.json")))
            fold = part["repeats"][r][k]
            train = sorted(set(S) - set(fold["test"]))
            folds = _inner_folds(fold, train)
            est, _, kind = models(model_seed(r, k, 0))[name]
            est = clone(est).set_params(**rec["best_params"])
            Xtr = F.matrix(kind, train)
            ytr = np.array([lab[s] for s in train])
            inn = {}
            for tr_i, va_i in folds:
                e = clone(est).fit(Xtr[tr_i], ytr[tr_i])
                inn.update(zip([train[i] for i in va_i], map(float, e.predict_proba(Xtr[va_i])[:, 1])))
            out[r][k] = inn
    return out


def inner_from_checkpoints(arm, part, R, device=None, ckpt_dir=None):
    """Inner-fold predictions of the test-case network: each outer-training
    child is scored by the inner model whose validation set contained it
    (checkpoints models/cv/<arm>/r<r>_k<k>_i<i>.pt), recalibrated with that
    model's logistic recalibration. These children also chose that model's
    stopping epoch and recalibration."""
    import torch
    import yaml
    from protocol import build_model, predict_logits
    from run_cv import load_arrays
    ckpt_dir = ckpt_dir or os.path.join(ROOT, "models", "cv", arm)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "config.yaml")))
    arrays = load_arrays(cfg, part["subjects"], cfg["data"].get("align_onset", True))
    out = {}
    for r in range(R):
        out[r] = {}
        for k in range(part["n_outer"]):
            inn = {}
            for i in range(part["n_inner"]):
                ck = torch.load(os.path.join(ckpt_dir, f"r{r}_k{k}_i{i}.pt"), map_location=device, weights_only=False)
                m = build_model(ck["arm"], ck["config"]).to(device).eval()
                m.load_state_dict(ck["state_dict"])
                val = ck["val"]
                assert val == part["repeats"][r][k]["inner"][i]["val"], (r, k, i)
                lg = predict_logits(m, [arrays[s] for s in val], device)
                cal = expit(lg * ck["scaler"]["temperature"] + ck["scaler"]["bias"])
                inn.update(zip(val, map(float, cal)))
            out[r][k] = inn
    return out


# ── incremental value by nested stacking ──────────────────────────────────────

def _metrics(y, eta):
    """AUC and mean log loss of the linear predictor eta."""
    return fast_auc(y, eta), float(np.mean(np.logaddexp(0, eta) - y * eta))


def nested_incremental(part, outer, inner, controls, movement, subset=None, R=None, B=2000, seed=20261405):
    """E2: does `movement` add to the `controls`?

    outer[m][r]   {child: out-of-fold probability} of level-1 model m
    inner[m][r][k] {child: inner-fold probability} for the training children
                  of outer fold k
    Level 2 is fitted, for every repetition r and outer fold k, on the
    outer-training children (their inner-fold scores) and applied to the
    outer-test children (their out-of-fold scores), so no child's label
    influences its own second-level prediction. Level 2 is a Firth logistic
    regression of the label on the logits of (a) the controls and (b) the
    controls and the movement score. With `subset`, level 2 is fitted and
    evaluated within the subset only.

    Reported: cross-fitted AUC and mean log loss of (a) and (b), averaged over
    repetitions; AUC gain (b - a) and log-loss reduction (a - b) with 95%
    stratified bootstrap intervals over children (paired; out-of-fold
    predictions fixed). Adds: both lower bounds above 0."""
    R = R or len(part["repeats"])
    S = [s for s in part["subjects"] if subset is None or s in subset]
    pos = {s: i for i, s in enumerate(S)}
    y = np.array([part["labels"][s] for s in S], float)
    eta = {"a": np.full((R, len(S)), np.nan), "b": np.full((R, len(S)), np.nan)}
    for r in range(R):
        for k, fold in enumerate(part["repeats"][r]):
            tset = set(fold["test"])
            test = [s for s in fold["test"] if s in pos]
            train = [s for s in S if s not in tset]
            if not test:
                continue
            ytr = np.array([part["labels"][s] for s in train], float)
            for key, names in (("a", list(controls)), ("b", list(controls) + [movement])):
                Xtr = np.array([[logit(inner[m][r][k][s]) for m in names] for s in train])
                Xte = np.array([[logit(outer[m][r][s]) for m in names] for s in test])
                eta[key][r, [pos[s] for s in test]] = firth_predict(firth_fit(Xtr, ytr), Xte)
    assert not np.isnan(eta["a"]).any() and not np.isnan(eta["b"]).any()

    def stats(idx):
        a = [_metrics(y[idx], eta["a"][r, idx]) for r in range(R)]
        b = [_metrics(y[idx], eta["b"][r, idx]) for r in range(R)]
        return (np.mean([v[0] for v in a]), np.mean([v[1] for v in a]),
                np.mean([v[0] for v in b]), np.mean([v[1] for v in b]))
    auc_a, ll_a, auc_b, ll_b = stats(np.arange(len(S)))
    rng = np.random.default_rng(seed)
    pp, nn = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    dauc, dll = [], []
    for _ in range(B):
        i = np.concatenate([rng.choice(pp, len(pp)), rng.choice(nn, len(nn))])
        a1, l1, a2, l2 = stats(i)
        dauc.append(a2 - a1); dll.append(l1 - l2)
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    c_auc, c_ll = ci(dauc), ci(dll)
    return {"n": len(S), "n_positive": int(y.sum()), "repetitions": R, "controls": list(controls),
            "movement": movement,
            "auc_controls": float(auc_a), "auc_with_movement": float(auc_b), "auc_gain": float(auc_b - auc_a),
            "auc_gain_ci95": c_auc,
            "logloss_controls": float(ll_a), "logloss_with_movement": float(ll_b),
            "logloss_reduction": float(ll_a - ll_b), "logloss_reduction_ci95": c_ll,
            "adds": bool(c_auc[0] > 0 and c_ll[0] > 0)}
