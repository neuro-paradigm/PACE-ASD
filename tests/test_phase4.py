"""Tests for the fourth-phase additions: Firth-penalized second level,
nested (cross-fitted) incremental value, the spline encoding of the time of
day and the inner-fold predictions."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))


def test_firth_is_finite_under_complete_separation():
    from phase4_common import firth_fit, firth_predict
    x = np.r_[np.linspace(-3, -0.5, 10), np.linspace(0.5, 3, 10)][:, None]
    y = np.r_[np.zeros(10), np.ones(10)]
    w, mu, sd = firth_fit(x, y)
    assert np.all(np.isfinite(w)) and abs(w[1]) < 20
    eta = firth_predict((w, mu, sd), x)
    assert np.all((eta > 0) == (y == 1))


def test_firth_maximizes_the_penalized_likelihood():
    from phase4_common import _pen_loglik, firth_fit
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    y = (X[:, 0] + rng.normal(size=40) > 0).astype(float)
    w, mu, sd = firth_fit(X, y)
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    eps = 1e-5
    grad = [(_pen_loglik(Z, y, w + eps * e) - _pen_loglik(Z, y, w - eps * e)) / (2 * eps) for e in np.eye(len(w))]
    assert np.max(np.abs(grad)) < 1e-4


def test_firth_approaches_maximum_likelihood_in_large_samples():
    from phase4_common import firth_fit
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(1)
    X = rng.normal(size=(4000, 2))
    y = (rng.uniform(size=4000) < 1 / (1 + np.exp(-(0.5 + X @ [1.0, -0.7])))).astype(float)
    w, mu, sd = firth_fit(X, y)
    ml = LogisticRegression(penalty=None, max_iter=1000).fit((X - mu) / sd, y)
    assert np.allclose(w[1:], ml.coef_[0], atol=0.02) and abs(w[0] - ml.intercept_[0]) < 0.02


def _toy_partition(n=40, seed=3):
    from phase3_common import nested_partition
    subj = [f"c{i:02d}" for i in range(n)]
    lab = [i % 2 for i in range(n)]
    return nested_partition(subj, lab, seed)


def test_nested_incremental_detects_an_added_signal_and_not_noise():
    from phase4_common import nested_incremental
    part = _toy_partition(80)
    rng = np.random.default_rng(4)
    S, y = part["subjects"], np.array([part["labels"][s] for s in part["subjects"]])
    ctl = {s: 1 / (1 + np.exp(-(0.5 * (2 * yy - 1) + rng.normal()))) for s, yy in zip(S, y)}
    mov = {s: 1 / (1 + np.exp(-(2.5 * (2 * yy - 1) + rng.normal()))) for s, yy in zip(S, y)}
    noise = {s: 1 / (1 + np.exp(-rng.normal())) for s in S}

    def by_fold(d):
        return {0: {k: {s: d[s] for s in S if s not in f["test"]} for k, f in enumerate(part["repeats"][0])}}
    outer = {"c": {0: ctl}, "m": {0: mov}, "z": {0: noise}}
    inner = {"c": by_fold(ctl), "m": by_fold(mov), "z": by_fold(noise)}
    assert nested_incremental(part, outer, inner, ["c"], "m", B=300)["adds"]
    assert not nested_incremental(part, outer, inner, ["c"], "z", B=300)["adds"]


def test_nested_incremental_never_uses_a_test_childs_own_score_in_training():
    """Level 2 of fold k is fitted on inner scores only: corrupting the
    outer-test scores of fold k must leave the other folds' predictions as they are."""
    from phase4_common import nested_incremental
    part = _toy_partition(40)
    S = part["subjects"]
    rng = np.random.default_rng(5)
    a = {s: float(rng.uniform(0.05, 0.95)) for s in S}
    b = {s: float(rng.uniform(0.05, 0.95)) for s in S}
    inner = {m: {0: {k: {s: d[s] for s in S if s not in f["test"]} for k, f in enumerate(part["repeats"][0])}}
             for m, d in (("a", a), ("b", b))}
    r1 = nested_incremental(part, {"a": {0: a}, "b": {0: b}}, inner, ["a"], "b", B=50)
    b2 = dict(b)
    for s in part["repeats"][0][0]["test"]:
        b2[s] = 0.5
    r2 = nested_incremental(part, {"a": {0: a}, "b": {0: b2}}, inner, ["a"], "b", B=50)
    assert r1["auc_controls"] == r2["auc_controls"]


def test_spline_time_of_day_represents_a_group_recorded_earlier_and_later():
    from run_feature_models import models
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(6)
    t_mid = rng.uniform(16, 18, 60)                          # one group in the late afternoon
    t_out = np.r_[rng.uniform(8, 11, 30), rng.uniform(19, 23, 30)]   # the other earlier and later
    X = np.r_[t_mid, t_out][:, None]
    y = np.r_[np.zeros(60), np.ones(60)]
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    lin = models(0)["Time-of-day-linear"][0].set_params(clf__C=1.0)
    spl = models(0)["Time-of-day"][0].set_params(clf__C=1.0)
    auc_lin = roc_auc_score(y, cross_val_predict(lin, X, y, cv=cv, method="predict_proba")[:, 1])
    auc_spl = roc_auc_score(y, cross_val_predict(spl, X, y, cv=cv, method="predict_proba")[:, 1])
    assert auc_lin < 0.7 and auc_spl > 0.95


def test_fit_predict_nested_matches_fit_predict_and_is_out_of_fold():
    from feature_sets import fit_predict
    from phase4_common import fit_predict_nested

    class F:
        def __init__(self, X):
            self.X = X

        def matrix(self, kind, subjects):
            return np.stack([self.X[s] for s in subjects])
    part = _toy_partition(40, seed=7)
    rng = np.random.default_rng(8)
    X = {s: rng.normal(size=2) + part["labels"][s] for s in part["subjects"]}
    import run_feature_models as rfm
    o1 = fit_predict("Demographics", F(X), part)
    o2, inner, params = fit_predict_nested("Demographics", F(X), part)
    assert all(abs(o1[s] - o2[s]) < 1e-12 for s in o1)
    for k, f in enumerate(part["repeats"][0]):
        assert set(inner[k]) == set(part["subjects"]) - set(f["test"])
    assert rfm is not None
