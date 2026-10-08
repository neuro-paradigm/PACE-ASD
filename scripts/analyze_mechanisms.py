"""
PACE — analyses of what the models use and how the software behaves.

Writes results/cv/mechanisms.json with the sections below (each can be run
alone with --sections):

  cohort       recording properties by group (detected frames, frames rejected
               by the plausibility check, late onsets, internal gaps, clips
               truncated at 300 frames, valid blocks relative to the budget)
  shortcuts    out-of-fold AUC under test-time input changes (internal gaps
               filled, 30 undetected frames prepended, clips cut to their first
               90 frames), largest logit change from prepending frames, and
               within-group score-duration correlations
  confounds    AUC within boys, the shared age range and the cropped videos;
               format sensitivity within the autistic group
  selection    agreement of the selected blocks across the 15 models that
               scored each child, against random selection, by budget
  attribution  descriptor-level gradient x input by stream and region, stream
               occlusion, and the same attribution from randomly initialised
               networks (model-randomisation check)
  gait         group differences in each gait descriptor
  format       frame size by group, and whether gait descriptors recover the
               frame format within the autistic group when computed from
               image-normalised versus pixel coordinates
  runtime      latency of one forward pass and of the released ensemble;
               training time per model for every architecture

Second revision (docs/validation_criteria.md):
  selection_nulls  agreement of the selected blocks of randomly initialized
               networks, agreement with a motion-energy heuristic, and
               agreement between models of different repetitions (B3)
  movement     gait descriptors within the cropped videos, unadjusted and
               adjusted for age and sex, from standard, smoothed and
               first-90-frame trajectories; landmark jitter; scene
               properties; representativeness of the cropped autistic
               children (B4)
  depth        depth-camera skeletons: descriptors and recording properties by
               group, agreement with the video descriptors (B5)

Usage:
    python scripts/analyze_mechanisms.py
"""

import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)

from attribution import descriptor_gradxinput          # noqa: E402
from model import ASDMotionModel                             # noqa: E402
from protocol import ARMS, build_model, set_seed             # noqa: E402
from sequence import mask_summary, prepare_sequence, valid_mask  # noqa: E402

RUNS = "results/cv/runs"
CKPTS = "models/cv"
OUT = "results/cv/mechanisms.json"
B = 2000
RNG = 20261011


def fast_auc(y, p):
    """Area under the ROC curve via the Mann-Whitney statistic (ties averaged);
    identical to sklearn's roc_auc_score, several times faster."""
    from scipy.stats import rankdata
    y = np.asarray(y); r = rankdata(p)
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
STREAMS = {"position": slice(0, 66), "velocity": slice(66, 132), "acceleration": slice(132, 198)}
REGIONS = {"head": list(range(0, 11)), "arms_hands": list(range(11, 23)),
           "hips": [23, 24], "legs_feet": list(range(25, 33))}


def summ(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "median": float(np.median(v)), "q1": float(np.percentile(v, 25)),
            "q3": float(np.percentile(v, 75)), "min": float(v.min()), "max": float(v.max()),
            "n": int(len(v))}


def part_info():
    p = json.load(open("splits/cv_partition.json"))
    S = p["subjects"]
    return p, S, np.array([p["labels"][s] for s in S])


def raw(s):
    return np.load(f"processed/features/{s}.npy").astype(np.float32)


# ── cohort ────────────────────────────────────────────────────────────────────

def cohort():
    p, S, y = part_info()
    rows = []
    for s in S:
        a = raw(s)
        m = mask_summary(a)
        q = prepare_sequence(a, align=False)
        n_rej = int(valid_mask(a).sum() - valid_mask(q).sum())
        prep = prepare_sequence(a)
        vm = valid_mask(prep)
        vb = int(sum(vm[b * 15:(b + 1) * 15].any() for b in range(20)))
        rows.append({**m, "rejected": n_rej, "valid_blocks": vb,
                     "frac_rejected": n_rej / max(m["n_valid"], 1)})
    out = {}
    for g, lab in (("autistic", 1), ("td", 0), ("all", None)):
        R = [r for r, yy in zip(rows, y) if lab is None or yy == lab]
        out[g] = {
            "n": len(R),
            "detected_frames": summ([r["n_valid"] for r in R]),
            "late_onset_n": int(sum(r["first_valid"] > 0 for r in R)),
            "onset_frame_if_late": summ([r["first_valid"] for r in R if r["first_valid"] > 0]),
            "internal_gap_n": int(sum(r["gap_frames"] > 0 for r in R)),
            "truncated_at_300_n": int(sum(r["last_valid"] == 299 for r in R)),
            "rejected_frames_total": int(sum(r["rejected"] for r in R)),
            "clips_with_rejected_frames": int(sum(r["rejected"] > 0 for r in R)),
            "valid_blocks": summ([r["valid_blocks"] for r in R]),
            "valid_blocks_le_8": int(sum(r["valid_blocks"] <= 8 for r in R)),
            "valid_blocks_le_4": int(sum(r["valid_blocks"] <= 4 for r in R)),
        }
    out["all"]["detected_frames_total"] = int(sum(r["n_valid"] for r in rows))
    out["max_td_detected"] = int(max(r["n_valid"] for r, yy in zip(rows, y) if yy == 0))
    out["autistic_longer_than_all_td"] = int(sum(r["n_valid"] > out["max_td_detected"]
                                                 for r, yy in zip(rows, y) if yy == 1))
    out["late_onset_fisher_p"] = float(stats.fisher_exact(
        [[out["autistic"]["late_onset_n"], out["autistic"]["n"] - out["autistic"]["late_onset_n"]],
         [out["td"]["late_onset_n"], out["td"]["n"] - out["td"]["late_onset_n"]]])[1])
    out["auc_onset_frame_td_high"] = float(roc_auc_score(1 - y, [r["first_valid"] for r in rows]))
    out["auc_detected_frames"] = float(roc_auc_score(y, [r["n_valid"] for r in rows]))
    return out


# ── out-of-fold helpers ───────────────────────────────────────────────────────

def oof(name, variant="original", kind="probs"):
    """repeat -> {subject: value}; value is the mean over inner models."""
    by_r = defaultdict(dict)
    for p in glob.glob(os.path.join(RUNS, name, "r*_k*.json")):
        rec = json.load(open(p))
        if variant == "original" and kind == "probs":
            v = rec["probs"]
        else:
            v = np.mean([inf["test"][variant][kind] for inf in rec["inner"]], 0)
        by_r[rec["repeat"]].update(zip(rec["test_subject_ids"], map(float, v)))
    return dict(by_r)


def mat(by_r, S, R):
    return np.array([[by_r[r][s] for s in S] for r in range(R)])


def boot_auc(y, mats, idx_subset=None, pairs=(), B=B, seed=RNG):
    rng = np.random.default_rng(seed)
    sub = np.arange(len(y)) if idx_subset is None else np.asarray(idx_subset)
    ys = y[sub]
    pos, neg = sub[ys == 1], sub[ys == 0]
    est = {n: float(np.mean([fast_auc(y[sub], P[r, sub]) for r in range(len(P))])) for n, P in mats.items()}
    draws = {n: [] for n in mats}
    for _ in range(B):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        for n, P in mats.items():
            draws[n].append(np.mean([fast_auc(y[idx], P[r, idx]) for r in range(len(P))]))
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    out = {n: {"auc": est[n], "ci95": ci(draws[n])} for n in mats}
    for a, b in pairs:
        out[f"{a} - {b}"] = {"diff": est[a] - est[b],
                             "ci95": ci(np.array(draws[a]) - np.array(draws[b]))}
    return out


# ── shortcuts ─────────────────────────────────────────────────────────────────

def shortcuts(R):
    p, S, y = part_info()
    out = {}
    spans = np.array([mask_summary(prepare_sequence(raw(s)))["span"] for s in S])
    dur = np.array([valid_mask(raw(s)).sum() for s in S], float)
    long_idx = np.flatnonzero(spans >= 90)
    out["crop90_subset"] = {"n": int(len(long_idx)), "n_autistic": int(y[long_idx].sum()),
                            "duration_auc": float(roc_auc_score(y[long_idx], dur[long_idx]))}
    arms = [a for a in ARMS if a != "PACE-maskonly" and os.path.isdir(os.path.join(RUNS, a))]
    for arm in arms:
        try:
            mats = {v: mat(oof(arm, v), S, R) for v in ("original", "gaps_filled", "lead30", "crop90")}
        except (KeyError, IndexError):
            continue
        res = boot_auc(y, {k: mats[k] for k in ("original", "gaps_filled", "lead30")},
                       pairs=[("gaps_filled", "original"), ("lead30", "original")])
        res["crop90_subset"] = boot_auc(y, {"original": mats["original"], "crop90": mats["crop90"]},
                                        idx_subset=long_idx, pairs=[("crop90", "original")])
        lg0, lg1 = mat(oof(arm, "original", "logits"), S, R), mat(oof(arm, "lead30", "logits"), S, R)
        # prepending 30 frames to a 300-frame array drops its last 30 frames, so
        # the comparison is restricted to recordings that end before frame 270
        fits = spans <= 270
        res["n_lead30_untruncated"] = int(fits.sum())
        res["max_abs_logit_change_lead30"] = float(np.abs(lg0 - lg1)[:, fits].max())
        res["median_abs_logit_change_lead30"] = float(np.median(np.abs(lg0 - lg1)[:, fits]))
        out[arm] = res

    # within-group association of the network's probability with clip duration
    if os.path.isdir(os.path.join(RUNS, "PACE")):
        P = mat(oof("PACE"), S, R).mean(0)
        rng = np.random.default_rng(RNG)
        corr = {}
        for g, lab in (("autistic", 1), ("td", 0)):
            m = y == lab
            rho = stats.spearmanr(P[m], dur[m]).correlation
            bs = []
            for _ in range(B):
                i = rng.choice(np.flatnonzero(m), m.sum())
                bs.append(stats.spearmanr(P[i], dur[i]).correlation)
            corr[g] = {"rho": float(rho), "ci95": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))],
                       "n": int(m.sum())}
        out["within_group_spearman_probability_duration"] = corr
    return out


# ── demographic confounding ───────────────────────────────────────────────────

def confounds(R):
    """For every model: out-of-fold AUC within boys, within the age range shared
    by both groups, within boys of that range and within the cropped videos
    (the only frame format both groups share); the AUC with which the model's
    output separates uncropped from cropped videos within the autistic group;
    The Covariates control (run_feature_models.py) gives the counterpart for
    everything that is not movement."""
    import pandas as pd
    p, S, y = part_info()
    meta = pd.read_csv("processed/participant_metadata.csv").set_index("subject_id")
    age = np.array([meta.loc[s, "age_years"] for s in S], float)
    male = np.array([meta.loc[s, "sex"] == "M" for s in S])
    dur = np.array([valid_mask(raw(s)).sum() for s in S], float)
    lo = max(age[y == 1].min(), age[y == 0].min()); hi = min(age[y == 1].max(), age[y == 0].max())
    shared = (age >= lo) & (age <= hi)
    vid = pd.read_csv("processed/video_metadata.csv").set_index("clip_id")
    full = np.array([vid.loc[s, "width"] == 1080 and vid.loc[s, "height"] == 1920 for s in S])
    subsets = {"boys": male, "shared_age": shared, "boys_shared_age": male & shared,
               "cropped": ~full, "cropped_boys_shared_age": ~full & male & shared}
    out = {"shared_age_range": [float(lo), float(hi)],
           "subset_sizes": {k: {"autistic": int((m & (y == 1)).sum()), "td": int((m & (y == 0)).sum())}
                            for k, m in subsets.items()},
           "age_by_group": {g: summ(age[y == lab]) for g, lab in (("autistic", 1), ("td", 0))},
           "boys_by_group": {g: int(male[y == lab].sum()) for g, lab in (("autistic", 1), ("td", 0))},
           "models": {}}
    for name in sorted(os.listdir(RUNS)):
        if name == "pilot" or not os.path.isdir(os.path.join(RUNS, name)):
            continue
        try:
            P = mat(oof(name), S, R)
        except (KeyError, IndexError):
            continue
        res = {}
        for k, m in subsets.items():
            res[f"auc_{k}"] = boot_auc(y, {"m": P}, idx_subset=np.flatnonzero(m), B=1000)["m"]
        a = y == 1
        res["format_auc_within_autistic"] = float(np.mean([roc_auc_score(full[a], P[r, a])
                                                           for r in range(len(P))]))
        out["models"][name] = res
    return out


# ── checkpoints ───────────────────────────────────────────────────────────────

def load_ckpt(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = build_model(ck["arm"], ck["config"]).to(device).eval()
    m.load_state_dict(ck["state_dict"])
    return m, ck


def chance_jaccard(n: int, k: int) -> float:
    """Expected Jaccard index of two independent uniformly random k-subsets of
    n items. The overlap j is hypergeometric, P(j) = C(k,j) C(n-k,k-j) / C(n,k),
    and the union has 2k - j items, so E[J] = sum_j P(j) j / (2k - j)."""
    j = np.arange(max(0, 2 * k - n), k + 1)
    return float(np.sum(stats.hypergeom.pmf(j, n, k, k) * j / (2 * k - j)))


@torch.no_grad()
def selection(device):
    p, S, y = part_info()
    out = {}
    for arm in ("PACE", "PACE-M4", "PACE-frames"):
        files = sorted(glob.glob(os.path.join(CKPTS, arm, "*.pt")))
        if not files:
            continue
        sel = defaultdict(list)
        nvb = {}
        for f in files:
            m, ck = load_ckpt(f, device)
            L, M = m.saliency_gate.block_size, m.saliency_gate.top_m
            for s in ck["test"]:
                a = prepare_sequence(raw(s))
                vm = valid_mask(a)
                nb = int(np.ceil(len(vm) / L))
                vb = np.array([vm[b * L:(b + 1) * L].any() for b in range(nb)])
                nvb[s] = int(vb.sum())
                _, idx, _ = m.get_attention_maps(torch.from_numpy(a)[None].to(device))
                sel[s].append({b for b in (idx[0].cpu().numpy() // L).tolist() if b < nb and vb[b]})
        per_child, jac, chance = {}, [], []
        nontrivial = [s for s in sel if nvb[s] > M]
        for s in nontrivial:
            sets = sel[s]
            js = [len(sets[i] & sets[j]) / max(len(sets[i] | sets[j]), 1)
                  for i in range(len(sets)) for j in range(i + 1, len(sets))]
            n, k = nvb[s], min(M, nvb[s])
            ch = chance_jaccard(n, k)
            per_child[s] = {"jaccard": float(np.mean(js)), "chance": ch, "n_valid_units": n,
                            "n_models": len(sets)}
            jac.append(np.mean(js)); chance.append(ch)
        excess = np.array(jac) - np.array(chance)
        out[arm] = {"budget_units": int(M), "unit_frames": int(L),
                    "n_children": len(sel), "n_trivial": len(sel) - len(nontrivial),
                    "n_nontrivial": len(nontrivial),
                    "jaccard": summ(jac) if jac else None, "chance": summ(chance) if chance else None,
                    "excess_over_chance": summ(excess) if len(excess) else None,
                    "wilcoxon_p_excess": float(stats.wilcoxon(excess).pvalue) if len(excess) > 5 else None,
                    "per_child": per_child}
    return out


def attribution(device, max_files=None):
    p, S, y = part_info()
    files = sorted(glob.glob(os.path.join(CKPTS, "PACE", "*.pt")))[:max_files]
    if not files:
        return {}
    shares = {k: [] for k in STREAMS}
    regions = {k: [] for k in REGIONS}
    rand_shares = {k: [] for k in STREAMS}
    rand_regions = {k: [] for k in REGIONS}
    profile_corr = []
    occl = defaultdict(lambda: defaultdict(dict))         # stream -> repeat -> {subject: logit}
    for f in files:
        m, ck = load_ckpt(f, device)
        set_seed(RNG + len(profile_corr))
        rnd = build_model(ck["arm"], ck["config"]).to(device).eval()
        test = ck["test"]
        xs = torch.from_numpy(np.stack([prepare_sequence(raw(s)) for s in test])).to(device)
        vm = (xs.abs().sum(dim=(-2, -1)) > 1e-4).cpu().numpy()
        for model, sh, rg in ((m, shares, regions), (rnd, rand_shares, rand_regions)):
            gx = descriptor_gradxinput(model, xs)                       # (n, T, 198)
            prof = []
            for i in range(len(test)):
                g = gx[i][vm[i]]
                tot = g.sum() + 1e-12
                for k, sl in STREAMS.items():
                    sh[k].append(g[:, sl].sum() / tot)
                lm = g.reshape(-1, 3, 33, 2).sum(axis=(0, 3))[1:].sum(0)
                for k, j in REGIONS.items():
                    rg[k].append(lm[j].sum() / (lm.sum() + 1e-12))
                prof.append(lm / (lm.sum() + 1e-12))
            if model is m:
                trained_prof = np.mean(prof, 0)
            else:
                profile_corr.append(stats.spearmanr(trained_prof, np.mean(prof, 0)).correlation)
        # stream occlusion: zero one stream's descriptor columns at the encoder input
        layer = m.spatial_encoder.input_proj[0]
        for k, sl in [("none", None), *STREAMS.items()]:
            def hook(mod, args, sl=sl):
                if sl is None:
                    return None
                f_ = args[0].clone(); f_[:, sl] = 0.0
                return (f_,)
            h = layer.register_forward_pre_hook(hook)
            with torch.no_grad():
                lg = m(xs)[1].cpu().numpy()
            h.remove()
            occl[k][(ck["repeat"], ck["inner_fold"])].update(zip(test, lg))
    base = {}
    out = {"n_models": len(files),
           "stream_share": {k: summ(v) for k, v in shares.items()},
           "region_share_velocity_acceleration": {k: summ(v) for k, v in regions.items()},
           "random_init_stream_share": {k: summ(v) for k, v in rand_shares.items()},
           "random_init_region_share_velocity_acceleration": {k: summ(v) for k, v in rand_regions.items()},
           "trained_vs_random_landmark_profile_spearman": summ(profile_corr)}
    occ = {}
    for k, d in occl.items():
        aucs = []
        for key, vals in d.items():
            if len(vals) == len(S):
                aucs.append(roc_auc_score(y, [vals[s] for s in S]))
        occ[k] = summ(aucs) if aucs else None
    out["occlusion_auc_per_repeat_and_inner_fold"] = occ
    return out


# ── gait descriptors ──────────────────────────────────────────────────────────

def gait():
    p, S, y = part_info()
    g = json.load(open("results/cv/gait_features.json"))
    X = np.array([[np.nan if v is None else v for v in g["values"][s]] for s in S], float)
    rows = []
    rng = np.random.default_rng(RNG)
    for j, name in enumerate(g["names"]):
        x = X[:, j]; ok = np.isfinite(x)
        a, t = x[ok & (y == 1)], x[ok & (y == 0)]
        u = stats.mannwhitneyu(a, t)
        auc = u.statistic / (len(a) * len(t))
        bs = []
        for _ in range(1000):
            aa, tt = rng.choice(a, len(a)), rng.choice(t, len(t))
            bs.append(stats.mannwhitneyu(aa, tt).statistic / (len(aa) * len(tt)))
        rows.append({"name": name, "median_autistic": float(np.median(a)), "median_td": float(np.median(t)),
                     "auc": float(auc), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                     "p": float(u.pvalue), "n_missing": int((~ok).sum())})
    ps = np.array([r["p"] for r in rows])
    order = np.argsort(ps)
    q = np.empty_like(ps)
    q[order] = np.minimum.accumulate((ps[order] * len(ps) / (np.arange(len(ps)) + 1))[::-1])[::-1]
    for r, qq in zip(rows, np.minimum(q, 1)):
        r["q_bh"] = float(qq)
    lr = [json.load(open(f))["coef"] for f in glob.glob(os.path.join(RUNS, "Gait-LR", "r*_k*.json"))]
    coef = {k: summ([c[k] for c in lr]) for k in g["names"]} if lr else {}
    return {"descriptors": rows, "gait_lr_coefficients": coef}


# ── frame format ──────────────────────────────────────────────────────────────

def frame_format():
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from gait_features import FEATURE_NAMES, feature_matrix
    from run_cv import stored_array
    import yaml
    p, S, y = part_info()
    vid = pd.read_csv("processed/video_metadata.csv").set_index("clip_id")
    full = np.array([vid.loc[s, "width"] == 1080 and vid.loc[s, "height"] == 1920 for s in S])
    aspect = np.array([vid.loc[s, "width"] / vid.loc[s, "height"] for s in S])
    out = {"fps_values": sorted(set(float(v) for v in vid.loc[S, "fps"])),
           "cohort_all_60fps": bool((vid.loc[S, "fps"] == 60).all())}
    for g, lab in (("autistic", 1), ("td", 0)):
        m = y == lab
        out[g] = {"n": int(m.sum()), "uncropped_1080x1920": int(full[m].sum()),
                  "aspect": summ(aspect[m]), "n_distinct_sizes": int(len(set(
                      zip(vid.loc[np.array(S)[m], "width"], vid.loc[np.array(S)[m], "height"]))))}
    out["auc_uncropped"] = float(roc_auc_score(y, full.astype(float)))
    out["auc_aspect"] = float(roc_auc_score(y, aspect))
    cfg = yaml.safe_load(open("configs/config.yaml"))
    reps = {}
    for coords in ("pixel", "image"):
        c = {**cfg, "data": {**cfg["data"], "coordinates": coords}}
        reps[coords] = feature_matrix([prepare_sequence(stored_array(c, s)) for s in S])
    # within the autistic group: can the descriptors tell uncropped from cropped videos?
    a = y == 1
    for coords, X in reps.items():
        Xa, fa = X[a], full[a].astype(int)
        aucs = []
        for rep in range(20):
            pr = np.zeros(len(fa))
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(Xa, fa):
                clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                    LogisticRegression(C=1.0, max_iter=5000)).fit(Xa[tr], fa[tr])
                pr[te] = clf.predict_proba(Xa[te])[:, 1]
            aucs.append(roc_auc_score(fa, pr))
        rho = [stats.spearmanr(X[:, j][np.isfinite(X[:, j])], aspect[np.isfinite(X[:, j])]).correlation
               for j in range(X.shape[1])]
        out[f"within_autistic_format_auc_{coords}"] = summ(aucs)
        out[f"n_descriptors_abs_rho_aspect_gt_0.3_{coords}"] = int(np.sum(np.abs(rho) > 0.3))
        out[f"descriptor_aspect_rho_{coords}"] = dict(zip(FEATURE_NAMES, map(float, rho)))
    return out


# ── second revision: selection nulls (docs/validation_criteria.md, B3) ────────

def _valid_units(vm, L):
    nb = int(np.ceil(len(vm) / L))
    return np.array([vm[b * L:(b + 1) * L].any() for b in range(nb)])


def _motion_units(a, L, M):
    """Top-M valid blocks (frames) of a prepared sequence by mean motion
    energy over pairs of valid frames (the heuristic gate)."""
    vm = valid_mask(a)
    e = np.zeros(len(a)); pair = np.zeros(len(a), bool)
    pair[1:] = vm[1:] & vm[:-1]
    e[1:] = np.linalg.norm(a[1:] - a[:-1], axis=-1).sum(-1)
    vb = _valid_units(vm, L)
    sc = []
    for b in range(len(vb)):
        p = pair[b * L:(b + 1) * L]
        sc.append(e[b * L:(b + 1) * L][p].mean() if p.any() else 0.0)
    sc = np.where(vb, sc, -np.inf)
    order = np.argsort(-sc, kind="stable")
    return {int(b) for b in order[:min(M, int(vb.sum()))]}


def _mean_pairwise_jaccard(sets, pairs=None):
    idx = pairs if pairs is not None else [(i, j) for i in range(len(sets)) for j in range(i + 1, len(sets))]
    js = [len(sets[i] & sets[j]) / max(len(sets[i] | sets[j]), 1) for i, j in idx]
    return float(np.mean(js)) if js else np.nan


@torch.no_grad()
def selection_nulls(device):
    out = {}
    for arm in ("PACE", "PACE-M4", "PACE-frames"):
        files = sorted(glob.glob(os.path.join(CKPTS, arm, "*.pt")))
        if not files:
            continue
        trained, rand, reps = defaultdict(list), defaultdict(list), defaultdict(list)
        nvb, heur = {}, {}
        for n, f in enumerate(files):
            m, ck = load_ckpt(f, device)
            set_seed(RNG + 7919 * n)
            r = build_model(ck["arm"], ck["config"]).to(device).eval()
            L, M = m.saliency_gate.block_size, m.saliency_gate.top_m
            for s in ck["test"]:
                a = prepare_sequence(raw(s))
                vb = _valid_units(valid_mask(a), L)
                nvb[s] = int(vb.sum())
                if s not in heur:
                    heur[s] = _motion_units(a, L, M)
                x = torch.from_numpy(a)[None].to(device)
                for model, store in ((m, trained), (r, rand)):
                    _, idx, _ = model.get_attention_maps(x)
                    store[s].append({b for b in (idx[0].cpu().numpy() // L).tolist()
                                     if b < len(vb) and vb[b]})
                reps[s].append(ck["repeat"])
        rows = []
        for s in trained:
            if nvb[s] <= M:
                continue
            k = min(M, nvb[s])
            T, Rn, rp = trained[s], rand[s], reps[s]
            across = [(i, j) for i in range(len(T)) for j in range(i + 1, len(T)) if rp[i] != rp[j]]
            rows.append({"child": s, "chance": chance_jaccard(nvb[s], k),
                         "trained": _mean_pairwise_jaccard(T),
                         "random_init": _mean_pairwise_jaccard(Rn),
                         "trained_across_repeats": _mean_pairwise_jaccard(T, across),
                         "trained_vs_motion": float(np.mean([len(t & heur[s]) / max(len(t | heur[s]), 1)
                                                             for t in T])),
                         "random_vs_motion": float(np.mean([len(t & heur[s]) / max(len(t | heur[s]), 1)
                                                            for t in Rn]))})
        col = lambda k: np.array([q[k] for q in rows])
        res = {"budget_units": int(M), "unit_frames": int(L), "n_nontrivial": len(rows)}
        for k in ("chance", "trained", "random_init", "trained_across_repeats", "trained_vs_motion",
                  "random_vs_motion"):
            res[k] = summ(col(k))
        d = col("trained") - col("random_init")
        res["trained_minus_random"] = summ(d)
        res["wilcoxon_p_trained_vs_random"] = float(stats.wilcoxon(d).pvalue)
        res["wilcoxon_p_random_vs_chance"] = float(stats.wilcoxon(col("random_init") - col("chance")).pvalue)
        res["wilcoxon_p_motion_vs_chance"] = float(stats.wilcoxon(col("trained_vs_motion") - col("chance")).pvalue)
        res["per_child"] = rows
        out[arm] = res
    return out


# ── second revision: movement evidence (B4) ───────────────────────────────────

def _ols_group_effect(x, group, age, male, B=2000, seed=RNG):
    """Coefficient of group in an OLS regression of the standardized
    descriptor on group, age and sex; two-sided t-test p and bootstrap 95%
    interval (resampling children within groups)."""
    ok = np.isfinite(x)
    x, g, a, s = x[ok], group[ok].astype(float), age[ok], male[ok].astype(float)
    z = (x - x.mean()) / (x.std(ddof=1) + 1e-12)
    X = np.c_[np.ones(len(z)), g, a, s]

    def fit(ix):
        beta, *_ = np.linalg.lstsq(X[ix], z[ix], rcond=None)
        return beta
    beta = fit(np.arange(len(z)))
    resid = z - X @ beta
    dof = len(z) - X.shape[1]
    cov = np.linalg.pinv(X.T @ X) * (resid @ resid) / dof
    t = beta[1] / np.sqrt(cov[1, 1])
    p = float(2 * stats.t.sf(abs(t), dof))
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(g == 1), np.flatnonzero(g == 0)
    bs = [fit(np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]))[1]
          for _ in range(B)]
    return {"beta": float(beta[1]), "p": p, "ci95": [float(np.percentile(bs, 2.5)),
                                                      float(np.percentile(bs, 97.5))], "n": int(len(z))}


def _bh(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    q = np.empty_like(ps)
    q[order] = np.minimum.accumulate((ps[order] * len(ps) / (np.arange(len(ps)) + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def _auc_ci(y, x, B=1000, seed=RNG):
    ok = np.isfinite(x)
    y, x = y[ok], x[ok]
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    bs = []
    for _ in range(B):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        bs.append(fast_auc(y[i], x[i]))
    return {"auc": fast_auc(y, x), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def movement():
    import pandas as pd
    from gait_features import FEATURE_NAMES, feature_matrix
    from protocol import crop_valid
    from recording_features import JITTER_NAMES, SCENE_NAMES, jitter_features, smooth_sequence
    p, S, y = part_info()
    meta = pd.read_csv("processed/participant_metadata.csv").set_index("subject_id")
    age = np.array([meta.loc[s, "age_years"] for s in S], float)
    male = np.array([meta.loc[s, "sex"] == "M" for s in S])
    vid = pd.read_csv("processed/video_metadata.csv").set_index("clip_id")
    full = np.array([vid.loc[s, "width"] == 1080 and vid.loc[s, "height"] == 1920 for s in S])
    crop = ~full
    prep = [prepare_sequence(raw(s)) for s in S]
    span = np.array([mask_summary(a)["span"] for a in prep])
    versions = {"standard": feature_matrix(prep),
                "smoothed": feature_matrix([smooth_sequence(a) for a in prep]),
                "first90": feature_matrix([crop_valid(a, 90) for a in prep])}
    out = {"n_cropped": {"autistic": int((crop & (y == 1)).sum()), "td": int((crop & (y == 0)).sum())},
           "n_first90_cropped": {"autistic": int((crop & (span >= 90) & (y == 1)).sum()),
                                 "td": int((crop & (span >= 90) & (y == 0)).sum())}}
    # (b, c) descriptors within the cropped videos: unadjusted and adjusted for age and sex
    desc = {}
    for v, X in versions.items():
        sub = crop & ((span >= 90) if v == "first90" else True)
        rows = []
        for j, name in enumerate(FEATURE_NAMES):
            x = X[:, j]
            a_, t_ = x[sub & (y == 1)], x[sub & (y == 0)]
            a_, t_ = a_[np.isfinite(a_)], t_[np.isfinite(t_)]
            u = stats.mannwhitneyu(a_, t_)
            row = {"name": name, "median_autistic": float(np.median(a_)), "median_td": float(np.median(t_)),
                   "p_mwu": float(u.pvalue), **{f"auc_{k}": v_ for k, v_ in _auc_ci(y[sub], x[sub]).items()},
                   "adjusted": _ols_group_effect(x[sub], y[sub], age[sub], male[sub])}
            rows.append(row)
        for r, q in zip(rows, _bh([r["p_mwu"] for r in rows])):
            r["q_mwu"] = float(q)
        for r, q in zip(rows, _bh([r["adjusted"]["p"] for r in rows])):
            r["adjusted"]["q"] = float(q)
        desc[v] = rows
    robust = []
    for j, name in enumerate(FEATURE_NAMES):
        st, sm, f9 = (desc[v][j]["adjusted"] for v in ("standard", "smoothed", "first90"))
        same = lambda e: np.sign(e["beta"]) == np.sign(st["beta"]) and (e["ci95"][0] > 0 or e["ci95"][1] < 0)
        if st["q"] < 0.05 and same(sm) and same(f9):
            robust.append(name)
    out["descriptors_cropped"] = desc
    out["robust_descriptors"] = robust
    # all 95 children, adjusted (for comparison with the first-revision table)
    rows = []
    for j, name in enumerate(FEATURE_NAMES):
        rows.append({"name": name, "adjusted": _ols_group_effect(versions["standard"][:, j], y, age, male)})
    for r, q in zip(rows, _bh([r["adjusted"]["p"] for r in rows])):
        r["adjusted"]["q"] = float(q)
    out["descriptors_all_adjusted"] = rows
    # (c) jitter by group
    J = np.array([[jitter_features(a)[k] for k in JITTER_NAMES] for a in prep])
    out["jitter"] = {k: {"median_autistic": float(np.nanmedian(J[y == 1, i])),
                         "median_td": float(np.nanmedian(J[y == 0, i])),
                         "all": _auc_ci(y, J[:, i]), "cropped": _auc_ci(y[crop], J[crop, i]),
                         "autistic_uncropped_vs_cropped": _auc_ci(full[y == 1].astype(int), J[y == 1, i])["auc"],
                         "n_missing": int(np.isnan(J[:, i]).sum())}
                     for i, k in enumerate(JITTER_NAMES)}
    # descriptive (not in the plan): slow movement of the landmarks relative to the
    # hips over 0.5 s, against the jitter of the same recording
    from recording_features import _runs
    from scipy.ndimage import uniform_filter1d
    ratio = {"all": [], "arms_hands": [], "legs_feet": []}
    for a_ in prep:
        mv_, jt_ = [], []
        for st, en in _runs(valid_mask(a_), 40):
            seg = a_[st:en]
            sm_ = uniform_filter1d(seg, 9, axis=0, mode="nearest")
            jt_.append(np.linalg.norm(seg - sm_, axis=-1))
            mv_.append(np.linalg.norm(sm_[30:] - sm_[:-30], axis=-1))
        if not mv_:
            continue
        Jr = np.sqrt((np.concatenate(jt_) ** 2).mean(0)); Mr = np.sqrt((np.concatenate(mv_) ** 2).mean(0))
        for k, idx in (("all", list(range(33))), ("arms_hands", list(range(11, 23))), ("legs_feet", list(range(25, 33)))):
            ratio[k].append(float(np.median(Mr[idx] / Jr[idx])))
    out["movement_to_jitter_ratio_0p5s"] = {k: summ(v) for k, v in ratio.items()}
    # jitter relative to the range-type descriptors
    rng_names = ["step_amplitude", "arm_swing", "elbow_rom", "knee_rom", "step_width_var", "trunk_sway",
                 "head_sway", "head_bob", "wrist_lateral"]
    Xs = versions["standard"]
    out["jitter_descriptor_spearman"] = {
        n: float(stats.spearmanr(J[:, 0], Xs[:, FEATURE_NAMES.index(n)], nan_policy="omit").correlation)
        for n in rng_names}
    # (d) scene features by group
    sc_path = "results/cv/scene_features.json"
    if os.path.isfile(sc_path):
        sc = json.load(open(sc_path))
        Z = np.array([[sc[s][k] for k in SCENE_NAMES] for s in S], float)
        out["scene"] = {k: {"median_autistic_cropped": float(np.nanmedian(Z[crop & (y == 1), i])),
                            "median_td": float(np.nanmedian(Z[y == 0, i])),
                            "all": _auc_ci(y, Z[:, i]), "cropped": _auc_ci(y[crop], Z[crop, i])}
                        for i, k in enumerate(SCENE_NAMES)}
    # (f) representativeness of the cropped autistic children
    a = y == 1
    rep = {"n_cropped": int((a & crop).sum()), "n_uncropped": int((a & full).sum()),
           "age_cropped": summ(age[a & crop]), "age_uncropped": summ(age[a & full]),
           "age_mwu_p": float(stats.mannwhitneyu(age[a & crop], age[a & full]).pvalue),
           "boys_cropped": int(male[a & crop].sum()), "boys_uncropped": int(male[a & full].sum()),
           "sex_fisher_p": float(stats.fisher_exact([[male[a & crop].sum(), (~male[a & crop]).sum()],
                                                     [male[a & full].sum(), (~male[a & full]).sum()]])[1]),
           "detected_cropped": summ([valid_mask(raw(s)).sum() for s, c in zip(np.array(S)[a], crop[a]) if c]),
           "detected_uncropped": summ([valid_mask(raw(s)).sum() for s, c in zip(np.array(S)[a], crop[a]) if not c])}
    adj = []
    for j, name in enumerate(FEATURE_NAMES):
        x = Xs[a, j]
        ok = np.isfinite(x)
        adj.append({"name": name, "auc_cropped_vs_uncropped": fast_auc(crop[a][ok], x[ok]),
                    "p": float(stats.mannwhitneyu(x[ok & crop[a]], x[ok & full[a]]).pvalue)})
    for r, q in zip(adj, _bh([r["p"] for r in adj])):
        r["q"] = float(q)
    rep["descriptors"] = adj
    out["representativeness"] = rep
    return out


def depth():
    """Depth-camera skeletons (B5): descriptors and recording properties by
    group, and agreement of video and depth descriptors of the same walk."""
    import pandas as pd
    import depth_skeleton as ds
    from gait_features import FEATURE_NAMES, feature_matrix
    path = "results/cv/depth_features.json"
    if not os.path.isfile(path):
        return {}
    tab = json.load(open(path))
    p, S, y = part_info()
    vid = pd.read_csv("processed/video_metadata.csv").set_index("clip_id")
    crop = np.array([not (vid.loc[s, "width"] == 1080 and vid.loc[s, "height"] == 1920) for s in S])
    F = np.array([[tab[s]["features"][k] for k in ds.FEATURE_NAMES] for s in S], float)
    Rr = np.array([[tab[s]["recording"][k] for k in ds.RECORDING_NAMES] for s in S], float)
    meta = pd.read_csv("processed/participant_metadata.csv").set_index("subject_id")
    age = np.array([meta.loc[s, "age_years"] for s in S], float)
    male = np.array([meta.loc[s, "sex"] == "M" for s in S])
    out = {"frames": summ(Rr[:, 0]), "features": {}, "recording": {}}
    for i, k in enumerate(ds.FEATURE_NAMES):
        # the age- and sex-adjusted effect is exploratory (not in the B5 plan)
        out["features"][k] = {"median_autistic": float(np.median(F[y == 1, i])),
                              "median_td": float(np.median(F[y == 0, i])),
                              "all": _auc_ci(y, F[:, i]), "cropped": _auc_ci(y[crop], F[crop, i]),
                              "adjusted_all_exploratory": _ols_group_effect(F[:, i], y, age, male)}
    for i, k in enumerate(ds.RECORDING_NAMES):
        out["recording"][k] = {"median_autistic": float(np.median(Rr[y == 1, i])),
                               "median_td": float(np.median(Rr[y == 0, i])),
                               "all": _auc_ci(y, Rr[:, i]), "cropped": _auc_ci(y[crop], Rr[crop, i])}
    tod = Rr[:, ds.RECORDING_NAMES.index("time_of_day")]
    out["time_of_day_quartiles"] = {g: [float(v) for v in np.percentile(tod[y == lab], [0, 25, 50, 75, 100])]
                                    for g, lab in (("autistic", 1), ("td", 0))}
    # convergent validity: the same walk measured from video and from depth
    V = feature_matrix([prepare_sequence(raw(s)) for s in S])
    pairs = {"step_width": "step_width", "arm_swing": "arm_swing", "knee_rom": "knee_rom",
             "elbow_rom": "elbow_rom", "trunk_sway": "trunk_sway", "head_bob": "head_bob",
             "wrist_height": "wrist_height"}
    conv = {}
    for dk, vk in pairs.items():
        a, b = F[:, ds.FEATURE_NAMES.index(dk)], V[:, FEATURE_NAMES.index(vk)]
        ok = np.isfinite(a) & np.isfinite(b)
        rho = stats.spearmanr(a[ok], b[ok])
        conv[dk] = {"rho": float(rho.correlation), "p": float(rho.pvalue)}
    out["video_depth_spearman"] = conv
    return out


# ── runtime ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def runtime():
    out = {"training_seconds_per_model": {}}
    for arm in os.listdir(RUNS):
        if arm == "pilot":
            continue
        secs, eps, npar = [], [], None
        for f in glob.glob(os.path.join(RUNS, arm, "r*_k*.json")):
            rec = json.load(open(f))
            for inf in rec.get("inner", []) if isinstance(rec.get("inner"), list) else []:
                if isinstance(inf, dict) and "seconds" in inf:
                    secs.append(inf["seconds"]); eps.append(inf["epochs_run"]); npar = inf["n_parameters"]
        if secs:
            out["training_seconds_per_model"][arm] = {"seconds": summ(secs), "epochs": summ(eps),
                                                      "n_parameters": npar, "total_minutes": float(sum(secs) / 60)}
    rel = sorted(glob.glob("models/release/*.pt"))
    if rel:
        x = torch.from_numpy(prepare_sequence(raw("asd_1")))[None]
        for dev_name in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            dev = torch.device(dev_name)
            m, _ = load_ckpt(rel[0], dev)
            ms = [load_ckpt(f, dev)[0] for f in rel]
            xx = x.to(dev)
            for _ in range(20):
                m(xx)
            def timeit(fn, n):
                t = []
                for _ in range(n):
                    if dev_name == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter(); fn()
                    if dev_name == "cuda":
                        torch.cuda.synchronize()
                    t.append((time.perf_counter() - t0) * 1000)
                return summ(t)
            out[f"single_model_ms_{dev_name}"] = timeit(lambda: m(xx), 200)
            out[f"ensemble_{len(ms)}_ms_{dev_name}"] = timeit(lambda: [mm(xx) for mm in ms], 50)
        out["device_names"] = {"cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        out["n_parameters_pace"] = int(sum(p.numel() for p in m.parameters() if p.requires_grad))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", nargs="*", default=None)
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    want = lambda s: a.sections is None or s in a.sections
    res = json.load(open(OUT)) if os.path.isfile(OUT) else {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name, fn in (("cohort", cohort), ("shortcuts", lambda: shortcuts(a.repeats)),
                     ("confounds", lambda: confounds(a.repeats)),
                     ("selection", lambda: selection(device)),
                     ("attribution", lambda: attribution(device)), ("gait", gait),
                     ("format", frame_format),
                     ("runtime", runtime),
                     ("selection_nulls", lambda: selection_nulls(device)),
                     ("movement", movement), ("depth", depth)):
        if want(name):
            t0 = time.time()
            res[name] = fn()
            json.dump(res, open(OUT, "w"), indent=1)
            print(f"{name} done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
