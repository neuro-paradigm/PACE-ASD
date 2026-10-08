"""
PACE — planted-event benchmark: can the selection mechanism find an event
whose position is known?

Backgrounds are the prepared walking sequences of the cohort. From each
child's sequence two synthetic sequences are made: one with a planted event
(label 1) and one without (label 0), so the background carries no
information about the label. The event lasts DURATION frames under a
flat-topped (Tukey) window, at a uniformly random position inside the
detected span (the same positions for every event type). Three event types:

  raise   both hands and wrists (landmarks 15-22; elbows at half the
          displacement) are raised by `amp` shoulder widths and oscillate
          vertically by OSC shoulder widths with a period of PERIOD frames
  flap    the same oscillation with amplitude `amp`, without the raise
  freeze  arrest of movement: every landmark is moved towards its smoothed
          position at the event's first frame plus the recording's own
          high-frequency residual, by the fraction `amp` of the window

Three reference detectors (no training) measure how detectable an event is
and where it is: D1 hand height, D2 motion energy, D3 a 6-Hz matched filter
on the hand height (`detectors`). Models are trained with the standard
recipe in a 5-fold cross-validation grouped by background child (one inner
validation fold per outer fold) and evaluated on the held-out backgrounds:

  detection     AUC for event versus no event
  coverage      share of the event's frames inside the selected tokens,
                against the share expected from selecting the same number
                of valid blocks (or frames) at random
  top-1 hit     whether the highest-scoring valid block (frame) overlaps
                the event
  attribution   share of descriptor-level |gradient x input| falling on event
                frames and on the arm and hand landmarks, for the trained
                model and for the same architecture with random weights
                (model-randomisation check)

The settings of the second-revision runs (gate remedies, event types,
amplitude rule) are fixed in docs/validation_criteria.md.

Usage:
    python scripts/synthetic_events.py --event raise --amplitudes 1.0 1.5 2.0 3.0 --jobs 3
    python scripts/synthetic_events.py --reference-grid
    python scripts/synthetic_events.py --summarize
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np
import torch
import yaml
from scipy import stats
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)

from attribution import descriptor_gradxinput              # noqa: E402
from protocol import arm_config, build_model, fit_model, predict_logits, set_seed  # noqa: E402
from run_cv import load_arrays, make_partition                   # noqa: E402
from sequence import fill_internal_gaps, valid_mask              # noqa: E402

EVENT_LANDMARKS = list(range(15, 23))
ELBOWS = [13, 14]
ARMS_HANDS = list(range(11, 23))
DURATION = 30          # frames (0.5 s at 60 frames per second)
PERIOD = 10            # frames per oscillation cycle (6 Hz)
OSC = 0.2              # oscillation amplitude of the raise event, shoulder widths
BLOCK = 15
SEED = 20261009
OUT = os.path.join("results", "synthetic")

SYN_ARMS = {           # name: (arm, overrides)
    "blocks-M2": ("PACE", {"model.event_top_m": 2}),
    "frames-K30": ("PACE-frames", {"model.event_top_m": 30}),
    "blocks-M8": ("PACE", {}),
    "nogate": ("PACE-nogate", {}),
    # second revision: remedies for the hard top-M gate and a heuristic gate
    "anneal-M2": ("PACE", {"model.event_top_m": 2, "model.gate_train_mode": "anneal"}),
    "gumbel-M2": ("PACE", {"model.event_top_m": 2, "model.gate_train_mode": "gumbel"}),
    "perturbed-M2": ("PACE", {"model.event_top_m": 2, "model.gate_train_mode": "perturbed"}),
    "motion-M2": ("PACE", {"model.event_top_m": 2, "model.gate_scores": "motion"}),
}
EVENTS = ("raise", "flap", "freeze")
GRID = {"raise": [1.0, 1.5, 2.0, 3.0],
        "flap": [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0],
        "freeze": [0.2, 0.4, 0.6, 0.8, 1.0]}
INFORMATIVE = (0.70, 0.95)       # best reference AUC for an informative amplitude
TARGETS = (0.75, 0.90)           # amplitude rule for the new event types


def _window():
    from scipy.signal.windows import tukey
    return tukey(DURATION, 0.5)


def plant(seq, start, amp, event="raise"):
    out = seq.copy()
    t = np.arange(DURATION)
    env = _window()
    seg = slice(start, start + DURATION)
    vm = valid_mask(seq)[seg]
    n = len(vm)
    if event in ("raise", "flap"):
        lift = amp if event == "raise" else 0.0
        osc = OSC if event == "raise" else amp
        disp = ((-lift + osc * np.sin(2 * np.pi * t / PERIOD)) * env)[:n]   # image y points down
        out[seg, EVENT_LANDMARKS, 1] += (disp * vm)[:, None]
        out[seg, ELBOWS, 1] += (0.5 * disp * vm)[:, None]
        return out
    if event == "freeze":
        full = valid_mask(seq)
        idx = np.flatnonzero(full)
        a, b = idx[0], idx[-1] + 1
        span = fill_internal_gaps(seq)[a:b]
        smooth = np.zeros_like(seq)
        smooth[a:b] = uniform_filter1d(span, size=9, axis=0, mode="nearest")
        resid = np.zeros_like(seq)
        resid[a:b] = span - smooth[a:b]
        target = smooth[start][None] + resid[seg]
        w = (amp * env[:n] * vm)[:, None, None]
        out[seg] = seq[seg] + w * (target - seq[seg])
        return out
    raise ValueError(event)


def make_dataset(arrays, amp, event="raise", seed=SEED):
    rng = np.random.default_rng(seed)
    data = {}
    for s in sorted(arrays):
        seq = arrays[s]
        idx = np.flatnonzero(valid_mask(seq))
        first, last = idx[0], idx[-1]
        start = int(rng.integers(first, max(first + 1, last - DURATION + 2)))
        data[f"{s}+"] = {"x": plant(seq, start, amp, event), "y": 1, "bg": s,
                         "event": [start, start + DURATION]}
        data[f"{s}-"] = {"x": seq, "y": 0, "bg": s, "event": None}
    return data


# ── reference detectors ───────────────────────────────────────────────────────

def _signals(x):
    """Hand height relative to the shoulders and motion energy over the
    detected span of a prepared (onset-aligned) sequence."""
    vm = valid_mask(x)
    xf = fill_internal_gaps(x)
    n = int(valid_mask(xf).sum())
    h = (xf[:n, 15:23, 1] - xf[:n, [11, 12], 1].mean(1, keepdims=True)).mean(1)
    e = np.zeros(n)
    pair = vm[1:n] & vm[:n - 1]
    e[1:] = np.linalg.norm(x[1:n] - x[:n - 1], axis=-1).sum(-1) * pair
    return h, e, vm[:n]


def _osc_power(h, i):
    w = h[i:i + DURATION]
    w = w - w.mean()
    return float(np.abs(np.sum(w * np.exp(-2j * np.pi * np.arange(len(w)) / PERIOD))))


def detectors(x):
    """Sequence scores of D1-D3 and, for each, the 15-frame block it points
    at (index into the blocks of the prepared sequence)."""
    h, e, vm = _signals(x)
    n = len(h)
    starts = range(max(1, n - DURATION))                 # D1 exactly as in the first revision
    d1 = float(np.median(h) - min(h[i:i + DURATION].mean() for i in starts))
    starts2 = range(max(1, n - DURATION + 1))
    med_e = np.median(e[1:]) if n > 1 else 0.0
    d2 = float(max(abs(e[i:i + DURATION].mean() - med_e) for i in starts2))
    d3 = max(_osc_power(h, i) for i in starts2)
    nb = int(np.ceil(n / BLOCK))
    vb = [b for b in range(nb) if vm[b * BLOCK:(b + 1) * BLOCK].any()]
    loc = {
        "D1": min(vb, key=lambda b: h[b * BLOCK:(b + 1) * BLOCK].mean()),
        "D2": max(vb, key=lambda b: abs(e[b * BLOCK:(b + 1) * BLOCK].mean() - med_e)),
        "D3": max(vb, key=lambda b: _osc_power(h, min(b * BLOCK, max(0, n - DURATION)))),
    }
    return {"D1": d1, "D2": d2, "D3": d3}, loc


def reference_detection(arrays, amp, event="raise"):
    """AUC of each reference detector for event versus no event over all
    backgrounds, and its block-level localization (top-1 hit) on the event
    sequences, with the per-sequence hits for paired comparisons."""
    data = make_dataset(arrays, amp, event)
    ys, sc, hits = [], {k: [] for k in ("D1", "D2", "D3")}, {k: {} for k in ("D1", "D2", "D3")}
    chance = []
    for key, d in data.items():
        s, loc = detectors(d["x"])
        ys.append(d["y"])
        for k in sc:
            sc[k].append(s[k])
        if d["event"] is not None:
            vm = valid_mask(d["x"])
            ev = np.zeros(len(vm), bool); ev[d["event"][0]:d["event"][1]] = True
            ev &= vm
            nb = int(np.ceil(len(vm) / BLOCK))
            vb = [b for b in range(nb) if vm[b * BLOCK:(b + 1) * BLOCK].any()]
            evb = {b for b in vb if ev[b * BLOCK:(b + 1) * BLOCK].any()}
            chance.append(len(evb) / len(vb))
            for k in hits:
                hits[k][key] = bool(loc[k] in evb)
    out = {"auc": {k: float(roc_auc_score(ys, v)) for k, v in sc.items()},
           "top1": {k: float(np.mean(list(v.values()))) for k, v in hits.items()},
           "top1_chance": float(np.mean(chance)), "hits": hits}
    out["best"] = max(out["auc"], key=out["auc"].get)
    out["best_auc"] = out["auc"][out["best"]]
    return out


def amplitude_grid():
    """Best reference AUC over the grid of each event type and the amplitudes
    chosen by the rule of docs/validation_criteria.md (B2). No network is
    trained. Writes results/synthetic/reference_grid.json."""
    config = yaml.safe_load(open("configs/config.yaml"))
    part = make_partition(config)
    arrays = load_arrays(config, part["subjects"], True)
    res = {"informative_range": INFORMATIVE, "targets": TARGETS}
    for event, grid in GRID.items():
        rows = {}
        for a in grid:
            r = reference_detection(arrays, a, event)
            rows[str(a)] = {"auc": r["auc"], "best": r["best"], "best_auc": r["best_auc"],
                            "top1": r["top1"], "top1_chance": r["top1_chance"],
                            "informative": INFORMATIVE[0] <= r["best_auc"] <= INFORMATIVE[1]}
            print(f"[grid] {event} {a}: {r['auc']}", flush=True)
        chosen = []
        if event != "raise":
            if max(v["best_auc"] for v in rows.values()) < INFORMATIVE[0]:
                chosen = [grid[-1]]
            else:
                for target in TARGETS:
                    best = min(grid, key=lambda a: (abs(rows[str(a)]["best_auc"] - target), a))
                    if best not in chosen:
                        chosen.append(best)
        res[event] = {"grid": rows, "chosen": chosen}
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "reference_grid.json"), "w"), indent=1)
    return res


# ── training and evaluation ───────────────────────────────────────────────────

def splits(backgrounds, seed=SEED):
    bgs = np.array(sorted(backgrounds))
    out = []
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(bgs):
        rng = np.random.default_rng(seed + len(out))
        tr = rng.permutation(bgs[tr])
        n_val = len(tr) // 3
        out.append({"train": sorted(tr[n_val:].tolist()), "val": sorted(tr[:n_val].tolist()),
                    "test": sorted(bgs[te].tolist())})
    return out


@torch.no_grad()
def selection(model, x, device):
    """Selected frame indices and per-block (per-frame) gate scores."""
    attn, idx, scores = model.get_attention_maps(torch.from_numpy(x)[None].to(device))
    return idx[0].cpu().numpy(), None if scores is None else scores[0].cpu().numpy()


def evaluate(model, data, test_ids, device, L):
    keys = [k for k in data if data[k]["bg"] in test_ids]
    xs = [data[k]["x"] for k in keys]
    ys = np.array([data[k]["y"] for k in keys])
    lg = predict_logits(model, xs, device)
    res = {"auc": float(roc_auc_score(ys, lg)), "per_sequence": []}
    for k, x in zip(keys, xs):
        d = data[k]
        if d["event"] is None:
            continue
        s0, s1 = d["event"]
        vm = valid_mask(x)
        ev = np.zeros(len(x), bool); ev[s0:s1] = True
        ev &= vm
        rec = {"key": k, "n_valid": int(vm.sum())}
        if model.use_gate:
            idx, scores = selection(model, x, device)
            sel = np.zeros(len(x), bool); sel[idx[idx < len(x)]] = True
            sel &= vm
            rec["coverage"] = float((sel & ev).sum() / max(ev.sum(), 1))
            n_blocks_valid = int(np.ceil(len(x) / L))
            vb = np.array([vm[b * L:(b + 1) * L].any() for b in range(n_blocks_valid)])
            n_sel_blocks = len(set((idx // L).tolist()) & set(np.flatnonzero(vb).tolist()))
            ev_blocks = {b for b in range(n_blocks_valid) if ev[b * L:(b + 1) * L].any()}
            # expected coverage if the same number of valid blocks were drawn at random
            frac = n_sel_blocks / max(vb.sum(), 1)
            rec["coverage_chance"] = float(min(1.0, frac))
            sc = np.where(vb, scores[:len(vb)], -np.inf)
            rec["top1_hit"] = bool(int(np.argmax(sc)) in ev_blocks)
            rec["top1_chance"] = float(len(ev_blocks) / max(vb.sum(), 1))
        gx = descriptor_gradxinput(model, torch.from_numpy(x)[None].to(device))[0]   # (T, 198)
        per_frame = gx.sum(1) * vm
        per_lm = gx[vm].reshape(-1, 3, 33, 2).sum(axis=(0, 1, 3))
        rec["attr_event_share"] = float(per_frame[ev].sum() / (per_frame.sum() + 1e-12))
        rec["attr_event_chance"] = float(ev.sum() / vm.sum())
        rec["attr_arms_share"] = float(per_lm[ARMS_HANDS].sum() / (per_lm.sum() + 1e-12))
        res["per_sequence"].append(rec)
    return res


def run_dir(event, amp):
    return os.path.join(OUT, f"amp{amp}" if event == "raise" else f"{event}{amp}")


def run_one(arm_name, event, amp, fold, device):
    out_dir = os.path.join(run_dir(event, amp), arm_name)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"fold{fold}.json")
    if os.path.isfile(path):
        return
    config = yaml.safe_load(open("configs/config.yaml"))
    arm, overrides = SYN_ARMS[arm_name]
    cfg = arm_config(arm, config, overrides)
    part = make_partition(cfg)
    arrays = load_arrays(cfg, part["subjects"], True)
    data = make_dataset(arrays, amp, event)
    sp = splits(arrays)[fold]
    pick = lambda bgs: [k for k in sorted(data) if data[k]["bg"] in set(bgs)]
    tr, va = pick(sp["train"]), pick(sp["val"])
    model, scaler, info = fit_model(arm, cfg, [data[k]["x"] for k in tr], [data[k]["y"] for k in tr],
                                    [data[k]["x"] for k in va], [data[k]["y"] for k in va],
                                    SEED + fold, device)
    L = cfg["model"]["event_block_size"]
    res = {"arm": arm_name, "event": event, "amplitude": amp, "fold": fold, "train_info": info,
           "trained": evaluate(model, data, set(sp["test"]), device, L)}
    set_seed(SEED + 100 + fold)
    rand = build_model(arm, cfg).to(device).eval()
    res["random_weights"] = evaluate(rand, data, set(sp["test"]), device, L)
    json.dump(res, open(path, "w"))
    print(f"[synthetic] {event} {arm_name} amp={amp} fold={fold} auc={res['trained']['auc']:.3f}", flush=True)


LEARNED_AUC = 0.9


def sign_test_p(hits, chance):
    """One-sided sign test of per-sequence (hit - chance) > 0."""
    d = np.asarray(hits, float) - np.asarray(chance, float)
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    return float(stats.binomtest(pos, pos + neg, 0.5, alternative="greater").pvalue) if pos + neg else 1.0


def summarize():
    """results/synthetic/summary.json: per event type, amplitude and
    configuration, the detection AUC over folds and the localisation measures
    over held-out sequences with an event, for trained and randomly
    initialised networks, and the reference detectors."""
    out = {"duration_frames": DURATION, "period_frames": PERIOD, "oscillation": OSC,
           "learned_auc": LEARNED_AUC}
    config = yaml.safe_load(open("configs/config.yaml"))
    part = make_partition(config)
    arrays = load_arrays(config, part["subjects"], True)
    for d in sorted(glob.glob(os.path.join(OUT, "*"))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        event = next((e for e in ("flap", "freeze") if name.startswith(e)), "raise")
        amp = name[3:] if event == "raise" else name[len(event):]
        key = amp if event == "raise" else name
        for arm_dir in sorted(glob.glob(os.path.join(d, "*"))):
            recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(arm_dir, "fold*.json")))]
            if len(recs) < 5:
                continue
            res = {"n_folds": len(recs),
                   "train_minutes": float(sum(r["train_info"]["seconds"] for r in recs) / 60)}
            for kind in ("trained", "random_weights"):
                seqs = [q for r in recs for q in r[kind]["per_sequence"]]
                k = {"auc_mean": float(np.mean([r[kind]["auc"] for r in recs])),
                     "auc_folds": [r[kind]["auc"] for r in recs], "n_event_sequences": len(seqs)}
                for m in ("coverage", "coverage_chance", "top1_hit", "top1_chance",
                          "attr_event_share", "attr_event_chance", "attr_arms_share"):
                    vals = [float(q[m]) for q in seqs if m in q]
                    if vals:
                        k[m] = float(np.mean(vals))
                if "coverage" in seqs[0]:
                    diff = np.array([q["coverage"] - q["coverage_chance"] for q in seqs])
                    k["coverage_excess_p"] = float(stats.wilcoxon(diff).pvalue)
                    k["top1_sign_p"] = sign_test_p([q["top1_hit"] for q in seqs],
                                                   [q["top1_chance"] for q in seqs])
                    k["top1_folds"] = [float(np.mean([q["top1_hit"] for q in r[kind]["per_sequence"]]))
                                       for r in recs]
                # learning can be all-or-nothing across folds: report how many folds
                # detected the event (AUC >= LEARNED_AUC) and localisation within them
                learned = [r for r in recs if r[kind]["auc"] >= LEARNED_AUC]
                k["n_learned"] = len(learned)
                k["auc_min"], k["auc_max"] = float(min(k["auc_folds"])), float(max(k["auc_folds"]))
                lseqs = [q for r in learned for q in r[kind]["per_sequence"]]
                for m in ("coverage", "coverage_chance", "top1_hit", "top1_chance", "attr_event_share"):
                    vals = [float(q[m]) for q in lseqs if m in q]
                    if vals:
                        k[m + "_learned"] = float(np.mean(vals))
                res[kind] = k
            out.setdefault(key, {})[os.path.basename(arm_dir)] = res
        if key in out:
            ref = reference_detection(arrays, float(amp), event)
            out[key]["event"] = event
            out[key]["amplitude"] = float(amp)
            out[key]["reference_detection_auc"] = ref["auc"]["D1"]
            out[key]["reference"] = {k: v for k, v in ref.items() if k != "hits"}
            out[key]["reference"]["informative"] = INFORMATIVE[0] <= ref["best_auc"] <= INFORMATIVE[1]
            json.dump(ref["hits"], open(os.path.join(d, "reference_hits.json"), "w"))
    json.dump(out, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", choices=EVENTS, default="raise")
    ap.add_argument("--amplitudes", type=float, nargs="+", default=None)
    ap.add_argument("--arms", nargs="+", default=["blocks-M2", "frames-K30", "blocks-M8", "nogate"])
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--one", nargs=4, default=None, metavar=("ARM", "EVENT", "AMP", "FOLD"))
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--reference-grid", action="store_true")
    args = ap.parse_args()
    if args.summarize:
        summarize()
        return
    if args.reference_grid:
        amplitude_grid()
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    if args.one:
        run_one(args.one[0], args.one[1], float(args.one[2]), int(args.one[3]), device)
        return
    amps = args.amplitudes or GRID[args.event]
    jobs = [[sys.executable, __file__, "--one", a, args.event, str(amp), str(f)]
            for amp in amps for a in args.arms for f in range(5)]
    running = []
    while jobs or running:
        while jobs and len(running) < args.jobs:
            running.append(subprocess.Popen(jobs.pop(0)))
        running[0].wait()
        running = [p for p in running if p.poll() is None]


if __name__ == "__main__":
    main()
