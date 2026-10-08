"""
PACE-ASD — planted-event benchmark: can the selection mechanism find an event
whose position is known?

Backgrounds are the prepared walking sequences of the cohort. From each
child's sequence two synthetic sequences are made: one with a planted event
(label 1) and one without (label 0), so the background carries no
information about the label. The event resembles hand flapping: both hands
and wrists (landmarks 15-22; elbows at half the displacement) are raised by
`amp` shoulder widths and oscillate vertically by OSC shoulder widths with a
period of PERIOD frames, under a flat-topped (Tukey) window lasting DURATION
frames, at a uniformly random position inside the detected span. (A pure
oscillation without the raise was not separable from the tracking noise of
the hand landmarks even by a matched filter, so the raise is what makes the
event detectable; `reference_detection` reports how detectable each
amplitude is for a simple hand-height detector.) Models are trained with the standard recipe in a
5-fold cross-validation grouped by background child (one inner validation
fold per outer fold) and evaluated on the held-out backgrounds:

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

Usage:
    python scripts/synthetic_events.py --amplitudes 1.0 1.5 2.0 3.0 --jobs 3
    python scripts/synthetic_events.py --summarize
"""

import argparse
import copy
import json
import os
import subprocess
import sys

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)

from attribution import descriptor_gradxinput              # noqa: E402
from protocol import arm_config, build_model, fit_model, predict_logits, set_seed  # noqa: E402
from run_cv import load_arrays, make_partition                   # noqa: E402
from sequence import valid_mask                                  # noqa: E402

EVENT_LANDMARKS = list(range(15, 23))
ELBOWS = [13, 14]
ARMS_HANDS = list(range(11, 23))
DURATION = 30          # frames (0.5 s at 60 frames per second)
PERIOD = 10            # frames per oscillation cycle (6 Hz)
OSC = 0.2              # oscillation amplitude, shoulder widths
SEED = 20261009

SYN_ARMS = {           # name: (arm, overrides)
    "blocks-M2": ("PACE", {"model.event_top_m": 2}),
    "frames-K30": ("PACE-frames", {"model.event_top_m": 30}),
    "blocks-M8": ("PACE", {}),
    "nogate": ("PACE-nogate", {}),
}


def plant(seq, start, amp):
    from scipy.signal.windows import tukey
    out = seq.copy()
    t = np.arange(DURATION)
    env = tukey(DURATION, 0.5)
    disp = (-amp + OSC * np.sin(2 * np.pi * t / PERIOD)) * env     # image y points down
    vm = valid_mask(seq)[start:start + DURATION]
    out[start:start + DURATION, EVENT_LANDMARKS, 1] += (disp * vm)[:, None]
    out[start:start + DURATION, ELBOWS, 1] += (0.5 * disp * vm)[:, None]
    return out


def make_dataset(arrays, amp, seed=SEED):
    rng = np.random.default_rng(seed)
    data = {}
    for s in sorted(arrays):
        seq = arrays[s]
        idx = np.flatnonzero(valid_mask(seq))
        first, last = idx[0], idx[-1]
        start = int(rng.integers(first, max(first + 1, last - DURATION + 2)))
        data[f"{s}+"] = {"x": plant(seq, start, amp), "y": 1, "bg": s, "event": [start, start + DURATION]}
        data[f"{s}-"] = {"x": seq, "y": 0, "bg": s, "event": None}
    return data


def reference_detection(arrays, amp):
    """AUC of a simple detector (lowest 30-frame mean of hand height relative
    to the shoulders, against the recording's median) for event versus no
    event, over all backgrounds."""
    from sequence import fill_internal_gaps
    data = make_dataset(arrays, amp)
    ys, sc = [], []
    for d in data.values():
        x = fill_internal_gaps(d["x"])
        n = int(valid_mask(x).sum())
        h = (x[:n, 15:23, 1] - x[:n, [11, 12], 1].mean(1, keepdims=True)).mean(1)
        sc.append(float(np.median(h) - min(h[i:i + DURATION].mean() for i in range(max(1, n - DURATION)))))
        ys.append(d["y"])
    return float(roc_auc_score(ys, sc))


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


def run_one(arm_name, amp, fold, device):
    out_dir = os.path.join("results", "synthetic", f"amp{amp}", arm_name)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"fold{fold}.json")
    if os.path.isfile(path):
        return
    config = yaml.safe_load(open("configs/config.yaml"))
    arm, overrides = SYN_ARMS[arm_name]
    cfg = arm_config(arm, config, overrides)
    part = make_partition(cfg)
    arrays = load_arrays(cfg, part["subjects"], True)
    data = make_dataset(arrays, amp)
    sp = splits(arrays)[fold]
    pick = lambda bgs: [k for k in sorted(data) if data[k]["bg"] in set(bgs)]
    tr, va = pick(sp["train"]), pick(sp["val"])
    model, scaler, info = fit_model(arm, cfg, [data[k]["x"] for k in tr], [data[k]["y"] for k in tr],
                                    [data[k]["x"] for k in va], [data[k]["y"] for k in va],
                                    SEED + fold, device)
    L = cfg["model"]["event_block_size"]
    res = {"arm": arm_name, "amplitude": amp, "fold": fold, "train_info": info,
           "trained": evaluate(model, data, set(sp["test"]), device, L)}
    set_seed(SEED + 100 + fold)
    rand = build_model(arm, cfg).to(device).eval()
    res["random_weights"] = evaluate(rand, data, set(sp["test"]), device, L)
    json.dump(res, open(path, "w"))
    print(f"[synthetic] {arm_name} amp={amp} fold={fold} auc={res['trained']['auc']:.3f}", flush=True)


LEARNED_AUC = 0.9


def summarize():
    """results/synthetic/summary.json: per amplitude and configuration, the
    detection AUC over folds and the localisation measures over held-out
    sequences with an event, for trained and randomly initialised networks."""
    import glob
    out = {"duration_frames": DURATION, "period_frames": PERIOD, "oscillation": OSC,
           "learned_auc": LEARNED_AUC}
    config = yaml.safe_load(open("configs/config.yaml"))
    part = make_partition(config)
    arrays = load_arrays(config, part["subjects"], True)
    for d in sorted(glob.glob(os.path.join("results", "synthetic", "amp*"))):
        amp = os.path.basename(d)[3:]
        for arm_dir in sorted(glob.glob(os.path.join(d, "*"))):
            recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(arm_dir, "fold*.json")))]
            if len(recs) < 5:
                continue
            res = {"n_folds": len(recs)}
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
                    k["coverage_excess_p"] = float(__import__("scipy").stats.wilcoxon(diff).pvalue)
                # learning is all-or-nothing across folds: report how many folds
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
            out.setdefault(amp, {})[os.path.basename(arm_dir)] = res
        if amp in out:
            out[amp]["reference_detection_auc"] = reference_detection(arrays, float(amp))
    json.dump(out, open(os.path.join("results", "synthetic", "summary.json"), "w"), indent=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amplitudes", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0])
    ap.add_argument("--arms", nargs="+", default=list(SYN_ARMS))
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--one", nargs=3, default=None, metavar=("ARM", "AMP", "FOLD"))
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if args.summarize:
        summarize()
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    if args.one:
        run_one(args.one[0], float(args.one[1]), int(args.one[2]), device)
        return
    jobs = [[sys.executable, __file__, "--one", a, str(amp), str(f)]
            for amp in args.amplitudes for a in args.arms for f in range(5)]
    running = []
    while jobs or running:
        while jobs and len(running) < args.jobs:
            running.append(subprocess.Popen(jobs.pop(0)))
        running[0].wait()
        running = [p for p in running if p.poll() is None]


if __name__ == "__main__":
    main()
