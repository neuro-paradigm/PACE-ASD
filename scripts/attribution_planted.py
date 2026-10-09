"""
PACE — integrated gradients and temporal occlusion on planted events
(docs/validation_criteria_phase3.md, D13).

The two-block gate is retrained for the raise (A = 1) and the flap (A = 0.6)
with the folds, seeds and recipe of scripts/synthetic_events.py; the held-out
AUC of each fold is compared with the stored run (a reproduction check). For
every held-out sequence with an event, the share of |integrated gradients|
and of the temporal-occlusion effect on event frames (blocks) is computed for
the trained network and for the same architecture with random weights, against
the share expected by chance.

Usage:
    python scripts/attribution_planted.py
"""

import json
import os
import sys

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)

from phase3_analyses import integrated_gradients, temporal_occlusion   # noqa: E402
from protocol import arm_config, build_model, fit_model, predict_logits, set_seed  # noqa: E402
from run_cv import load_arrays, make_partition                          # noqa: E402
from sequence import valid_mask                                         # noqa: E402
import synthetic_events as se                                           # noqa: E402

OUT = os.path.join("results", "phase3", "attribution_planted.json")
CASES = [("raise", 1.0), ("flap", 0.6)]


def shares(model, data, keys, device):
    ig_sh, ig_ch, oc_sh, oc_ch = [], [], [], []
    for k in keys:
        d = data[k]
        x = d["x"]
        vm = valid_mask(x)
        ev = np.zeros(len(x), bool); ev[d["event"][0]:d["event"][1]] = True
        ev &= vm
        g = integrated_gradients(model, torch.from_numpy(x)[None].to(device))[0].sum(1) * vm
        ig_sh.append(g[ev].sum() / (g.sum() + 1e-12)); ig_ch.append(ev.sum() / vm.sum())
        blocks, occ = temporal_occlusion(model, x)
        evb = np.array([ev[b * 15:(b + 1) * 15].any() for b in blocks])
        oc_sh.append(float(occ[evb].sum())); oc_ch.append(float(evb.mean()))
    return {"ig_event_share": float(np.mean(ig_sh)), "ig_event_chance": float(np.mean(ig_ch)),
            "occlusion_event_share": float(np.mean(oc_sh)), "occlusion_event_chance": float(np.mean(oc_ch))}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = json.load(open(OUT)) if os.path.isfile(OUT) else {}
    config = yaml.safe_load(open("configs/config.yaml"))
    arm, overrides = se.SYN_ARMS["blocks-M2"]
    cfg = arm_config(arm, config, overrides)
    part = make_partition(cfg)
    arrays = load_arrays(cfg, part["subjects"], True)
    for event, amp in CASES:
        key = f"{event}{amp}"
        if key in res and len(res[key]["folds"]) == 5:
            continue
        data = se.make_dataset(arrays, amp, event)
        folds = []
        stored_dir = se.run_dir(event, amp)
        for fold, sp in enumerate(se.splits(arrays)):
            pick = lambda bgs: [k for k in sorted(data) if data[k]["bg"] in set(bgs)]
            tr, va = pick(sp["train"]), pick(sp["val"])
            model, scaler, info = fit_model(arm, cfg, [data[k]["x"] for k in tr], [data[k]["y"] for k in tr],
                                            [data[k]["x"] for k in va], [data[k]["y"] for k in va],
                                            se.SEED + fold, device)
            te = [k for k in sorted(data) if data[k]["bg"] in set(sp["test"])]
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score([data[k]["y"] for k in te], predict_logits(model, [data[k]["x"] for k in te], device)))
            stored = json.load(open(os.path.join(stored_dir, "blocks-M2", f"fold{fold}.json")))["trained"]["auc"]
            ev_keys = [k for k in te if data[k]["event"] is not None]
            set_seed(se.SEED + 100 + fold)
            rand = build_model(arm, cfg).to(device).eval()
            row = {"fold": fold, "auc": auc, "stored_auc": stored, "reproduced": bool(abs(auc - stored) < 1e-6),
                   "trained": shares(model, data, ev_keys, device), "random": shares(rand, data, ev_keys, device)}
            folds.append(row)
            print(f"[{key}] fold {fold}: auc {auc:.3f} (stored {stored:.3f}) IG {row['trained']['ig_event_share']:.2f} "
                  f"vs {row['trained']['ig_event_chance']:.2f}; random {row['random']['ig_event_share']:.2f}", flush=True)
        agg = {}
        for kind in ("trained", "random"):
            agg[kind] = {m: float(np.mean([f[kind][m] for f in folds])) for m in folds[0][kind]}
        t, r = agg["trained"], agg["random"]
        crit = {}
        for meth in ("ig", "occlusion"):
            crit[meth] = {"trained_ratio": t[f"{meth}_event_share"] / t[f"{meth}_event_chance"],
                          "random_ratio": r[f"{meth}_event_share"] / r[f"{meth}_event_chance"]}
            crit[meth]["passes"] = bool(crit[meth]["trained_ratio"] >= 1.5 and crit[meth]["random_ratio"] < 1.2)
        res[key] = {"folds": folds, "mean": agg, "criterion": crit,
                    "all_reproduced": bool(all(f["reproduced"] for f in folds))}
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
