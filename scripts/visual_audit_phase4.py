"""
PACE — the visual part of the recording audit, fourth phase
(docs/validation_criteria_phase4.md, E7; codebook in docs/annotation_codebook.md).

  --approach    objective cross-check of "walks towards the camera": the
                child's apparent size (shoulder-midpoint to hip-midpoint
                distance in pixels) in the last against the first quarter of
                the detected frames; compared with the annotations
                -> results/phase4/approach_index.json
  --kit         blinded annotation material for two human annotators: one
                sheet per recording (eight overlays evenly spaced over the
                detected frames and the first detected frame without
                overlay), identified only by a random code; a contact sheet
                of the first frames for grouping recordings into settings;
                an empty annotation file per annotator; the code key
                -> results/phase4/annotation/ (not published: shows children)
  --agreement   agreement of the two annotators (Cohen's kappa with bootstrap
                intervals; adjusted Rand index of the settings), of the
                consensus with the AI-assisted annotation and with the
                approach index -> results/phase4/annotation_agreement.json

Usage:
    python scripts/visual_audit_phase4.py --approach
    python scripts/visual_audit_phase4.py --kit --raw_dir Dataset-2/Dataset
    python scripts/visual_audit_phase4.py --agreement
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

OUT4 = os.path.join("results", "phase4")
ANN = os.path.join(OUT4, "annotation")
AI_CSV = os.path.join("results", "phase3", "qc", "visual_audit.csv")
ITEMS = ["walks_toward_camera", "other_person_visible", "gross_tracking_error"]
APPROACH_THRESHOLD = float(np.log(1.2))
KIT_SEED = 20261401
B = 2000


def cohort():
    p = json.load(open("splits/cv_partition.json"))
    return p["subjects"], {s: p["labels"][s] for s in p["subjects"]}


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = np.mean(a == b)
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def kappa_ci(a, b, seed=20261402):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    rng = np.random.default_rng(seed)
    bs = [kappa(a[i], b[i]) for i in (rng.integers(0, len(a), len(a)) for _ in range(B))]
    return {"kappa": kappa(a, b), "ci95": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))],
            "agreement": float(np.mean(a == b)), "n": int(len(a)), "n_yes_a": int(a.sum()), "n_yes_b": int(b.sum())}


# ── objective cross-check ─────────────────────────────────────────────────────

def approach_index(s, vid):
    kp = np.load(os.path.join("processed", "keypoints", f"{s}.npy")).astype(float)
    det = np.flatnonzero(np.abs(kp).sum((1, 2)) > 0)
    W, H = float(vid.loc[s, "width"]), float(vid.loc[s, "height"])
    p = kp[det] * np.array([W, H])
    torso = np.linalg.norm((p[:, 11] + p[:, 12]) / 2 - (p[:, 23] + p[:, 24]) / 2, axis=-1)
    q = max(1, len(det) // 4)
    return float(np.log(np.median(torso[-q:]) / np.median(torso[:q])))


def yes(v):
    return str(v).strip().lower() in ("yes", "y", "1", "true")


def approach():
    from sklearn.metrics import roc_auc_score
    S, lab = cohort()
    vid = pd.read_csv("processed/video_metadata.csv").set_index("clip_id")
    idx = {s: approach_index(s, vid) for s in S}
    out = {"threshold_log_ratio": APPROACH_THRESHOLD, "index": idx,
           "by_group": {g: {"median": float(np.median([idx[s] for s in S if lab[s] == l])),
                            "n_approaching": int(sum(idx[s] >= APPROACH_THRESHOLD for s in S if lab[s] == l)),
                            "n": int(sum(lab[s] == l for s in S))}
                        for g, l in (("autistic", 1), ("td", 0))}}
    refs = {"ai": AI_CSV}
    cons = consensus_by_child()
    if cons is not None:
        refs["consensus"] = cons
    for name, src in refs.items():
        a = pd.read_csv(src).set_index("child") if isinstance(src, str) else src
        w = np.array([yes(a.loc[s, "walks_toward_camera"]) for s in S])
        x = np.array([idx[s] for s in S])
        out[f"vs_{name}"] = {"auc": float(roc_auc_score(w, x)), **kappa_ci(w, x >= APPROACH_THRESHOLD),
                             "median_walk": float(np.median(x[w])), "median_not_walk": float(np.median(x[~w])),
                             "within_autistic": kappa_ci(w[np.array([lab[s] == 1 for s in S])],
                                                         (x >= APPROACH_THRESHOLD)[np.array([lab[s] == 1 for s in S])])}
    os.makedirs(OUT4, exist_ok=True)
    json.dump(out, open(os.path.join(OUT4, "approach_index.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "index"}, indent=1))
    return out


# ── annotation kit ────────────────────────────────────────────────────────────

def kit(raw_dir):
    import cv2
    import mediapipe as mp
    S, lab = cohort()
    vid = pd.read_csv("processed/video_metadata.csv").set_index("clip_id")
    perm = np.random.default_rng(KIT_SEED).permutation(len(S))
    codes = {S[i]: f"R{n + 1:02d}" for n, i in enumerate(perm)}
    os.makedirs(os.path.join(ANN, "sheets"), exist_ok=True)
    conn = mp.solutions.pose.POSE_CONNECTIONS
    firsts = {}
    for s in S:
        kp = np.load(os.path.join("results", "phase3", "cache", "keypoints_vis", f"{s}.npy"))
        det = np.flatnonzero(np.abs(kp[..., :2]).sum((1, 2)) > 0)
        want = sorted({int(det[int(round(q))]) for q in np.linspace(0, len(det) - 1, 8)})
        cap = cv2.VideoCapture(os.path.join(raw_dir, vid.loc[s, "video"]))
        frames, t = {}, 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if t in want:
                frames[t] = f
            t += 1
        cap.release()
        tiles = []
        f0 = frames[want[0]]
        h = 360
        sc = lambda im: cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h))
        tiles.append(sc(f0.copy()))
        firsts[codes[s]] = sc(f0.copy())
        for n, t in enumerate(want):
            img = frames[t].copy()
            H, W = img.shape[:2]
            pts = [(int(x * W), int(y * H)) for x, y, _ in kp[t]]
            for a, b in conn:
                cv2.line(img, pts[a], pts[b], (0, 255, 0), 3)
            for j, p in enumerate(pts):
                cv2.circle(img, p, 6, (255, 0, 0) if j % 2 == 1 and j > 10 else (0, 0, 255), -1)
            img = sc(img)
            cv2.putText(img, f"{n + 1}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            tiles.append(img)
        row1, row2 = np.concatenate(tiles[:5], 1), np.concatenate(tiles[5:], 1)
        wmax = max(row1.shape[1], row2.shape[1])
        pad = lambda r: np.concatenate([r, np.zeros((h, wmax - r.shape[1], 3), np.uint8)], 1)
        sheet = np.concatenate([pad(row1), pad(row2)], 0)
        cv2.putText(sheet, codes[s], (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.imwrite(os.path.join(ANN, "sheets", f"{codes[s]}.jpg"), sheet)
        print(f"[kit] {codes[s]}", flush=True)
    # contact sheet of first frames for the grouping into settings, 10 per row, in code order
    order = sorted(firsts)
    thumbs = []
    for c in order:
        im = cv2.resize(firsts[c], (int(firsts[c].shape[1] * 240 / firsts[c].shape[0]), 240))
        im = np.concatenate([im, np.zeros((240, 150 - im.shape[1] % 150 if im.shape[1] < 150 else 0, 3), np.uint8)], 1) \
            if im.shape[1] < 150 else im
        im = cv2.resize(im, (150, 240))
        cv2.putText(im, c, (5, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        thumbs.append(im)
    rows = [np.concatenate(thumbs[i:i + 10] + [np.zeros((240, 150, 3), np.uint8)] * (10 - len(thumbs[i:i + 10])), 1)
            for i in range(0, len(thumbs), 10)]
    cv2.imwrite(os.path.join(ANN, "settings_contact_sheet.jpg"), np.concatenate(rows, 0))
    with open(os.path.join(ANN, "key_do_not_open_before_consensus.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["code", "child"])
        for s in S:
            w.writerow([codes[s], s])
    for who in ("A", "B"):
        path = os.path.join(ANN, f"annotator_{who}.csv")
        if os.path.isfile(path):
            continue
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["code"] + ITEMS + ["setting", "note"])
            for c in order:
                w.writerow([c, "", "", "", "", ""])
    print(f"[kit] {len(S)} sheets in {ANN}")


# ── agreement ─────────────────────────────────────────────────────────────────

def read_annotation(path):
    d = pd.read_csv(path, dtype=str).fillna("")
    d["code"] = d["code"].str.strip()
    return d.set_index("code")


def consensus_by_child():
    """The consensus annotation indexed by child, or None if it does not exist."""
    cpath, kpath = os.path.join(ANN, "consensus.csv"), os.path.join(ANN, "key_do_not_open_before_consensus.csv")
    if not (os.path.isfile(cpath) and os.path.isfile(kpath)):
        return None
    c = read_annotation(cpath)
    key = pd.read_csv(kpath).set_index("code")["child"]
    if (c[ITEMS + ["setting"]] == "").any().any():
        return None
    c = c.assign(child=[key[k] for k in c.index]).set_index("child")
    return c


def agreement():
    from sklearn.metrics import adjusted_rand_score
    S, lab = cohort()
    paths = {w: os.path.join(ANN, f"annotator_{w}.csv") for w in ("A", "B")}
    A, Bn = read_annotation(paths["A"]), read_annotation(paths["B"])
    if (A[ITEMS + ["setting"]] == "").any().any() or (Bn[ITEMS + ["setting"]] == "").any().any():
        raise SystemExit("annotation files are incomplete")
    key = pd.read_csv(os.path.join(ANN, "key_do_not_open_before_consensus.csv")).set_index("code")["child"]
    codes = sorted(A.index)
    out = {"items": {}}
    for it in ITEMS:
        out["items"][it] = kappa_ci([yes(A.loc[c, it]) for c in codes], [yes(Bn.loc[c, it]) for c in codes])
    out["settings"] = {"adjusted_rand_index": float(adjusted_rand_score([A.loc[c, "setting"] for c in codes],
                                                                         [Bn.loc[c, "setting"] for c in codes])),
                       "n_settings_A": int(A["setting"].nunique()), "n_settings_B": int(Bn["setting"].nunique())}
    for who, D in (("A", A), ("B", Bn)):
        grp = {c: lab[key[c]] for c in codes}
        shared = sorted({D.loc[c, "setting"] for c in codes if grp[c] == 1} & {D.loc[c, "setting"] for c in codes if grp[c] == 0})
        out["settings"][f"shared_between_groups_{who}"] = shared
    cons = consensus_by_child()
    if cons is not None:
        ai = pd.read_csv(AI_CSV).set_index("child")
        out["consensus_vs_ai"] = {it: kappa_ci([yes(cons.loc[s, it]) for s in S], [yes(ai.loc[s, it]) for s in S])
                                  for it in ITEMS}
        out["consensus_vs_ai"]["settings_ari"] = float(adjusted_rand_score([cons.loc[s, "setting"] for s in S],
                                                                           [ai.loc[s, "setting"] for s in S]))
        out["criterion_kappa_ge_0.6"] = {it: out["items"][it]["kappa"] >= 0.6 for it in ITEMS}
    os.makedirs(OUT4, exist_ok=True)
    json.dump(out, open(os.path.join(OUT4, "annotation_agreement.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--approach", action="store_true")
    ap.add_argument("--kit", action="store_true")
    ap.add_argument("--agreement", action="store_true")
    ap.add_argument("--raw_dir", default=os.environ.get("PACE_RAW_DIR", os.path.join("Dataset-2", "Dataset")))
    a = ap.parse_args()
    if a.approach:
        approach()
    if a.kit:
        kit(a.raw_dir)
    if a.agreement:
        agreement()
