"""
PACE — pose quality control (docs/validation_criteria_phase3.md, D8).

  1. MediaPipe Pose is run again on every cohort video with the settings of
     src/preprocess.py, keeping per-landmark visibility
     (results/phase3/cache/keypoints_vis/<child>.npy, (T, 33, 3): x, y,
     visibility); the coordinates are compared with the stored keypoints.
  2. Frame-level screen: a detected frame is flagged if the hip midpoint
     moves by more than 0.5 shoulder widths from the previous detected frame,
     if the shoulder distance changes by more than 30%, or if the mean
     visibility is below 0.5; a recording is flagged if more than 5% of its
     detected frames are flagged.
  3. Overlays for visual review: 30 recordings drawn at random, the first,
     middle and last detected frame of each with the landmarks drawn
     (results/phase3/qc/overlays/, not published: the videos show children).

Writes results/phase3/pose_qc.json.

Usage:
    python scripts/pose_qc.py --raw_dir Dataset-2/Dataset --workers 5
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

CACHE = os.path.join("results", "phase3", "cache", "keypoints_vis")
OVER = os.path.join("results", "phase3", "qc", "overlays")
OUT = os.path.join("results", "phase3", "pose_qc.json")
SEED = 20261102
N_OVERLAY = 30


def extract(args):
    """(T, 33, 3) x, y, visibility for one video; zeros where undetected.
    Same MediaPipe settings as src/preprocess.py."""
    s, path, overlay = args
    import cv2
    import mediapipe as mp
    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=2, smooth_landmarks=True,
                                  enable_segmentation=False, min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(path)
    frames, images = [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.pose_landmarks:
            frames.append(np.array([[l.x, l.y, l.visibility] for l in res.pose_landmarks.landmark], np.float32))
        else:
            frames.append(np.zeros((33, 3), np.float32))
        if overlay:
            images.append(frame)
    cap.release()
    pose.close()
    kp = np.stack(frames)
    if overlay:
        det = np.flatnonzero(np.abs(kp[..., :2]).sum((1, 2)) > 0)
        if len(det):
            os.makedirs(OVER, exist_ok=True)
            conn = mp.solutions.pose.POSE_CONNECTIONS
            for tag, t in (("first", det[0]), ("middle", det[len(det) // 2]), ("last", det[-1])):
                img = images[t].copy()
                H, W = img.shape[:2]
                pts = [(int(x * W), int(y * H)) for x, y, _ in kp[t]]
                for a, b in conn:
                    cv2.line(img, pts[a], pts[b], (0, 255, 0), 3)
                for j, p in enumerate(pts):
                    # BGR: left-side landmarks (odd indices from 11 on) blue, right side and face red
                    col = (255, 0, 0) if j % 2 == 1 and j > 10 else (0, 0, 255)
                    cv2.circle(img, p, 6, col, -1)
                sc = 640 / H
                img = cv2.resize(img, (int(W * sc), 640))
                cv2.putText(img, f"{s} {tag} frame {t}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imwrite(os.path.join(OVER, f"{s}_{tag}.jpg"), img)
    return s, kp


def screen(kp_px):
    """Frame-level flags of a (T, 33, 3) array in pixel units (x, y, vis)."""
    det = np.abs(kp_px[..., :2]).sum((1, 2)) > 0
    idx = np.flatnonzero(det)
    hip = (kp_px[:, 23, :2] + kp_px[:, 24, :2]) / 2
    sw = np.linalg.norm(kp_px[:, 11, :2] - kp_px[:, 12, :2], axis=-1)
    vis = kp_px[..., 2].mean(1)
    jump = np.zeros(len(kp_px), bool); scale = np.zeros(len(kp_px), bool)
    for a, b in zip(idx[:-1], idx[1:]):
        ref = max(sw[b], 1e-6)
        jump[b] = np.linalg.norm(hip[b] - hip[a]) > 0.5 * ref
        scale[b] = abs(sw[b] - sw[a]) > 0.3 * max(sw[a], 1e-6)
    lowvis = det & (vis < 0.5)
    flag = jump | scale | lowvis
    n = max(len(idx), 1)
    return {"n_detected": int(len(idx)), "hip_jump": int(jump.sum()), "scale_jump": int(scale.sum()),
            "low_visibility": int(lowvis.sum()), "flagged": int(flag.sum()),
            "flagged_share": float(flag.sum() / n), "flagged_recording": bool(flag.sum() / n > 0.05),
            "mean_visibility": float(vis[det].mean()) if det.any() else float("nan"),
            "mean_visibility_regions": {
                r: float(kp_px[det][:, j, 2].mean()) for r, j in
                (("face", list(range(0, 11))), ("arms_hands", list(range(11, 23))),
                 ("legs_feet", list(range(25, 33))))}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default=os.environ.get("PACE_RAW_DIR", os.path.join("Dataset-2", "Dataset")))
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    part = json.load(open("splits/cv_partition.json"))
    S = part["subjects"]
    y = {s: part["labels"][s] for s in S}
    vid = pd.read_csv("processed/video_metadata.csv").set_index("clip_id")
    over = set(np.random.default_rng(SEED).choice(S, N_OVERLAY, replace=False).tolist())
    todo = [(s, os.path.join(a.raw_dir, vid.loc[s, "video"]), s in over) for s in S
            if not os.path.isfile(os.path.join(CACHE, f"{s}.npy"))]
    os.makedirs(CACHE, exist_ok=True)
    if todo:
        from multiprocessing import Pool
        with Pool(a.workers) as pool:
            for s, kp in pool.imap_unordered(extract, todo):
                np.save(os.path.join(CACHE, f"{s}.npy"), kp)
                print(f"[qc] {s} {len(kp)} frames", flush=True)
    rows = {}
    for s in S:
        kp = np.load(os.path.join(CACHE, f"{s}.npy"))
        old = np.load(os.path.join("processed", "keypoints", f"{s}.npy"))
        W, H = float(vid.loc[s, "width"]), float(vid.loc[s, "height"])
        px = kp.copy(); px[..., 0] *= W; px[..., 1] *= H
        r = screen(px)
        same_len = len(old) == len(kp)
        r["reextraction_max_abs_diff"] = float(np.abs(old - kp[..., :2]).max()) if same_len else None
        r["reextraction_same_detections"] = bool(same_len and np.array_equal(
            np.abs(old).sum((1, 2)) > 0, np.abs(kp[..., :2]).sum((1, 2)) > 0))
        r["group"] = "autistic" if y[s] == 1 else "td"
        r["uncropped"] = bool(W == 1080 and H == 1920)
        r["overlay"] = s in over
        rows[s] = r
    summ = {}
    for name, sel in (("autistic", lambda r: r["group"] == "autistic"), ("td", lambda r: r["group"] == "td"),
                      ("uncropped", lambda r: r["uncropped"]), ("cropped", lambda r: not r["uncropped"]),
                      ("all", lambda r: True)):
        R = [r for r in rows.values() if sel(r)]
        summ[name] = {"n": len(R), "flagged_recordings": int(sum(r["flagged_recording"] for r in R)),
                      "flagged_share_median": float(np.median([r["flagged_share"] for r in R])),
                      "flagged_share_max": float(np.max([r["flagged_share"] for r in R])),
                      "hip_jump_frames": int(sum(r["hip_jump"] for r in R)),
                      "scale_jump_frames": int(sum(r["scale_jump"] for r in R)),
                      "low_visibility_frames": int(sum(r["low_visibility"] for r in R)),
                      "mean_visibility_median": float(np.median([r["mean_visibility"] for r in R]))}
    diffs = [r["reextraction_max_abs_diff"] for r in rows.values() if r["reextraction_max_abs_diff"] is not None]
    out = {"summary": summ, "overlay_children": sorted(over),
           "reextraction": {"n_same_length": len(diffs), "max_abs_diff": float(max(diffs)) if diffs else None,
                            "n_same_detections": int(sum(r["reextraction_same_detections"] for r in rows.values()))},
           "per_child": rows}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_child"}, indent=1))


if __name__ == "__main__":
    main()
