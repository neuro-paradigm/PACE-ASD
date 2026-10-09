"""
PACE — planted recording differences for the negative controls
(docs/validation_criteria_phase4.md, E4).

The color video of every typically developing child is transformed frame by
frame and MediaPipe Pose is run on the transformed frames with the settings
of src/preprocess.py (keeping visibility); the image statistics of the scene
control are computed from the same transformed frames.

  n1   image: every channel scaled by 0.75 (darker, lower contrast), then
       saturation multiplied by 1.4 in HSV
  n2   framing: the central 80% of the frame in width and height (a tighter
       crop; the frame size changes)

The bit rate is that of the original file (the frames are not re-encoded).
Outputs (not published: derived from videos of children):
results/phase4/cache/keypoints_<v>/<child>.npy and videos_<v>.json.

Usage:
    python scripts/negative_control_videos.py --variant n1 --workers 4
    python scripts/negative_control_videos.py --summarize
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

CACHE = os.path.join("results", "phase4", "cache")


def transform(frame, variant):
    import cv2
    if variant == "n1":
        f = np.clip(frame.astype(np.float32) * 0.75, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * 1.4, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if variant == "n2":
        H, W = frame.shape[:2]
        h, w = int(round(0.8 * H)), int(round(0.8 * W))
        y0, x0 = (H - h) // 2, (W - w) // 2
        return np.ascontiguousarray(frame[y0:y0 + h, x0:x0 + w])
    raise ValueError(variant)


def one(args):
    """Landmarks (x, y, visibility) and scene statistics of one transformed video."""
    s, path, variant, fps, step = args
    import cv2
    import mediapipe as mp
    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=2, smooth_landmarks=True,
                                  enable_segmentation=False, min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(path)
    kps, stats = [], {k: [] for k in ("lum", "con", "sat", "sharp", "border", "face", "shoulder")}
    t, W, H = 0, None, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fr = transform(frame, variant)
        H, W = fr.shape[:2]
        res = pose.process(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        kp = (np.array([[l.x, l.y, l.visibility] for l in res.pose_landmarks.landmark], np.float32)
              if res.pose_landmarks else np.zeros((33, 3), np.float32))
        kps.append(kp)
        if t % step == 0:
            # the statistics of recording_features.scene_features, on the transformed frame
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
            stats["lum"].append(gray.mean()); stats["con"].append(gray.std()); stats["sat"].append(hsv[..., 1].mean())
            stats["sharp"].append(cv2.Laplacian(gray, cv2.CV_32F).var())
            bw, bh = max(1, int(0.05 * W)), max(1, int(0.05 * H))
            band = np.concatenate([gray[:bh].ravel(), gray[-bh:].ravel(), gray[:, :bw].ravel(), gray[:, -bw:].ravel()])
            stats["border"].append(band.mean())
            if np.any(kp[:, :2] != 0):
                p = kp[:, :2] * np.array([W, H])
                sw = np.linalg.norm(p[11] - p[12])
                stats["shoulder"].append(sw / H)
                r = max(4, int(0.5 * sw))
                cx, cy = int(p[0, 0]), int(p[0, 1])
                box = gray[max(0, cy - r):min(H, cy + r), max(0, cx - r):min(W, cx + r)]
                if box.size > 16:
                    stats["face"].append(cv2.Laplacian(box, cv2.CV_32F).var())
        t += 1
    cap.release()
    pose.close()
    seconds = t / fps
    scene = {"luminance": float(np.mean(stats["lum"])), "contrast": float(np.mean(stats["con"])),
             "saturation": float(np.mean(stats["sat"])), "sharpness": float(np.mean(stats["sharp"])),
             "border_luminance": float(np.mean(stats["border"])),
             "face_sharpness": float(np.median(stats["face"])) if stats["face"] else float("nan"),
             "bitrate_kbps": float(os.path.getsize(path) * 8 / 1000 / seconds),
             "shoulder_px_rel_height": float(np.median(stats["shoulder"])) if stats["shoulder"] else float("nan")}
    return s, np.stack(kps), {"width": int(W), "height": int(H), "frames": t, "scene": scene}


def summarize():
    """How the transformations changed the pose estimates and the image
    statistics -> results/phase4/negative_control_videos.json."""
    from recording_features import SCENE_NAMES
    vid = pd.read_csv("processed/video_metadata.csv").set_index("clip_id")
    base = json.load(open(os.path.join("results", "cv", "scene_features.json")))
    out = {}
    for v in ("n1", "n2"):
        meta = json.load(open(os.path.join(CACHE, f"videos_{v}.json")))
        d, diff = [], []
        for s, m in meta.items():
            kp = np.load(os.path.join(CACHE, f"keypoints_{v}", f"{s}.npy"))
            det = np.abs(kp[..., :2]).sum((1, 2)) > 0
            d.append(int(det.sum()) - int(vid.loc[s, "frames_detected"]))
            orig = np.load(os.path.join("processed", "keypoints", f"{s}.npy"))
            both = det[:len(orig)] & (np.abs(orig).sum((1, 2)) > 0)[:len(det)]
            if both.any():
                # landmark displacement in pixels of the original frame (crop offset removed)
                W0, H0 = float(vid.loc[s, "width"]), float(vid.loc[s, "height"])
                W1, H1 = float(m["width"]), float(m["height"])
                p0 = orig[:len(both)][both] * [W0, H0]
                p1 = kp[:len(both)][both][..., :2] * [W1, H1] + [(W0 - W1) // 2, (H0 - H1) // 2]
                sw = np.linalg.norm(p0[:, 11] - p0[:, 12], axis=-1)[:, None]
                diff.append(float(np.median(np.linalg.norm(p1 - p0, axis=-1) / np.maximum(sw, 1e-6))))
        d = np.array(d)
        sc = {k: float(np.median([meta[s]["scene"][k] - base[s][k] for s in meta])) for k in SCENE_NAMES}
        out[v] = {"n": len(meta), "n_detection_changed": int((d != 0).sum()), "detected_change_median": float(np.median(d)),
                  "detected_change_range": [int(d.min()), int(d.max())],
                  "landmark_shift_median_shoulder_widths": float(np.median(diff)) if diff else None,
                  "scene_change_median": sc}
    json.dump(out, open(os.path.join("results", "phase4", "negative_control_videos.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["n1", "n2"])
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--raw_dir", default=os.environ.get("PACE_RAW_DIR", os.path.join("Dataset-2", "Dataset")))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.summarize:
        summarize()
        return
    p = json.load(open("splits/cv_partition.json"))
    td = sorted(s for s in p["subjects"] if p["labels"][s] == 0)
    vid = pd.read_csv("processed/video_metadata.csv").set_index("clip_id")
    kd = os.path.join(CACHE, f"keypoints_{a.variant}")
    os.makedirs(kd, exist_ok=True)
    mpath = os.path.join(CACHE, f"videos_{a.variant}.json")
    meta = json.load(open(mpath)) if os.path.isfile(mpath) else {}
    todo = [(s, os.path.join(a.raw_dir, vid.loc[s, "video"]), a.variant, float(vid.loc[s, "fps"]), 10)
            for s in td if s not in meta]
    from multiprocessing import Pool
    with Pool(a.workers) as pool:
        for s, kp, m in pool.imap_unordered(one, todo):
            np.save(os.path.join(kd, f"{s}.npy"), kp)
            meta[s] = m
            json.dump(meta, open(mpath, "w"), indent=1)
            print(f"[{a.variant}] {s} {m['width']}x{m['height']} frames {m['frames']} "
                  f"detected {(np.abs(kp[..., :2]).sum((1, 2)) > 0).sum()}", flush=True)


if __name__ == "__main__":
    main()
