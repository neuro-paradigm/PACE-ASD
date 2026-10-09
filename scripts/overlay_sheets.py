"""
PACE — overlay contact sheets of every cohort recording for the visual part of
the recording audit (exploratory extension of D8 in
docs/validation_criteria_phase3.md, which samples 30 recordings).

For each child, the first, middle and last detected frame of the color video
with the landmarks of the MediaPipe re-extraction (results/phase3/cache/
keypoints_vis) drawn on it; six children per sheet, in random order so that
the group is not apparent from the position (the identifier is printed).
Output: results/phase3/qc/all_sheets/ (not published: the videos show
children). Annotations of the sheets are kept in results/phase3/qc/visual_audit.csv.

Usage:
    python scripts/overlay_sheets.py --raw_dir Dataset-2/Dataset
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
OUTD = os.path.join("results", "phase3", "qc", "all_sheets")


def frames(path, idx):
    import cv2
    cap = cv2.VideoCapture(path)
    out, t = {}, 0
    want = set(idx)
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if t in want:
            out[t] = f
        t += 1
    cap.release()
    return out


def draw(img, kp):
    import cv2
    import mediapipe as mp
    H, W = img.shape[:2]
    pts = [(int(x * W), int(y * H)) for x, y, _ in kp]
    for a, b in mp.solutions.pose.POSE_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], (0, 255, 0), 3)
    for j, p in enumerate(pts):
        cv2.circle(img, p, 6, (255, 0, 0) if j % 2 == 1 and j > 10 else (0, 0, 255), -1)
    return img


def main():
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default=os.environ.get("PACE_RAW_DIR", os.path.join("Dataset-2", "Dataset")))
    a = ap.parse_args()
    S = json.load(open("splits/cv_partition.json"))["subjects"]
    vid = pd.read_csv("processed/video_metadata.csv").set_index("clip_id")
    order = list(np.random.default_rng(20261109).permutation(S))
    os.makedirs(OUTD, exist_ok=True)
    rows = []
    for s in order:
        kp = np.load(os.path.join("results", "phase3", "cache", "keypoints_vis", f"{s}.npy"))
        det = np.flatnonzero(np.abs(kp[..., :2]).sum((1, 2)) > 0)
        idx = [int(det[0]), int(det[len(det) // 2]), int(det[-1])]
        fr = frames(os.path.join(a.raw_dir, vid.loc[s, "video"]), idx)
        ims = []
        for t in idx:
            im = draw(fr[t].copy(), kp[t])
            im = cv2.resize(im, (int(im.shape[1] * 300 / im.shape[0]), 300))
            cv2.putText(im, f"{s} f{t}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            ims.append(im)
        rows.append(np.concatenate(ims, 1))
    for i in range(0, len(rows), 6):
        part = rows[i:i + 6]
        W = max(r.shape[1] for r in part)
        part = [np.pad(r, ((0, 0), (0, W - r.shape[1]), (0, 0)), constant_values=255) for r in part]
        half = (len(part) + 1) // 2
        left = np.concatenate(part[:half], 0)
        right = np.concatenate(part[half:], 0) if part[half:] else np.full_like(left, 255)
        if right.shape[0] < left.shape[0]:
            right = np.pad(right, ((0, left.shape[0] - right.shape[0]), (0, 0), (0, 0)), constant_values=255)
        sheet = np.concatenate([left, np.full((left.shape[0], 10, 3), 255, np.uint8), right], 1)
        cv2.imwrite(os.path.join(OUTD, f"sheet{i // 6:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
        print(i // 6, order[i:i + 6], flush=True)


if __name__ == "__main__":
    main()
