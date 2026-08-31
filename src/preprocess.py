"""
PACE-ASD — Preprocessing Pipeline (Protocol Section 2)

Runs MediaPipe Pose on every raw video, applies:
  1. Mid-hip centering   (landmarks 23 + 24 mean)
  2. Inter-shoulder scale normalisation  (‖L11 – L12‖)
  3. Pad / truncate to T=300 frames
  4. Save (300, 33, 2) float32 .npy

Writes processed/labels.csv with columns:
    clip_id, subject_id, label, group

Usage:
    python src/preprocess.py --raw_dir "D:/dryad" --out_dir processed
    python src/preprocess.py --raw_dir "D:/dryad" --out_dir processed --dry_run
    python src/preprocess.py --raw_dir "D:/dryad" --out_dir processed --subjects asd_1 td_1
"""

import argparse
import os
import sys
import csv
import warnings
import numpy as np
import cv2
from tqdm import tqdm

# Suppress mediapipe/protobuf warnings
os.environ["GLOG_minloglevel"] = "2"
warnings.filterwarnings("ignore")
import mediapipe as mp

# ── Constants ─────────────────────────────────────────────────────────────────

T_MAX          = 300          # target sequence length (frames)
N_LANDMARKS    = 33           # MediaPipe Pose landmarks
LEFT_HIP       = 23
RIGHT_HIP      = 24
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12

ASD_DIR    = "Autism/children with ASD"
TD_DIR     = "Typical"
SEVERE_DIR = "Autism/Severe level of ASD"


# ── Video catalogue builder ───────────────────────────────────────────────────

def _find_regular_video(video_dir: str):
    """
    Find the primary RGB video in a subject's video/ folder.
    Priority: video.avi -> video1.avi -> first non-Svideo/Tvideo .avi found.
    Returns absolute path string or None.
    """
    for name in ("video.avi", "video1.avi"):
        p = os.path.join(video_dir, name)
        if os.path.isfile(p):
            return p
    if os.path.isdir(video_dir):
        avis = sorted([
            f for f in os.listdir(video_dir)
            if f.lower().endswith(".avi")
            and not f.lower().startswith("s")
            and not f.lower().startswith("t")
        ])
        if avis:
            return os.path.join(video_dir, avis[0])
    return None


def build_video_catalogue(raw_dir: str) -> list:
    """
    Return a list of dicts describing every raw video to process.
    Each dict: {clip_id, subject_id, label, group, video_path}

    Rules (per protocol):
      - Regular ASD/TD: one primary video per subject (video.avi or video1.avi)
      - Severe ASD (supplement): all .avi files in the case folder
    """
    catalogue = []

    def _sort_key(name):
        # Numeric names sort numerically before non-numeric names
        return (0, int(name)) if name.isdigit() else (1, name)

    # Regular ASD
    asd_base = os.path.join(raw_dir, ASD_DIR)
    for subj in sorted(os.listdir(asd_base), key=_sort_key):
        video_dir = os.path.join(asd_base, subj, "video")
        path = _find_regular_video(video_dir)
        if path:
            catalogue.append({
                "clip_id":    f"asd_{subj}",
                "subject_id": f"asd_{subj}",
                "label":      1,
                "group":      "regular",
                "video_path": path,
            })

    # Regular TD
    td_base = os.path.join(raw_dir, TD_DIR)
    for subj in sorted(os.listdir(td_base), key=_sort_key):
        if not os.path.isdir(os.path.join(td_base, subj)):
            continue
        video_dir = os.path.join(td_base, subj, "video")
        path = _find_regular_video(video_dir)
        if path:
            catalogue.append({
                "clip_id":    f"td_{subj}",
                "subject_id": f"td_{subj}",
                "label":      0,
                "group":      "regular",
                "video_path": path,
            })

    # Severe ASD (supplement)
    severe_base = os.path.join(raw_dir, SEVERE_DIR)
    for case in sorted(os.listdir(severe_base)):
        case_path = os.path.join(severe_base, case)
        if not os.path.isdir(case_path):
            continue
        avis = sorted(f for f in os.listdir(case_path) if f.lower().endswith(".avi"))
        for i, avi in enumerate(avis, start=1):
            catalogue.append({
                "clip_id":    f"severe_{case}_v{i}",
                "subject_id": f"severe_{case}",
                "label":      1,
                "group":      "supplement",
                "video_path": os.path.join(case_path, avi),
            })

    return catalogue


# ── MediaPipe extraction ──────────────────────────────────────────────────────

def extract_keypoints_from_video(video_path: str) -> np.ndarray:
    """
    Run MediaPipe Pose on every frame of a video.

    Returns:
        keypoints: (actual_frame_count, 33, 2) float32
                   Zero rows where MediaPipe failed to detect a pose.
    """
    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        if results.pose_landmarks:
            kp = np.array(
                [[lm.x, lm.y] for lm in results.pose_landmarks.landmark],
                dtype=np.float32,
            )  # (33, 2)
        else:
            kp = np.zeros((N_LANDMARKS, 2), dtype=np.float32)
        frames.append(kp)

    cap.release()
    pose.close()

    if not frames:
        return np.zeros((1, N_LANDMARKS, 2), dtype=np.float32)
    return np.stack(frames, axis=0)  # (T_actual, 33, 2)


# ── Normalisation ─────────────────────────────────────────────────────────────

def normalise(keypoints: np.ndarray) -> np.ndarray:
    """
    Apply centering and scale normalisation per frame.

    1. Centering: subtract mid-hip = mean(L23, L24) per frame
    2. Scale:     divide by inter-shoulder distance ‖L11 – L12‖, clamped ≥ 1e-5

    Input / output: (T, 33, 2) float32
    """
    kp = keypoints.copy()
    T  = kp.shape[0]

    for t in range(T):
        # Only normalise frames where at least one landmark is non-zero
        if not np.any(kp[t] != 0):
            continue

        # 1. Mid-hip centering
        mid_hip = (kp[t, LEFT_HIP] + kp[t, RIGHT_HIP]) / 2.0
        kp[t]   = kp[t] - mid_hip

        # 2. Inter-shoulder scale
        shoulder_dist = float(np.linalg.norm(kp[t, LEFT_SHOULDER] - kp[t, RIGHT_SHOULDER]))
        shoulder_dist = max(shoulder_dist, 1e-5)
        kp[t]         = kp[t] / shoulder_dist

    return kp


# ── Padding / truncation ──────────────────────────────────────────────────────

def pad_or_truncate(keypoints: np.ndarray, target_len: int = T_MAX) -> np.ndarray:
    """
    Pad (with zeros at end) or truncate to exactly target_len frames.

    Input / output: (*, 33, 2) → (target_len, 33, 2)
    """
    T = keypoints.shape[0]
    if T >= target_len:
        return keypoints[:target_len]
    pad = np.zeros((target_len - T, N_LANDMARKS, 2), dtype=np.float32)
    return np.concatenate([keypoints, pad], axis=0)


# ── Single-clip processor ─────────────────────────────────────────────────────

def process_video(video_path: str) -> np.ndarray:
    """
    Full preprocessing pipeline for one video.

    Returns (300, 33, 2) float32 — ready to save as .npy.
    """
    raw   = extract_keypoints_from_video(video_path)   # (T_actual, 33, 2)
    normd = normalise(raw)                             # (T_actual, 33, 2)
    final = pad_or_truncate(normd)                     # (300, 33, 2)
    return final


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PACE-ASD preprocessing")
    parser.add_argument("--raw_dir",  default="D:/dryad")
    parser.add_argument("--out_dir",  default="processed")
    parser.add_argument("--dry_run",  action="store_true",
                        help="Scan and report what would be processed; write nothing.")
    parser.add_argument("--subjects", nargs="*", default=None,
                        help="Limit to specific clip_ids (e.g. asd_1 td_3).")
    args = parser.parse_args()

    raw_dir     = os.path.abspath(args.raw_dir)
    out_dir     = os.path.abspath(args.out_dir)
    features_dir = os.path.join(out_dir, "features")

    catalogue = build_video_catalogue(raw_dir)

    # Filter to requested subjects
    if args.subjects:
        subject_set = set(args.subjects)
        catalogue   = [c for c in catalogue if c["clip_id"] in subject_set]

    print(f"\nPACE-ASD Preprocessor")
    print(f"  Raw dir    : {raw_dir}")
    print(f"  Output dir : {out_dir}")
    print(f"  Videos     : {len(catalogue)} to process")
    print(f"  Dry run    : {args.dry_run}\n")

    if args.dry_run:
        for entry in catalogue:
            status = "EXISTS" if os.path.isfile(
                os.path.join(features_dir, f"{entry['clip_id']}.npy")
            ) else "PENDING"
            print(f"  [{status}] {entry['clip_id']:30s}  {entry['video_path']}")
        print(f"\nTotal: {len(catalogue)} clips")
        return

    os.makedirs(features_dir, exist_ok=True)

    # Labels list (built from catalogue; clip_ids already processed are included)
    # We collect all metadata upfront so labels.csv is always complete.
    labels_rows = []
    skipped     = 0
    processed   = 0
    failed      = 0

    for entry in tqdm(catalogue, desc="Preprocessing", unit="clip"):
        npy_path = os.path.join(features_dir, f"{entry['clip_id']}.npy")
        labels_rows.append({
            "clip_id":    entry["clip_id"],
            "subject_id": entry["subject_id"],
            "label":      entry["label"],
            "group":      entry["group"],
        })

        if os.path.isfile(npy_path):
            skipped += 1
            continue

        try:
            arr = process_video(entry["video_path"])  # (300, 33, 2)
            np.save(npy_path, arr)
            processed += 1
        except Exception as exc:
            print(f"\n  [ERROR] {entry['clip_id']}: {exc}")
            # Save a zero array so the rest of the pipeline doesn't break
            np.save(npy_path, np.zeros((T_MAX, N_LANDMARKS, 2), dtype=np.float32))
            failed += 1

    # Write labels.csv
    labels_path = os.path.join(out_dir, "labels.csv")
    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["clip_id", "subject_id", "label", "group"]
        )
        writer.writeheader()
        writer.writerows(labels_rows)

    print(f"\nDone.")
    print(f"  Processed : {processed}")
    print(f"  Skipped   : {skipped} (already existed)")
    print(f"  Failed    : {failed}")
    print(f"  Labels CSV: {labels_path}")
    print(f"  Features  : {features_dir}/")

    if failed > 0:
        print(f"\n  [WARN] {failed} clip(s) failed — zero arrays saved. "
              "Check video files and MediaPipe installation.")


if __name__ == "__main__":
    main()
