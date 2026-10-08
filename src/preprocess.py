"""
PACE-ASD — Preprocessing (video -> landmark arrays)

For every video:
  1. MediaPipe Pose on every frame -> raw keypoints (T_video, 33, 2): image
     coordinates normalised by frame width and height, zero where no pose
     was detected. Saved to <out_dir>/keypoints/<clip_id>.npy, with the frame
     rate, frame size, frame count and extraction time written to
     <out_dir>/video_metadata.csv.
  2. Conversion to pixel units (x * width, y * height). MediaPipe normalises
     the two axes by different lengths, so without this step body geometry is
     stretched by each video's aspect ratio, which varies with how the video
     was cropped.
  3. Mid-hip centering and division by the shoulder distance.
  4. Truncation or zero-padding at the end to T = 300 frames.
  5. Saved as <out_dir>/features/<clip_id>.npy, (300, 33, 2) float32.

Usage:
    python src/preprocess.py --raw_dir data/raw/Dataset --out_dir processed --workers 6
    python src/preprocess.py --raw_dir data/raw/Dataset --dry_run
    python src/preprocess.py --raw_dir data/raw/Dataset --out_dir processed --compare_to old/features
"""

import argparse
import csv
import os
import time
import warnings

import numpy as np

os.environ["GLOG_minloglevel"] = "2"
warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────

T_MAX          = 300          # frames per stored sequence
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

      - autistic and typically developing children: one colour video each
        (video.avi or video1.avi)
      - children with severe autism: every .avi file in the case folder
    """
    catalogue = []

    def _sort_key(name):
        return (0, int(name)) if name.isdigit() else (1, name)

    asd_base = os.path.join(raw_dir, ASD_DIR)
    for subj in sorted(os.listdir(asd_base), key=_sort_key):
        path = _find_regular_video(os.path.join(asd_base, subj, "video"))
        if path:
            catalogue.append({"clip_id": f"asd_{subj}", "subject_id": f"asd_{subj}",
                              "label": 1, "group": "regular", "video_path": path})

    td_base = os.path.join(raw_dir, TD_DIR)
    for subj in sorted(os.listdir(td_base), key=_sort_key):
        if not os.path.isdir(os.path.join(td_base, subj)):
            continue
        path = _find_regular_video(os.path.join(td_base, subj, "video"))
        if path:
            catalogue.append({"clip_id": f"td_{subj}", "subject_id": f"td_{subj}",
                              "label": 0, "group": "regular", "video_path": path})

    severe_base = os.path.join(raw_dir, SEVERE_DIR)
    for case in sorted(os.listdir(severe_base)):
        case_path = os.path.join(severe_base, case)
        if not os.path.isdir(case_path):
            continue
        avis = sorted(f for f in os.listdir(case_path) if f.lower().endswith(".avi"))
        for i, avi in enumerate(avis, start=1):
            catalogue.append({"clip_id": f"severe_{case}_v{i}", "subject_id": f"severe_{case}",
                              "label": 1, "group": "supplement",
                              "video_path": os.path.join(case_path, avi)})
    return catalogue


# ── MediaPipe extraction ──────────────────────────────────────────────────────

def extract_keypoints_from_video(video_path: str, return_meta: bool = False,
                                 model_complexity: int = 2, smooth_landmarks: bool = True,
                                 min_detection_confidence: float = 0.5,
                                 min_tracking_confidence: float = 0.5):
    """
    Run MediaPipe Pose on every frame of a video.

    Returns:
        keypoints: (n_frames, 33, 2) float32 image-normalised (x, y); zero rows
                   where no pose was detected.
        meta (if return_meta): frame rate, frame size, frames read, frames
                   with a detection, and decoding + pose-estimation time.
    """
    import cv2
    import mediapipe as mp
    pose = mp.solutions.pose.Pose(
        static_image_mode=False, model_complexity=model_complexity,
        smooth_landmarks=smooth_landmarks, enable_segmentation=False,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    meta = {"fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frame_count_header": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
    t0 = time.perf_counter()
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.pose_landmarks:
            frames.append(np.array([[lm.x, lm.y] for lm in res.pose_landmarks.landmark],
                                   dtype=np.float32))
        else:
            frames.append(np.zeros((N_LANDMARKS, 2), dtype=np.float32))
    cap.release()
    pose.close()
    meta["seconds"] = round(time.perf_counter() - t0, 3)
    meta["frames_read"] = len(frames)
    out = np.stack(frames) if frames else np.zeros((1, N_LANDMARKS, 2), np.float32)
    meta["frames_detected"] = int((np.abs(out).sum(axis=(1, 2)) > 0).sum())
    return (out, meta) if return_meta else out


# ── Normalisation ─────────────────────────────────────────────────────────────

def normalise(keypoints: np.ndarray, width: float | None = None,
              height: float | None = None) -> np.ndarray:
    """
    Per frame with a detection: convert to pixel units (when the frame size is
    given), subtract the mid-hip point and divide by the shoulder distance
    (bounded below by 1e-5). Frames without a detection stay zero.

    Without width and height the image-normalised coordinates are used
    directly, which stretches body geometry by the frame's aspect ratio; this
    form is kept only for comparison with arrays produced that way.
    """
    kp = keypoints.astype(np.float64).copy()
    if width is not None and height is not None:
        kp[..., 0] *= width
        kp[..., 1] *= height
    valid = np.any(keypoints != 0, axis=(1, 2))
    mid_hip = (kp[:, LEFT_HIP] + kp[:, RIGHT_HIP]) / 2.0
    kp = kp - mid_hip[:, None, :]
    sd = np.maximum(np.linalg.norm(kp[:, LEFT_SHOULDER] - kp[:, RIGHT_SHOULDER], axis=-1), 1e-5)
    kp = kp / sd[:, None, None]
    kp[~valid] = 0.0
    return kp.astype(np.float32)


def pad_or_truncate(keypoints: np.ndarray, target_len: int = T_MAX) -> np.ndarray:
    """Zero-pad at the end or truncate to exactly target_len frames."""
    T = keypoints.shape[0]
    if T >= target_len:
        return keypoints[:target_len].astype(np.float32)
    pad = np.zeros((target_len - T, N_LANDMARKS, 2), dtype=np.float32)
    return np.concatenate([keypoints.astype(np.float32), pad], axis=0)


def process_video(video_path: str, return_meta: bool = False):
    """Video -> (300, 33, 2) pixel-normalised array (and metadata)."""
    raw, meta = extract_keypoints_from_video(video_path, return_meta=True)
    final = pad_or_truncate(normalise(raw, meta["width"], meta["height"]))
    return (final, meta) if return_meta else final


# ── Batch processing ──────────────────────────────────────────────────────────

def _work(entry):
    raw, meta = extract_keypoints_from_video(entry["video_path"], return_meta=True)
    return entry, raw, meta


def main():
    ap = argparse.ArgumentParser(description="PACE-ASD preprocessing")
    ap.add_argument("--raw_dir", default="data/raw/Dataset")
    ap.add_argument("--out_dir", default="processed")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--subjects", nargs="*", default=None, help="limit to these clip_ids")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--compare_to", default=None,
                    help="directory of arrays made from image-normalised coordinates; each is "
                         "compared with the same normalisation of the new keypoints")
    args = ap.parse_args()

    raw_dir, out_dir = os.path.abspath(args.raw_dir), os.path.abspath(args.out_dir)
    catalogue = build_video_catalogue(raw_dir)
    if args.subjects:
        catalogue = [c for c in catalogue if c["clip_id"] in set(args.subjects)]
    print(f"{len(catalogue)} videos under {raw_dir}")
    if args.dry_run:
        for e in catalogue:
            print(f"  {e['clip_id']:22s} {os.path.relpath(e['video_path'], raw_dir)}")
        return

    for sub in ("features", "keypoints"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    t0 = time.perf_counter()
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            results = list(pool.imap(_work, catalogue))
    else:
        results = [_work(e) for e in catalogue]
    wall = time.perf_counter() - t0

    rows = []
    for entry, raw, meta in results:
        cid = entry["clip_id"]
        np.save(os.path.join(out_dir, "keypoints", f"{cid}.npy"), raw)
        arr = pad_or_truncate(normalise(raw, meta["width"], meta["height"]))
        np.save(os.path.join(out_dir, "features", f"{cid}.npy"), arr)
        row = {"clip_id": cid, "subject_id": entry["subject_id"], "label": entry["label"],
               "video": os.path.relpath(entry["video_path"], raw_dir).replace(os.sep, "/"),
               **meta}
        if args.compare_to:
            ref_path = os.path.join(args.compare_to, f"{cid}.npy")
            if os.path.isfile(ref_path):
                ref = np.load(ref_path)
                legacy = pad_or_truncate(normalise(raw))
                row["legacy_identical"] = bool(np.array_equal(legacy, ref))
                row["legacy_max_abs_diff"] = float(np.abs(legacy - ref).max())
                row["legacy_mask_equal"] = bool(np.array_equal(
                    np.abs(legacy).sum(axis=(1, 2)) > 1e-4, np.abs(ref).sum(axis=(1, 2)) > 1e-4))
        rows.append(row)
        print(f"  {cid:22s} {meta['width']}x{meta['height']} {meta['fps']:.0f} fps "
              f"{meta['frames_read']} frames ({meta['frames_detected']} detected) "
              f"{meta['seconds']:.1f} s", flush=True)

    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(out_dir, "video_metadata.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} arrays to {out_dir} in {wall:.0f} s wall time "
          f"({args.workers} worker processes)")


if __name__ == "__main__":
    main()
