"""
ASD Pose Dataset Converter -> ASDMotion Training Pipeline

Converts pre-segmented ADOS clinical assessment pose clips from dataset.pkl
into the (300, 33, 3) skeleton format used by ASDMotion.

Source format:
  - 17 COCO keypoints x (x, y) pixel coordinates + confidence scores
  - binary_label: 1 = ASD stereotypical behavior, 0 = typical behavior
  - identifier: '{child_id}_{session}_{camera}_{action}_{start}_{end}'
  - Pre-split into train/test

Target format:
  - (300, 33, 3) MediaPipe-compatible skeleton sequences
  - Normalized coordinates with z=0

COCO 17 -> MediaPipe 33 Mapping:
  Direct: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles
  Approximated: eye inner/outer, mouth, pinky/index/thumb, heel/foot_index
"""

import argparse
import csv
import os
import re
import numpy as np
from scipy.signal import savgol_filter, resample
from tqdm import tqdm
import yaml
import pickle


# ──────────────────────────────────────────────────────────────
# COCO 17 -> MediaPipe 33 Joint Mapping
# ──────────────────────────────────────────────────────────────

# COCO keypoint indices
COCO = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2,
    'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6,
    'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10,
    'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14,
    'left_ankle': 15, 'right_ankle': 16,
}

# MediaPipe landmark indices
MP = {
    'nose': 0,
    'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
    'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
    'left_ear': 7, 'right_ear': 8,
    'mouth_left': 9, 'mouth_right': 10,
    'left_shoulder': 11, 'right_shoulder': 12,
    'left_elbow': 13, 'right_elbow': 14,
    'left_wrist': 15, 'right_wrist': 16,
    'left_pinky': 17, 'right_pinky': 18,
    'left_index': 19, 'right_index': 20,
    'left_thumb': 21, 'right_thumb': 22,
    'left_hip': 23, 'right_hip': 24,
    'left_knee': 25, 'right_knee': 26,
    'left_ankle': 27, 'right_ankle': 28,
    'left_heel': 29, 'right_heel': 30,
    'left_foot_index': 31, 'right_foot_index': 32,
}


def map_coco_to_mediapipe(coco_kp, coco_scores, img_shape):
    """
    Map (T, 17, 2) COCO keypoints to (T, 33, 3) MediaPipe landmarks.

    Args:
        coco_kp: (T, 17, 2) pixel coordinates
        coco_scores: (T, 17) confidence scores
        img_shape: (height, width) of source video

    Returns:
        mp_landmarks: (T, 33, 3) normalized coordinates with z=0
    """
    T = coco_kp.shape[0]
    h, w = img_shape

    # Normalize pixel coords to [0, 1]
    kp_norm = np.zeros_like(coco_kp)
    if w > 0 and h > 0:
        kp_norm[:, :, 0] = coco_kp[:, :, 0] / w  # x
        kp_norm[:, :, 1] = coco_kp[:, :, 1] / h  # y

    # Zero out low-confidence keypoints
    low_conf = coco_scores < 0.2
    kp_norm[low_conf] = 0.0

    mp = np.zeros((T, 33, 3), dtype=np.float32)

    # Direct mappings (COCO -> MediaPipe)
    direct_map = {
        COCO['nose']: MP['nose'],
        COCO['left_eye']: MP['left_eye'],
        COCO['right_eye']: MP['right_eye'],
        COCO['left_ear']: MP['left_ear'],
        COCO['right_ear']: MP['right_ear'],
        COCO['left_shoulder']: MP['left_shoulder'],
        COCO['right_shoulder']: MP['right_shoulder'],
        COCO['left_elbow']: MP['left_elbow'],
        COCO['right_elbow']: MP['right_elbow'],
        COCO['left_wrist']: MP['left_wrist'],
        COCO['right_wrist']: MP['right_wrist'],
        COCO['left_hip']: MP['left_hip'],
        COCO['right_hip']: MP['right_hip'],
        COCO['left_knee']: MP['left_knee'],
        COCO['right_knee']: MP['right_knee'],
        COCO['left_ankle']: MP['left_ankle'],
        COCO['right_ankle']: MP['right_ankle'],
    }
    for coco_idx, mp_idx in direct_map.items():
        mp[:, mp_idx, 0] = kp_norm[:, coco_idx, 0]  # x
        mp[:, mp_idx, 1] = kp_norm[:, coco_idx, 1]  # y
        # z stays 0

    # Approximated landmarks
    nose = kp_norm[:, COCO['nose']]
    l_eye = kp_norm[:, COCO['left_eye']]
    r_eye = kp_norm[:, COCO['right_eye']]
    l_ear = kp_norm[:, COCO['left_ear']]
    r_ear = kp_norm[:, COCO['right_ear']]
    l_sho = kp_norm[:, COCO['left_shoulder']]
    r_sho = kp_norm[:, COCO['right_shoulder']]
    l_wrist = kp_norm[:, COCO['left_wrist']]
    r_wrist = kp_norm[:, COCO['right_wrist']]
    l_elbow = kp_norm[:, COCO['left_elbow']]
    r_elbow = kp_norm[:, COCO['right_elbow']]
    l_ankle = kp_norm[:, COCO['left_ankle']]
    r_ankle = kp_norm[:, COCO['right_ankle']]
    l_knee = kp_norm[:, COCO['left_knee']]
    r_knee = kp_norm[:, COCO['right_knee']]

    # Eye inner/outer: offset from eye center toward/away from nose
    eye_to_nose_l = (nose - l_eye) * 0.3
    eye_to_nose_r = (nose - r_eye) * 0.3
    mp[:, MP['left_eye_inner'], :2] = l_eye + eye_to_nose_l
    mp[:, MP['left_eye_outer'], :2] = l_eye - eye_to_nose_l
    mp[:, MP['right_eye_inner'], :2] = r_eye + eye_to_nose_r
    mp[:, MP['right_eye_outer'], :2] = r_eye - eye_to_nose_r

    # Mouth: midpoint between nose and shoulder midpoint
    sho_mid = (l_sho + r_sho) * 0.5
    mouth_center = nose * 0.4 + sho_mid * 0.6
    mouth_offset = (l_ear - r_ear) * 0.1
    mp[:, MP['mouth_left'], :2] = mouth_center + mouth_offset
    mp[:, MP['mouth_right'], :2] = mouth_center - mouth_offset

    # Hand landmarks: offset from wrist along forearm direction
    l_forearm = l_wrist - l_elbow
    r_forearm = r_wrist - r_elbow
    forearm_len_l = np.linalg.norm(l_forearm, axis=1, keepdims=True) + 1e-8
    forearm_len_r = np.linalg.norm(r_forearm, axis=1, keepdims=True) + 1e-8
    l_dir = l_forearm / forearm_len_l
    r_dir = r_forearm / forearm_len_r
    l_perp = np.stack([-l_dir[:, 1], l_dir[:, 0]], axis=1)
    r_perp = np.stack([-r_dir[:, 1], r_dir[:, 0]], axis=1)

    hand_ext = 0.15  # Extension beyond wrist
    mp[:, MP['left_index'], :2] = l_wrist + l_dir * hand_ext
    mp[:, MP['right_index'], :2] = r_wrist + r_dir * hand_ext
    mp[:, MP['left_pinky'], :2] = l_wrist + l_dir * hand_ext * 0.8 + l_perp * 0.02
    mp[:, MP['right_pinky'], :2] = r_wrist + r_dir * hand_ext * 0.8 - r_perp * 0.02
    mp[:, MP['left_thumb'], :2] = l_wrist + l_dir * hand_ext * 0.6 - l_perp * 0.02
    mp[:, MP['right_thumb'], :2] = r_wrist + r_dir * hand_ext * 0.6 + r_perp * 0.02

    # Foot landmarks: offset from ankle along shin direction
    l_shin = l_ankle - l_knee
    r_shin = r_ankle - r_knee
    shin_len_l = np.linalg.norm(l_shin, axis=1, keepdims=True) + 1e-8
    shin_len_r = np.linalg.norm(r_shin, axis=1, keepdims=True) + 1e-8
    l_shin_dir = l_shin / shin_len_l
    r_shin_dir = r_shin / shin_len_r

    foot_ext = 0.05
    mp[:, MP['left_heel'], :2] = l_ankle + l_shin_dir * foot_ext * 0.5
    mp[:, MP['right_heel'], :2] = r_ankle + r_shin_dir * foot_ext * 0.5
    mp[:, MP['left_foot_index'], :2] = l_ankle + l_shin_dir * foot_ext
    mp[:, MP['right_foot_index'], :2] = r_ankle + r_shin_dir * foot_ext

    return mp


# ──────────────────────────────────────────────────────────────
# Spatial Reconstruction (ensure canonical scale/translation)
# ──────────────────────────────────────────────────────────────

def compute_hip_center(landmarks):
    # MediaPipe Hip indices: 23 (left), 24 (right)
    left_hip = landmarks[23]
    right_hip = landmarks[24]
    return (left_hip + right_hip) / 2.0

def spatial_reconstruct_frame(landmarks):
    """
    Align skeleton: translate hip to origin, scale by torso length.
    Rotation is a no-op for 2D data (z=0).
    """
    landmarks = landmarks.copy()
    
    # 1. Translation
    hip_center = compute_hip_center(landmarks)
    landmarks -= hip_center
    
    # 2. Scale Normalization
    # Mid-shoulder (11, 12) to hip-center
    shoulder_center = (landmarks[11] + landmarks[12]) / 2.0
    torso_length = np.linalg.norm(shoulder_center) # hip_center is now origin
    
    # Safety threshold: avoid division by tiny values which cause coordinate explosions
    min_torso_length = 0.05 
    if torso_length > min_torso_length:
        landmarks /= torso_length
    else:
        # Fallback: if torso is too small, use a default scale or skip scaling
        # This prevents values like 494.0 appearing in your data
        pass
        
    return landmarks


def process_clip(kp, scores, img_shape, window_size=300, stride=150):
    """
    Process a single clip: map joints, filter, smooth, 
    and split into sliding window chunks of size `window_size`.

    Returns:
        List of (window_size, 33, 3) processed chunks
    """
    T = kp.shape[0]

    # Skip clips with too few valid frames
    valid_mask = np.any(kp != 0, axis=(1, 2))
    n_valid = np.sum(valid_mask)
    if n_valid < 10:
        return []

    # Map to MediaPipe format
    mp_full = map_coco_to_mediapipe(kp, scores, img_shape)

    # 0. Spatial Reconstruction (Canonical Alignment)
    for t in range(mp_full.shape[0]):
        if np.any(mp_full[t] != 0):
            mp_full[t] = spatial_reconstruct_frame(mp_full[t])

    # 1. Interpolate missing frames (zero-filled)
    for j in range(33):
        for c in range(3):
            signal = mp_full[:, j, c]
            nonzero = np.where(signal != 0)[0]
            if len(nonzero) < 2:
                continue
            zero_idx = np.where(signal == 0)[0]
            zero_in_range = zero_idx[
                (zero_idx > nonzero[0]) & (zero_idx < nonzero[-1])
            ]
            if len(zero_in_range) > 0:
                signal[zero_in_range] = np.interp(
                    zero_in_range, nonzero, signal[nonzero]
                )
                mp_full[:, j, c] = signal

    # 2. Savitzky-Golay smoothing
    min_frames_for_smooth = 15
    if T >= min_frames_for_smooth:
        window = min(11, T if T % 2 == 1 else T - 1)
        if window >= 5:
            for j in range(33):
                for c in range(3):
                    mp_full[:, j, c] = savgol_filter(
                        mp_full[:, j, c], window, polyorder=3
                    )

    # 3. Sliding Window Splitting
    chunks = []
    
    if T <= window_size:
        # Case A: Clip is shorter than window -> Resample to window_size
        resampled = np.zeros((window_size, 33, 3), dtype=np.float32)
        for j in range(33):
            for c in range(3):
                resampled[:, j, c] = resample(mp_full[:, j, c], window_size)
        chunks.append(resampled.astype(np.float32))
    else:
        # Case B: Clip is longer than window -> Extract sliding windows
        for start in range(0, T - window_size + 1, stride):
            end = start + window_size
            chunk = mp_full[start:end, :, :]
            chunks.append(chunk.astype(np.float32))
            
        # Add the final window if the last stride left a gap
        if (T - window_size) % stride != 0:
            chunks.append(mp_full[T-window_size:, :, :].astype(np.float32))

    return chunks


def extract_subject_from_identifier(identifier):
    """
    Extract child ID from identifier string.
    Format: '{child_id}_{session}_{camera}_{action}_{start}_{end}'
    Example: '48_1_3_Clapping_649_654' -> '48'
    """
    parts = identifier.split('_')
    return parts[0]


def run_conversion(pkl_path, config):
    """Convert all clips from dataset.pkl and integrate into training data."""
    processed_dir = config['data']['processed_dir']
    features_dir = os.path.join(processed_dir, 'features')
    labels_path = os.path.join(processed_dir, 'labels.csv')
    target_frames = config['data']['max_frames']

    os.makedirs(features_dir, exist_ok=True)

    # Load existing labels
    existing_labels = []
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            reader = csv.DictReader(f)
            existing_labels = list(reader)
        print(f"Loaded {len(existing_labels)} existing entries from labels.csv")

    existing_ids = set(entry['video_id'] for entry in existing_labels)

    # Load dataset
    print(f"Loading {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        dataset = pickle.load(f)

    annotations = dataset['annotations']
    print(f"Total annotations: {len(annotations)}")

    # 3. First Pass: Identify ASD Subjects
    print("Scanning for ASD subjects...")
    asd_subjects = set()
    for ann in annotations:
        identifier = ann['identifier']
        # Extract child ID (first part of identifier)
        child_id = identifier.split('_')[0]
        if ann.get('binary_label', 0) == 1:
            asd_subjects.add(child_id)
    
    print(f"Found {len(asd_subjects)} subjects with ASD markers.")

    # 4. Second Pass: Process all clips with correct labels
    new_entries = []
    asd_count = 0
    control_count = 0
    skipped = 0
    failed = 0

    print(f"\n{'='*60}")
    print(f"ASD Pose -> ASDMotion Converter (Subject-Level Labeling)")
    print(f"{'='*60}")
    
    for ann in tqdm(annotations, desc="Converting clips"):
        identifier = ann['identifier']
        child_id = identifier.split('_')[0]
        base_id = f"asdpose_{identifier}"

        kp = ann['keypoint']
        scores = ann['keypoint_score']
        img_shape = ann['img_shape']
        
        # CORRECT CLINICAL LABEL: If subject is in ASD list, all clips are 1
        label = 1 if child_id in asd_subjects else 0

        try:
            chunks = process_clip(kp, scores, img_shape, window_size=target_frames, stride=target_frames//2)
        except Exception:
            failed += 1
            continue

        if not chunks:
            failed += 1
            continue

        for idx, seq in enumerate(chunks):
            video_id = f"{base_id}_c{idx}"
            if video_id in existing_ids:
                skipped += 1
                continue

            npy_path = os.path.join(features_dir, f"{video_id}.npy")
            np.save(npy_path, seq)

            new_entries.append({'video_id': video_id, 'label': label})
            if label == 1: asd_count += 1
            else: control_count += 1

    # Merge with existing labels
    all_labels = existing_labels + new_entries
    with open(labels_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['video_id', 'label'])
        writer.writeheader()
        writer.writerows(all_labels)

    print(f"\n{'='*60}")
    print(f"Conversion Complete")
    print(f"{'='*60}")
    print(f"  New sequences added: {len(new_entries)}")
    print(f"    ASD (label=1):    {asd_count}")
    print(f"    Non-ASD (label=0): {control_count}")
    print(f"  Skipped (existing):  {skipped}")
    print(f"  Failed:              {failed}")
    print(f"  Total in labels.csv: {len(all_labels)}")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert ASD Pose dataset to ASDMotion format'
    )
    parser.add_argument(
        '--dataset', type=str, required=True,
        help='Path to dataset.pkl file'
    )
    parser.add_argument(
        '--config', type=str, default='configs/config.yaml',
        help='Path to ASDMotion config YAML'
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    run_conversion(args.dataset, config)
