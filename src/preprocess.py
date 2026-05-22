"""
ASDMotion — Preprocessing & Pose Extraction

Pipeline: Raw Video → MediaPipe Pose → Spatial Reconstruction → 
          Savitzky-Golay Smoothing → Normalized .npy sequences

Spatial reconstruction aligns every skeleton to a canonical front-facing
pose (facing the camera along the Z-axis) regardless of original camera angle.
"""

import argparse
import os
import sys
import csv
import numpy as np
import cv2
import mediapipe as mp
from scipy.signal import savgol_filter
from tqdm import tqdm
import yaml


# ────────────────────────────────────────────────────────────
# Spatial Reconstruction Utilities
# ────────────────────────────────────────────────────────────

def compute_hip_center(landmarks):
    """
    Compute hip center as midpoint of left hip (23) and right hip (24).
    
    Args:
        landmarks: (33, 3) or (33, 4) array of pose landmarks
    Returns:
        (3,) hip center coordinates
    """
    left_hip = landmarks[23, :3]
    right_hip = landmarks[24, :3]
    return (left_hip + right_hip) / 2.0


def compute_rotation_matrix_y(angle_rad):
    """
    Compute 3D rotation matrix around the Y-axis.
    
    Args:
        angle_rad: Rotation angle in radians
    Returns:
        (3, 3) rotation matrix
    """
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    return np.array([
        [cos_a,  0, sin_a],
        [0,      1, 0    ],
        [-sin_a, 0, cos_a]
    ])


def spatial_reconstruct_frame(landmarks):
    """
    Align a single frame's skeleton to canonical front-facing pose.
    Supports a robust upper-body fallback (using shoulder width scaling)
    if the hips are occluded or out-of-frame.
    
    Args:
        landmarks: (33, 3) or (33, 4) array of pose landmarks
    Returns:
        Spatially reconstructed landmarks
    """
    landmarks = landmarks.copy()
    has_vis = (landmarks.shape[1] > 3)
    
    # Step 1: Translation — hip center to origin
    hip_center = compute_hip_center(landmarks)
    landmarks[:, :3] -= hip_center
    
    # Step 2: Rotation — align shoulders to X-axis (face camera)
    left_shoulder = landmarks[11, :3]
    right_shoulder = landmarks[12, :3]
    shoulder_vec = right_shoulder - left_shoulder  # Vector from left to right shoulder
    
    # Project onto XZ plane and compute angle to X-axis
    angle = np.arctan2(shoulder_vec[2], shoulder_vec[0])  # atan2(z, x)
    
    # Rotate around Y-axis to zero out the Z-component of shoulder vector
    rot_matrix = compute_rotation_matrix_y(-angle)
    landmarks[:, :3] = landmarks[:, :3] @ rot_matrix.T  # Apply rotation to all landmarks
    
    # Step 3: Scale normalization by torso length with upper-body fallback
    hip_center_new = compute_hip_center(landmarks)  # Should be ~origin
    shoulder_center = (landmarks[11, :3] + landmarks[12, :3]) / 2.0
    torso_length = np.linalg.norm(shoulder_center - hip_center_new)
    
    # Evaluate hip visibility (if visibility scores are provided)
    hip_visible = True
    if has_vis:
        left_hip_vis = landmarks[23, 3]
        right_hip_vis = landmarks[24, 3]
        if left_hip_vis < 0.4 or right_hip_vis < 0.4:
            hip_visible = False
            
    shoulder_width = np.linalg.norm(landmarks[12, :3] - landmarks[11, :3])
    min_torso_length = 0.05
    
    # If hips are occluded/invisible, or torso length is implausible, use shoulder width fallback
    if not hip_visible or torso_length < min_torso_length or torso_length > 2.5:
        # Est Torso Length = 1.6811 * Shoulder Width
        estimated_torso_length = 1.6811 * shoulder_width
        if estimated_torso_length > min_torso_length:
            landmarks[:, :3] /= estimated_torso_length
    else:
        # Standard torso scale
        landmarks[:, :3] /= torso_length
    
    return landmarks


def spatial_reconstruct_sequence(sequence):
    """
    Apply spatial reconstruction to every frame in a sequence.
    
    Args:
        sequence: (T, 33, 3) pose landmark sequence
    Returns:
        (T, 33, 3) spatially reconstructed sequence
    """
    reconstructed = np.zeros_like(sequence)
    for t in range(sequence.shape[0]):
        reconstructed[t] = spatial_reconstruct_frame(sequence[t])
    return reconstructed


# ────────────────────────────────────────────────────────────
# Pose Extraction
# ────────────────────────────────────────────────────────────

def extract_poses_from_video(video_path, target_fps=30):
    """
    Extract MediaPipe Pose landmarks from a video file.
    
    Args:
        video_path: Path to the video file
        target_fps: Target FPS for resampling
    Returns:
        (T, 33, 3) numpy array of pose landmarks, or None if extraction fails
    """
    mp_pose = mp.solutions.pose
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open video: {video_path}")
        return None
    
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 30.0  # Fallback
    
    # Frame sampling interval for FPS resampling
    frame_interval = max(1, round(source_fps / target_fps))
    
    landmarks_sequence = []
    frame_idx = 0
    
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,         # Highest accuracy
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resample to target FPS
            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue
            
            # Convert BGR → RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Extract (33, 4) landmarks: x, y, z, visibility
                frame_landmarks = np.array([
                    [lm.x, lm.y, lm.z, lm.visibility]
                    for lm in results.pose_landmarks.landmark
                ])
                landmarks_sequence.append(frame_landmarks)
            else:
                # If pose not detected, use zeros (will be smoothed)
                landmarks_sequence.append(np.zeros((33, 4)))
            
            frame_idx += 1
    
    cap.release()
    
    if len(landmarks_sequence) == 0:
        print(f"  [ERROR] No frames extracted from: {video_path}")
        return None
    
    return np.array(landmarks_sequence, dtype=np.float32)


# ────────────────────────────────────────────────────────────
# Smoothing & Padding
# ────────────────────────────────────────────────────────────

def apply_savgol_smoothing(sequence, window_length=7, polyorder=3):
    """
    Apply Savitzky-Golay smoothing per landmark per axis.
    
    Args:
        sequence: (T, 33, 3) pose sequence
        window_length: Filter window length (must be odd)
        polyorder: Polynomial order
    Returns:
        (T, 33, 3) smoothed sequence
    """
    T, num_landmarks, num_coords = sequence.shape
    
    # Need enough frames for the filter
    if T < window_length:
        return sequence
    
    smoothed = np.copy(sequence)
    for lm in range(num_landmarks):
        for c in range(num_coords):
            signal = sequence[:, lm, c]
            # Only smooth if there's enough variance (not all zeros)
            if np.std(signal) > 1e-8:
                smoothed[:, lm, c] = savgol_filter(
                    signal, window_length=window_length, polyorder=polyorder
                )
    
    return smoothed


def pad_or_truncate(sequence, max_frames):
    """
    Pad with zeros or truncate a sequence to fixed length.
    
    Args:
        sequence: (T, 33, 3) pose sequence
        max_frames: Target sequence length
    Returns:
        (max_frames, 33, 3) fixed-length sequence
    """
    T = sequence.shape[0]
    
    if T >= max_frames:
        return sequence[:max_frames]
    else:
        # Pad with zeros
        pad_length = max_frames - T
        padding = np.zeros((pad_length, sequence.shape[1], sequence.shape[2]), dtype=sequence.dtype)
        return np.concatenate([sequence, padding], axis=0)


# ────────────────────────────────────────────────────────────
# Main Preprocessing Pipeline
# ────────────────────────────────────────────────────────────

def preprocess_video(video_path, config):
    """
    Full preprocessing pipeline for a single video.
    Splits long videos into multiple sliding window chunks.
    
    Returns:
        List of (max_frames, 33, 3) preprocessed skeleton chunks
    """
    target_fps = config['data']['fps']
    max_frames = config['data']['max_frames']
    window_length = config['data']['smoothing_window']
    polyorder = config['data']['smoothing_polyorder']
    
    # Step 1: Extract poses via MediaPipe
    full_sequence = extract_poses_from_video(video_path, target_fps=target_fps)
    if full_sequence is None:
        return []
    
    # Step 2: Spatial reconstruction (camera alignment)
    full_sequence = spatial_reconstruct_sequence(full_sequence)
    
    # Strip visibility column to restore shape (T, 33, 3) for downstream processing
    if full_sequence.shape[2] > 3:
        full_sequence = full_sequence[:, :, :3]
    
    # Step 3: Savitzky-Golay smoothing
    full_sequence = apply_savgol_smoothing(
        full_sequence, window_length=window_length, polyorder=polyorder
    )
    
    # Step 4: Chunking (Sliding Window)
    chunks = []
    T = full_sequence.shape[0]
    
    if T <= max_frames:
        chunks.append(pad_or_truncate(full_sequence, max_frames))
    else:
        stride = max_frames // 2
        for start in range(0, T - max_frames + 1, stride):
            end = start + max_frames
            chunks.append(full_sequence[start:end, :, :])
            
        # Add final window if gap exists
        if (T - max_frames) % stride != 0:
            chunks.append(full_sequence[T-max_frames:, :, :])
    
    return chunks


def run_preprocessing(config):
    """
    Run the full preprocessing pipeline on all videos.
    
    Expects:
        data/raw/asd/       — ASD-positive video files
        data/raw/non_asd/   — ASD-negative video files
    
    Outputs:
        data/processed/features/  — .npy files per video
        data/processed/labels.csv — video_id, label mapping
    """
    raw_dir = config['data']['raw_dir']
    processed_dir = config['data']['processed_dir']
    features_dir = os.path.join(processed_dir, 'features')
    
    os.makedirs(features_dir, exist_ok=True)
    
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
    
    # Collect all videos with labels
    video_entries = []
    
    for label_name, label_value in [('asd', 1), ('non_asd', 0)]:
        class_dir = os.path.join(raw_dir, label_name)
        if not os.path.isdir(class_dir):
            print(f"[WARNING] Directory not found: {class_dir}")
            continue
        
        for filename in sorted(os.listdir(class_dir)):
            ext = os.path.splitext(filename)[1].lower()
            if ext in video_extensions:
                video_entries.append({
                    'filename': filename,
                    'path': os.path.join(class_dir, filename),
                    'label': label_value,
                    'label_name': label_name
                })
    
    if len(video_entries) == 0:
        print("[ERROR] No video files found. Please check your data directory structure:")
        print(f"  Expected: {raw_dir}/asd/     (ASD-positive videos)")
        print(f"  Expected: {raw_dir}/non_asd/ (ASD-negative videos)")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"ASDMotion — Preprocessing Pipeline")
    print(f"{'='*60}")
    print(f"Total videos found: {len(video_entries)}")
    print(f"  ASD:     {sum(1 for e in video_entries if e['label'] == 1)}")
    print(f"  Non-ASD: {sum(1 for e in video_entries if e['label'] == 0)}")
    print(f"{'='*60}\n")
    
    # Process each video
    labels_data = []
    success_count = 0
    fail_count = 0
    
    for entry in tqdm(video_entries, desc="Processing videos"):
        base_id = os.path.splitext(entry['filename'])[0]
        
        tqdm.write(f"  Processing: {entry['filename']} ({entry['label_name']})")
        
        chunks = preprocess_video(entry['path'], config)
        
        if not chunks:
            tqdm.write(f"  [ERROR] Failed to extract poses from {entry['filename']}")
            fail_count += 1
            continue
        
        for idx, sequence in enumerate(chunks):
            video_id = f"{base_id}_c{idx}"
            # Save features as .npy
            npy_path = os.path.join(features_dir, f"{video_id}.npy")
            np.save(npy_path, sequence)
            
            labels_data.append({
                'video_id': video_id,
                'label': entry['label']
            })
            tqdm.write(f"    ✓ Saved: {npy_path} — shape: {sequence.shape}")
        
        success_count += 1
    
    # Save labels CSV
    labels_path = os.path.join(processed_dir, 'labels.csv')
    with open(labels_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['video_id', 'label'])
        writer.writeheader()
        writer.writerows(labels_data)
    
    print(f"\n{'='*60}")
    print(f"Preprocessing Complete")
    print(f"{'='*60}")
    print(f"  Successful: {success_count}")
    print(f"  Failed:     {fail_count}")
    print(f"  Features:   {features_dir}")
    print(f"  Labels:     {labels_path}")
    print(f"{'='*60}")


# ────────────────────────────────────────────────────────────
# CLI Entry Point
# ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASDMotion — Preprocess videos to skeleton sequences')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration YAML file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    run_preprocessing(config)
