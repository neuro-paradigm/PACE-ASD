"""
Move4AS Dataset Converter → ASDMotion Training Pipeline

Converts the Move4AS 3D motion capture dataset (.mat files) into the
same (T, 33, 3) skeleton format used by ASDMotion's MediaPipe-based pipeline.

Dataset: "A Multimodal Dataset Addressing Motor Function in Autism"
  - P* folders = ASD (clinical group, 14 participants)
  - S* folders = Control / Non-ASD (20 participants)
  - rigidbodyData: 21 joints × (3 pos + 4 quat) per frame
  - stimRegister: trial phase triggers (4 = execution)

Joint Mapping: 21 MoCap joints → 33 MediaPipe landmarks
"""

import argparse
import csv
import os
import sys
import glob
import numpy as np
import scipy.io as sio
from scipy.signal import savgol_filter, resample
from tqdm import tqdm
import yaml

# ────────────────────────────────────────────────────────────
# Move4AS Joint Definitions (from Skeletal_info.mat)
# ────────────────────────────────────────────────────────────
# Column indices in rigidbodyData (7 × 21) — rows 0:3 = XYZ position
MOCAP_JOINTS = {
    'Hip':        0,
    'Ab':         1,
    'Chest':      2,
    'Neck':       3,
    'Head':       4,
    'LShoulder':  5,
    'LUArm':      6,
    'LFArm':      7,   # ≈ Left Elbow
    'LHand':      8,   # ≈ Left Wrist
    'RShoulder':  9,
    'RUArm':     10,
    'RFArm':     11,   # ≈ Right Elbow
    'RHand':     12,   # ≈ Right Wrist
    'RThigh':    13,
    'RShin':     14,   # ≈ Right Knee
    'RFoot':     15,   # ≈ Right Ankle
    'RToe':      16,
    'LThigh':    17,
    'LShin':     18,   # ≈ Left Knee
    'LFoot':     19,   # ≈ Left Ankle
    'LToe':      20,
}

# ────────────────────────────────────────────────────────────
# MediaPipe Pose Landmark Indices
# ────────────────────────────────────────────────────────────
MP_NOSE = 0
MP_LEFT_EYE_INNER = 1
MP_LEFT_EYE = 2
MP_LEFT_EYE_OUTER = 3
MP_RIGHT_EYE_INNER = 4
MP_RIGHT_EYE = 5
MP_RIGHT_EYE_OUTER = 6
MP_LEFT_EAR = 7
MP_RIGHT_EAR = 8
MP_MOUTH_LEFT = 9
MP_MOUTH_RIGHT = 10
MP_LEFT_SHOULDER = 11
MP_RIGHT_SHOULDER = 12
MP_LEFT_ELBOW = 13
MP_RIGHT_ELBOW = 14
MP_LEFT_WRIST = 15
MP_RIGHT_WRIST = 16
MP_LEFT_PINKY = 17
MP_RIGHT_PINKY = 18
MP_LEFT_INDEX = 19
MP_RIGHT_INDEX = 20
MP_LEFT_THUMB = 21
MP_RIGHT_THUMB = 22
MP_LEFT_HIP = 23
MP_RIGHT_HIP = 24
MP_LEFT_KNEE = 25
MP_RIGHT_KNEE = 26
MP_LEFT_ANKLE = 27
MP_RIGHT_ANKLE = 28
MP_LEFT_HEEL = 29
MP_RIGHT_HEEL = 30
MP_LEFT_FOOT_INDEX = 31
MP_RIGHT_FOOT_INDEX = 32


def extract_positions_from_frame(rb_frame):
    """
    Extract XYZ positions from a single rigidbodyData frame.

    Args:
        rb_frame: (7, 21) array — rows 0-2 are XYZ positions, rows 3-6 are quaternion
    Returns:
        (21, 3) array of joint positions
    """
    # Rows 0, 1, 2 = X, Y, Z world positions
    positions = rb_frame[:3, :].T  # (21, 3)
    return positions


def map_mocap_to_mediapipe(mocap_positions):
    """
    Map 21 MoCap joint positions to 33 MediaPipe landmark positions.

    Uses direct mapping where possible, interpolation for approximate joints,
    and nearest-neighbor for face/hand landmarks that have no MoCap equivalent.

    Args:
        mocap_positions: (21, 3) MoCap joint positions
    Returns:
        (33, 3) MediaPipe-format landmark positions
    """
    mp_landmarks = np.zeros((33, 3), dtype=np.float32)

    # Helper to get MoCap joint position
    j = mocap_positions

    hip = j[MOCAP_JOINTS['Hip']]
    ab = j[MOCAP_JOINTS['Ab']]
    chest = j[MOCAP_JOINTS['Chest']]
    neck = j[MOCAP_JOINTS['Neck']]
    head = j[MOCAP_JOINTS['Head']]
    l_shoulder = j[MOCAP_JOINTS['LShoulder']]
    r_shoulder = j[MOCAP_JOINTS['RShoulder']]
    l_elbow = j[MOCAP_JOINTS['LFArm']]
    r_elbow = j[MOCAP_JOINTS['RFArm']]
    l_wrist = j[MOCAP_JOINTS['LHand']]
    r_wrist = j[MOCAP_JOINTS['RHand']]
    l_thigh = j[MOCAP_JOINTS['LThigh']]
    r_thigh = j[MOCAP_JOINTS['RThigh']]
    l_knee = j[MOCAP_JOINTS['LShin']]
    r_knee = j[MOCAP_JOINTS['RShin']]
    l_ankle = j[MOCAP_JOINTS['LFoot']]
    r_ankle = j[MOCAP_JOINTS['RFoot']]
    l_toe = j[MOCAP_JOINTS['LToe']]
    r_toe = j[MOCAP_JOINTS['RToe']]

    # === HEAD / FACE (approximate from Head + Neck) ===
    # Head is the only point we have — spread face landmarks around it
    head_to_neck = neck - head
    head_right = np.cross(head_to_neck, [0, 0, 1])
    head_right_norm = np.linalg.norm(head_right)
    if head_right_norm > 1e-6:
        head_right = head_right / head_right_norm * 0.05
    else:
        head_right = np.array([0.05, 0, 0])

    mp_landmarks[MP_NOSE] = head + head_to_neck * 0.3           # Nose slightly below head top
    mp_landmarks[MP_LEFT_EYE_INNER] = head + head_right * 0.3
    mp_landmarks[MP_LEFT_EYE] = head + head_right * 0.5
    mp_landmarks[MP_LEFT_EYE_OUTER] = head + head_right * 0.7
    mp_landmarks[MP_RIGHT_EYE_INNER] = head - head_right * 0.3
    mp_landmarks[MP_RIGHT_EYE] = head - head_right * 0.5
    mp_landmarks[MP_RIGHT_EYE_OUTER] = head - head_right * 0.7
    mp_landmarks[MP_LEFT_EAR] = head + head_right * 1.0
    mp_landmarks[MP_RIGHT_EAR] = head - head_right * 1.0
    mp_landmarks[MP_MOUTH_LEFT] = head + head_to_neck * 0.5 + head_right * 0.3
    mp_landmarks[MP_MOUTH_RIGHT] = head + head_to_neck * 0.5 - head_right * 0.3

    # === UPPER BODY (direct mappings) ===
    mp_landmarks[MP_LEFT_SHOULDER] = l_shoulder
    mp_landmarks[MP_RIGHT_SHOULDER] = r_shoulder
    mp_landmarks[MP_LEFT_ELBOW] = l_elbow
    mp_landmarks[MP_RIGHT_ELBOW] = r_elbow
    mp_landmarks[MP_LEFT_WRIST] = l_wrist
    mp_landmarks[MP_RIGHT_WRIST] = r_wrist

    # === HANDS (approximate — spread around wrist) ===
    l_forearm_dir = l_wrist - l_elbow
    l_forearm_norm = np.linalg.norm(l_forearm_dir)
    if l_forearm_norm > 1e-6:
        l_forearm_dir = l_forearm_dir / l_forearm_norm * 0.03
    mp_landmarks[MP_LEFT_PINKY] = l_wrist + l_forearm_dir + np.array([0.02, 0, 0])
    mp_landmarks[MP_LEFT_INDEX] = l_wrist + l_forearm_dir - np.array([0.02, 0, 0])
    mp_landmarks[MP_LEFT_THUMB] = l_wrist + l_forearm_dir + np.array([0, 0, 0.02])

    r_forearm_dir = r_wrist - r_elbow
    r_forearm_norm = np.linalg.norm(r_forearm_dir)
    if r_forearm_norm > 1e-6:
        r_forearm_dir = r_forearm_dir / r_forearm_norm * 0.03
    mp_landmarks[MP_RIGHT_PINKY] = r_wrist + r_forearm_dir + np.array([0.02, 0, 0])
    mp_landmarks[MP_RIGHT_INDEX] = r_wrist + r_forearm_dir - np.array([0.02, 0, 0])
    mp_landmarks[MP_RIGHT_THUMB] = r_wrist + r_forearm_dir + np.array([0, 0, 0.02])

    # === HIPS (derive L/R from Hip center + shoulder direction) ===
    shoulder_center = (l_shoulder + r_shoulder) / 2.0
    shoulder_vec = r_shoulder - l_shoulder
    shoulder_width = np.linalg.norm(shoulder_vec)
    if shoulder_width > 1e-6:
        hip_offset = (shoulder_vec / shoulder_width) * shoulder_width * 0.35
    else:
        hip_offset = np.array([0.1, 0, 0])

    mp_landmarks[MP_LEFT_HIP] = hip - hip_offset
    mp_landmarks[MP_RIGHT_HIP] = hip + hip_offset

    # === LOWER BODY (direct mappings) ===
    mp_landmarks[MP_LEFT_KNEE] = l_knee
    mp_landmarks[MP_RIGHT_KNEE] = r_knee
    mp_landmarks[MP_LEFT_ANKLE] = l_ankle
    mp_landmarks[MP_RIGHT_ANKLE] = r_ankle

    # === FEET (approximate heel/toe from ankle and toe) ===
    l_foot_dir = l_toe - l_ankle
    l_foot_norm = np.linalg.norm(l_foot_dir)
    if l_foot_norm > 1e-6:
        l_foot_unit = l_foot_dir / l_foot_norm
    else:
        l_foot_unit = np.array([0, 0, 1])

    mp_landmarks[MP_LEFT_HEEL] = l_ankle - l_foot_unit * 0.05
    mp_landmarks[MP_LEFT_FOOT_INDEX] = l_toe

    r_foot_dir = r_toe - r_ankle
    r_foot_norm = np.linalg.norm(r_foot_dir)
    if r_foot_norm > 1e-6:
        r_foot_unit = r_foot_dir / r_foot_norm
    else:
        r_foot_unit = np.array([0, 0, 1])

    mp_landmarks[MP_RIGHT_HEEL] = r_ankle - r_foot_unit * 0.05
    mp_landmarks[MP_RIGHT_FOOT_INDEX] = r_toe

    return mp_landmarks


# ────────────────────────────────────────────────────────────
# Spatial Reconstruction (same as preprocess.py)
# ────────────────────────────────────────────────────────────

def compute_hip_center(landmarks):
    left_hip = landmarks[MP_LEFT_HIP]
    right_hip = landmarks[MP_RIGHT_HIP]
    return (left_hip + right_hip) / 2.0


def compute_rotation_matrix_y(angle_rad):
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    return np.array([
        [cos_a,  0, sin_a],
        [0,      1, 0    ],
        [-sin_a, 0, cos_a]
    ])


def spatial_reconstruct_frame(landmarks):
    landmarks = landmarks.copy()
    hip_center = compute_hip_center(landmarks)
    landmarks -= hip_center

    left_shoulder = landmarks[MP_LEFT_SHOULDER]
    right_shoulder = landmarks[MP_RIGHT_SHOULDER]
    shoulder_vec = right_shoulder - left_shoulder
    angle = np.arctan2(shoulder_vec[2], shoulder_vec[0])
    rot_matrix = compute_rotation_matrix_y(-angle)
    landmarks = landmarks @ rot_matrix.T

    hip_center_new = compute_hip_center(landmarks)
    shoulder_center = (landmarks[MP_LEFT_SHOULDER] + landmarks[MP_RIGHT_SHOULDER]) / 2.0
    torso_length = np.linalg.norm(shoulder_center - hip_center_new)
    min_torso_length = 0.1 # Safety threshold
    if torso_length > min_torso_length:
        landmarks /= torso_length

    return landmarks


# ────────────────────────────────────────────────────────────
# Trial Segmentation
# ────────────────────────────────────────────────────────────

def extract_execution_segments(stim_register):
    """
    Find contiguous segments where stimRegister == 4 (execution phase).

    Args:
        stim_register: (T,) array of trigger values
    Returns:
        List of (start_frame, end_frame) tuples
    """
    segments = []
    in_exec = False
    start = 0

    for i in range(len(stim_register)):
        if stim_register[i] == 4 and not in_exec:
            start = i
            in_exec = True
        elif stim_register[i] != 4 and in_exec:
            segments.append((start, i - 1))
            in_exec = False

    if in_exec:
        segments.append((start, len(stim_register) - 1))

    return segments


# ────────────────────────────────────────────────────────────
# Resampling & Smoothing
# ────────────────────────────────────────────────────────────

def resample_sequence(sequence, source_fps, target_fps):
    """
    Resample a skeleton sequence from source_fps to target_fps.

    Args:
        sequence: (T_src, 33, 3) array
        source_fps: Original sampling rate
        target_fps: Target sampling rate
    Returns:
        (T_tgt, 33, 3) resampled sequence
    """
    T_src, num_landmarks, num_coords = sequence.shape
    T_tgt = int(round(T_src * target_fps / source_fps))

    if T_tgt <= 0:
        return None

    resampled = np.zeros((T_tgt, num_landmarks, num_coords), dtype=np.float32)

    for lm in range(num_landmarks):
        for c in range(num_coords):
            resampled[:, lm, c] = resample(sequence[:, lm, c], T_tgt)

    return resampled


def apply_savgol_smoothing(sequence, window_length=7, polyorder=3):
    T, num_landmarks, num_coords = sequence.shape
    if T < window_length:
        return sequence

    smoothed = np.copy(sequence)
    for lm in range(num_landmarks):
        for c in range(num_coords):
            signal = sequence[:, lm, c]
            if np.std(signal) > 1e-8:
                smoothed[:, lm, c] = savgol_filter(
                    signal, window_length=window_length, polyorder=polyorder
                )

    return smoothed


def pad_or_truncate(sequence, max_frames):
    T = sequence.shape[0]
    if T >= max_frames:
        return sequence[:max_frames]
    else:
        pad_length = max_frames - T
        padding = np.zeros(
            (pad_length, sequence.shape[1], sequence.shape[2]),
            dtype=sequence.dtype
        )
        return np.concatenate([sequence, padding], axis=0)


# ────────────────────────────────────────────────────────────
# Add MoCap-specific noise augmentation to simulate video noise
# ────────────────────────────────────────────────────────────

def add_mocap_to_video_noise(sequence, noise_std=0.008):
    """
    Add slight Gaussian noise to clean MoCap data to reduce domain gap
    with video-extracted skeletons (which have more jitter).

    Args:
        sequence: (T, 33, 3) skeleton sequence
        noise_std: Standard deviation of noise
    Returns:
        (T, 33, 3) noised sequence
    """
    noise = np.random.normal(0, noise_std, size=sequence.shape).astype(np.float32)
    # Don't add noise to zero-padded frames
    mask = np.any(sequence != 0, axis=(1, 2), keepdims=True)
    return sequence + noise * mask


# ────────────────────────────────────────────────────────────
# Main Conversion Pipeline
# ────────────────────────────────────────────────────────────

def convert_mat_file(mat_path, config, source_fps=50.0):
    """
    Convert a single mdata .mat file to a list of (T, 33, 3) sequences.

    Args:
        mat_path: Path to mdata_*.mat file
        config: Configuration dictionary
        source_fps: MoCap sampling rate
    Returns:
        List of (max_frames, 33, 3) processed sequences
    """
    target_fps = config['data']['fps']
    max_frames = config['data']['max_frames']
    window_length = config['data']['smoothing_window']
    polyorder = config['data']['smoothing_polyorder']

    try:
        data = sio.loadmat(mat_path)
    except Exception as e:
        print(f"  [ERROR] Cannot load {mat_path}: {e}")
        return []

    rigidbody = data.get('rigidbodyData', None)
    stim = data.get('stimRegister', None)

    if rigidbody is None or stim is None:
        print(f"  [ERROR] Missing rigidbodyData or stimRegister in {mat_path}")
        return []

    stim = stim.flatten()
    total_frames = rigidbody.shape[1]

    # Find execution segments
    segments = extract_execution_segments(stim)

    if len(segments) == 0:
        print(f"  [WARNING] No execution segments found in {mat_path}")
        return []

    processed_sequences = []

    for seg_idx, (start, end) in enumerate(segments):
        seg_length = end - start + 1
        if seg_length < 20:  # Skip very short segments
            continue

        # Extract (T, 21, 3) positions for this segment
        segment_positions = []
        valid = True

        for frame_idx in range(start, end + 1):
            rb_frame = rigidbody[0, frame_idx]  # (7, 21)

            # Check for NaN or missing data
            if rb_frame is None or not hasattr(rb_frame, 'shape'):
                valid = False
                break

            positions = extract_positions_from_frame(rb_frame)  # (21, 3)

            if np.any(np.isnan(positions)):
                # Interpolate or skip
                valid = False
                break

            segment_positions.append(positions)

        if not valid or len(segment_positions) < 20:
            continue

        segment_positions = np.array(segment_positions, dtype=np.float32)  # (T, 21, 3)

        # Map each frame: 21 MoCap → 33 MediaPipe
        mp_sequence = np.zeros(
            (segment_positions.shape[0], 33, 3), dtype=np.float32
        )
        for t in range(segment_positions.shape[0]):
            mp_sequence[t] = map_mocap_to_mediapipe(segment_positions[t])

        # Resample to target FPS
        mp_sequence = resample_sequence(mp_sequence, source_fps, target_fps)
        if mp_sequence is None or mp_sequence.shape[0] < 10:
            continue

        # Spatial reconstruction (frame-by-frame)
        for t in range(mp_sequence.shape[0]):
            mp_sequence[t] = spatial_reconstruct_frame(mp_sequence[t])

        # Savitzky-Golay smoothing
        mp_sequence = apply_savgol_smoothing(
            mp_sequence, window_length=window_length, polyorder=polyorder
        )

        # Add slight noise to bridge domain gap with video data
        mp_sequence = add_mocap_to_video_noise(mp_sequence, noise_std=0.008)

        # Pad/truncate or Chunk
        if mp_sequence.shape[0] <= max_frames:
            # Short segment: pad/resample to max_frames
            final_seq = pad_or_truncate(mp_sequence, max_frames)
            processed_sequences.append(final_seq)
        else:
            # Long segment: split into sliding windows (10s chunks with 50% overlap)
            stride = max_frames // 2
            T = mp_sequence.shape[0]
            for start in range(0, T - max_frames + 1, stride):
                end = start + max_frames
                chunk = mp_sequence[start:end, :, :]
                processed_sequences.append(chunk)
                
            # Add final window if gap exists
            if (T - max_frames) % stride != 0:
                processed_sequences.append(mp_sequence[T-max_frames:, :, :])

    return processed_sequences


def run_conversion(dataset_path, config):
    """
    Convert all Move4AS .mat files and integrate into ASDMotion training data.

    Args:
        dataset_path: Path to Move4AS Dataset folder
        config: Configuration dictionary
    """
    processed_dir = config['data']['processed_dir']
    features_dir = os.path.join(processed_dir, 'features')
    labels_path = os.path.join(processed_dir, 'labels.csv')

    os.makedirs(features_dir, exist_ok=True)

    # Load existing labels if present
    existing_labels = []
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            reader = csv.DictReader(f)
            existing_labels = list(reader)
        print(f"Loaded {len(existing_labels)} existing entries from labels.csv")

    existing_ids = set(entry['video_id'] for entry in existing_labels)

    # Scan the dataset
    participant_dirs = sorted(glob.glob(os.path.join(dataset_path, '*')))

    asd_count = 0
    control_count = 0
    new_entries = []
    total_sequences = 0
    failed = 0

    print(f"\n{'='*60}")
    print(f"Move4AS -> ASDMotion Converter (Walk Only)")
    print(f"{'='*60}")
    print(f"Dataset path: {dataset_path}")
    print(f"Output dir:   {features_dir}")
    print(f"{'='*60}\n")

    for pdir in tqdm(participant_dirs, desc="Processing participants"):
        dirname = os.path.basename(pdir)
        if not os.path.isdir(pdir):
            continue

        # Determine label from prefix
        if dirname.startswith('P'):
            label = 1  # ASD
            label_name = "ASD"
        elif dirname.startswith('S'):
            label = 0  # Non-ASD / Control
            label_name = "Control"
        else:
            continue

        # Process walk task only
        for task in ['walk']:
            task_dir = os.path.join(pdir, task)
            if not os.path.isdir(task_dir):
                continue

            # Find all mdata files
            mdata_files = sorted(glob.glob(
                os.path.join(task_dir, f'mdata_{dirname}{task}*.mat')
            ))

            for mat_file in mdata_files:
                mat_basename = os.path.splitext(os.path.basename(mat_file))[0]

                tqdm.write(f"  {mat_basename} ({label_name})")

                sequences = convert_mat_file(mat_file, config)

                if len(sequences) == 0:
                    tqdm.write(f"    [FAIL] No valid sequences")
                    failed += 1
                    continue

                for trial_idx, seq in enumerate(sequences):
                    video_id = f"m4a_{mat_basename}_t{trial_idx}"

                    # Skip if already converted
                    if video_id in existing_ids:
                        tqdm.write(f"    [SKIP] Already exists: {video_id}")
                        continue

                    npy_path = os.path.join(features_dir, f"{video_id}.npy")
                    np.save(npy_path, seq)

                    new_entries.append({
                        'video_id': video_id,
                        'label': label
                    })
                    total_sequences += 1

                    if label == 1:
                        asd_count += 1
                    else:
                        control_count += 1

                tqdm.write(f"    [OK] {len(sequences)} trials saved")

    # Merge with existing labels and write
    all_labels = existing_labels + new_entries

    with open(labels_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['video_id', 'label'])
        writer.writeheader()
        writer.writerows(all_labels)

    print(f"\n{'='*60}")
    print(f"Conversion Complete")
    print(f"{'='*60}")
    print(f"  New sequences added: {total_sequences}")
    print(f"    ASD (P*):         {asd_count}")
    print(f"    Control (S*):     {control_count}")
    print(f"  Failed files:       {failed}")
    print(f"  Total in labels.csv: {len(all_labels)}")
    print(f"  Features dir:       {features_dir}")
    print(f"  Labels file:        {labels_path}")
    print(f"{'='*60}")


# ────────────────────────────────────────────────────────────
# CLI Entry Point
# ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert Move4AS motion capture dataset to ASDMotion format'
    )
    parser.add_argument(
        '--dataset', type=str, required=True,
        help='Path to Move4AS Dataset folder'
    )
    parser.add_argument(
        '--config', type=str, default='configs/config.yaml',
        help='Path to ASDMotion configuration YAML file'
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    run_conversion(args.dataset, config)
