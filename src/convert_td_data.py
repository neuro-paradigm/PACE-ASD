import os
import glob
import json
import csv
import numpy as np
from scipy.signal import savgol_filter, resample
from tqdm import tqdm
import yaml

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

# BODY_25 keypoint indices
B25 = {
    'nose': 0, 'neck': 1, 'right_shoulder': 2, 'right_elbow': 3, 'right_wrist': 4,
    'left_shoulder': 5, 'left_elbow': 6, 'left_wrist': 7, 'mid_hip': 8, 'right_hip': 9,
    'right_knee': 10, 'right_ankle': 11, 'left_hip': 12, 'left_knee': 13, 'left_ankle': 14,
    'right_eye': 15, 'left_eye': 16, 'right_ear': 17, 'left_ear': 18, 'left_big_toe': 19,
    'left_small_toe': 20, 'left_heel': 21, 'right_big_toe': 22, 'right_small_toe': 23, 'right_heel': 24
}

def map_body25_to_mediapipe(kp, scores):
    """
    Map (T, 25, 2) BODY_25 keypoints to (T, 33, 3) MediaPipe landmarks.
    """
    T = kp.shape[0]
    
    # Zero out low-confidence keypoints
    kp_norm = kp.copy()
    low_conf = scores < 0.2
    kp_norm[low_conf] = 0.0

    mp = np.zeros((T, 33, 3), dtype=np.float32)

    # Direct mappings (BODY_25 -> MediaPipe)
    direct_map = {
        B25['nose']: MP['nose'],
        B25['left_eye']: MP['left_eye'],
        B25['right_eye']: MP['right_eye'],
        B25['left_ear']: MP['left_ear'],
        B25['right_ear']: MP['right_ear'],
        B25['left_shoulder']: MP['left_shoulder'],
        B25['right_shoulder']: MP['right_shoulder'],
        B25['left_elbow']: MP['left_elbow'],
        B25['right_elbow']: MP['right_elbow'],
        B25['left_wrist']: MP['left_wrist'],
        B25['right_wrist']: MP['right_wrist'],
        B25['left_hip']: MP['left_hip'],
        B25['right_hip']: MP['right_hip'],
        B25['left_knee']: MP['left_knee'],
        B25['right_knee']: MP['right_knee'],
        B25['left_ankle']: MP['left_ankle'],
        B25['right_ankle']: MP['right_ankle'],
        B25['left_heel']: MP['left_heel'],
        B25['right_heel']: MP['right_heel'],
        B25['left_big_toe']: MP['left_foot_index'],
        B25['right_big_toe']: MP['right_foot_index'],
    }
    for b25_idx, mp_idx in direct_map.items():
        mp[:, mp_idx, 0] = kp_norm[:, b25_idx, 0]  # x
        mp[:, mp_idx, 1] = kp_norm[:, b25_idx, 1]  # y

    # Approximated landmarks
    nose = kp_norm[:, B25['nose']]
    l_eye = kp_norm[:, B25['left_eye']]
    r_eye = kp_norm[:, B25['right_eye']]
    l_ear = kp_norm[:, B25['left_ear']]
    r_ear = kp_norm[:, B25['right_ear']]
    l_sho = kp_norm[:, B25['left_shoulder']]
    r_sho = kp_norm[:, B25['right_shoulder']]
    l_wrist = kp_norm[:, B25['left_wrist']]
    r_wrist = kp_norm[:, B25['right_wrist']]
    l_elbow = kp_norm[:, B25['left_elbow']]
    r_elbow = kp_norm[:, B25['right_elbow']]

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

    # Use pixel length offset (~10-20 pixels) scaled by forearm len if needed, 
    # but since this happens before spatial recon, we can just use 15% of forearm.
    hand_ext = forearm_len_l * 0.25 
    hand_ext_r = forearm_len_r * 0.25

    mp[:, MP['left_index'], :2] = l_wrist + l_dir * hand_ext
    mp[:, MP['right_index'], :2] = r_wrist + r_dir * hand_ext_r
    mp[:, MP['left_pinky'], :2] = l_wrist + l_dir * hand_ext * 0.8 + l_perp * (hand_ext * 0.15)
    mp[:, MP['right_pinky'], :2] = r_wrist + r_dir * hand_ext_r * 0.8 - r_perp * (hand_ext_r * 0.15)
    mp[:, MP['left_thumb'], :2] = l_wrist + l_dir * hand_ext * 0.6 - l_perp * (hand_ext * 0.15)
    mp[:, MP['right_thumb'], :2] = r_wrist + r_dir * hand_ext_r * 0.6 + r_perp * (hand_ext_r * 0.15)

    return mp

def compute_hip_center(landmarks):
    left_hip = landmarks[23]
    right_hip = landmarks[24]
    return (left_hip + right_hip) / 2.0

def spatial_reconstruct_frame(landmarks):
    landmarks = landmarks.copy()
    hip_center = compute_hip_center(landmarks)
    landmarks -= hip_center
    shoulder_center = (landmarks[11] + landmarks[12]) / 2.0
    torso_length = np.linalg.norm(shoulder_center)
    min_torso_length = 0.05 
    if torso_length > min_torso_length:
        landmarks /= torso_length
    return landmarks

def process_clip(kp, scores, window_size=300, stride=150):
    T = kp.shape[0]
    valid_mask = np.any(kp != 0, axis=(1, 2))
    n_valid = np.sum(valid_mask)
    if n_valid < 10:
        return []

    mp_full = map_body25_to_mediapipe(kp, scores)

    for t in range(mp_full.shape[0]):
        if np.any(mp_full[t] != 0):
            mp_full[t] = spatial_reconstruct_frame(mp_full[t])

    for j in range(33):
        for c in range(3):
            signal = mp_full[:, j, c]
            nonzero = np.where(signal != 0)[0]
            if len(nonzero) < 2:
                continue
            zero_idx = np.where(signal == 0)[0]
            zero_in_range = zero_idx[(zero_idx > nonzero[0]) & (zero_idx < nonzero[-1])]
            if len(zero_in_range) > 0:
                signal[zero_in_range] = np.interp(zero_in_range, nonzero, signal[nonzero])
                mp_full[:, j, c] = signal

    min_frames_for_smooth = 15
    if T >= min_frames_for_smooth:
        window = min(11, T if T % 2 == 1 else T - 1)
        if window >= 5:
            for j in range(33):
                for c in range(3):
                    mp_full[:, j, c] = savgol_filter(mp_full[:, j, c], window, polyorder=3)

    chunks = []
    if T <= window_size:
        resampled = np.zeros((window_size, 33, 3), dtype=np.float32)
        for j in range(33):
            for c in range(3):
                resampled[:, j, c] = resample(mp_full[:, j, c], window_size)
        chunks.append(resampled.astype(np.float32))
    else:
        for start in range(0, T - window_size + 1, stride):
            end = start + window_size
            chunks.append(mp_full[start:end, :, :].astype(np.float32))
        if (T - window_size) % stride != 0:
            chunks.append(mp_full[T-window_size:, :, :].astype(np.float32))

    return chunks

def run_conversion():
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    processed_dir = config['data']['processed_dir']
    features_dir = os.path.join(processed_dir, 'features')
    labels_path = os.path.join(processed_dir, 'labels.csv')
    target_frames = config['data']['max_frames']

    os.makedirs(features_dir, exist_ok=True)

    existing_labels = []
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            reader = csv.DictReader(f)
            existing_labels = list(reader)
            
    existing_ids = set(entry['video_id'] for entry in existing_labels)
    
    new_entries = []
    td_count = 0

    json_files = []
    for d in ['D:\\A_filtered\\A', 'D:\\B_filtered\\B', 'D:\\C_filtered\\C']:
        if os.path.exists(d):
            json_files.extend(glob.glob(os.path.join(d, '*.json')))

    print(f"Found {len(json_files)} TD json files.")
    
    for jpath in tqdm(json_files, desc="Converting TD clips"):
        with open(jpath, 'r') as f:
            data = json.load(f)['data']
            
        filename = os.path.basename(jpath)
        base_id = filename.replace('.json', '')
        
        kp_list = []
        score_list = []
        
        for frame in data:
            skels = frame.get('skeleton', [])
            if not skels:
                kp_list.append(np.zeros((25, 2)))
                score_list.append(np.zeros(25))
                continue
                
            # Get skeleton with highest mean score
            best_sk = max(skels, key=lambda s: np.mean(s.get('score', [0])))
            
            pose = np.array(best_sk['pose']).reshape(25, 2)
            score = np.array(best_sk['score'])
            
            kp_list.append(pose)
            score_list.append(score)
            
        if len(kp_list) < 10:
            continue
            
        kp = np.stack(kp_list)
        scores = np.stack(score_list)
        
        chunks = process_clip(kp, scores, window_size=target_frames, stride=target_frames//2)
        
        for idx, seq in enumerate(chunks):
            video_id = f"td_{base_id}_c{idx}"
            if video_id in existing_ids:
                continue

            npy_path = os.path.join(features_dir, f"{video_id}.npy")
            np.save(npy_path, seq)

            new_entries.append({'video_id': video_id, 'label': 0})
            td_count += 1

    all_labels = existing_labels + new_entries
    with open(labels_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['video_id', 'label'])
        writer.writeheader()
        writer.writerows(all_labels)

    print(f"Added {td_count} new TD clips.")

if __name__ == '__main__':
    run_conversion()
