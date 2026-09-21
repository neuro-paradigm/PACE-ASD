"""
PACE-ASD — Inference Pipeline

Run the full PACE-ASD pipeline on a single video or pre-extracted feature file.

Usage (raw video):
    python scripts/infer.py \\
        --input path/to/video.mp4 \\
        --checkpoint models/A1/fold1_seed42.pt \\
        --config configs/inference.yaml \\
        --output outputs/result

Usage (pre-extracted .npy, no MediaPipe required):
    python scripts/infer.py \\
        --input_npy processed/features/asd_1.npy \\
        --checkpoint models/A1/fold1_seed42.pt \\
        --config configs/inference.yaml \\
        --output outputs/result

Outputs:
    <output>/
    ├── result.json              # Prediction, probabilities, metadata
    ├── kinematics.npz           # positions, velocities, accelerations (T,33,2 each)
    ├── selected_events.json     # Block-ESG selected temporal events
    ├── attribution.json         # Body-region and kinematic-stream attribution
    └── visualizations/
        ├── kinematics.png       # Position/velocity/acceleration trajectories
        ├── event_saliency.png   # Block-ESG saliency heatmap
        └── evidence_summary.png # Multi-panel evidence summary
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import yaml

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Project imports
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

from model import ASDMotionModel
from calibration import PlattScaler


# ── Constants (must match preprocessing) ─────────────────────────────────────

N_LANDMARKS  = 33
T_MAX        = 300
LEFT_HIP     = 23
RIGHT_HIP    = 24
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12

JOINT_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "left_mouth", "right_mouth",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

BODY_REGIONS = {
    "head":  list(range(0, 11)),
    "arms":  list(range(11, 23)),
    "torso": [23, 24],
    "legs":  list(range(25, 33)),
}


# ── Video-based pose extraction ───────────────────────────────────────────────

def extract_pose_from_video(video_path: str, cfg: dict) -> np.ndarray:
    """
    Run MediaPipe Pose on a video file.
    Returns (T_actual, 33, 2) float32. Zeros where detection failed.
    """
    import cv2
    import warnings
    import os as _os
    _os.environ["GLOG_minloglevel"] = "2"
    warnings.filterwarnings("ignore")
    import mediapipe as mp

    prep = cfg.get("preprocessing", {})
    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=prep.get("mediapipe_model_complexity", 2),
        smooth_landmarks=prep.get("mediapipe_smooth_landmarks", True),
        enable_segmentation=False,
        min_detection_confidence=prep.get("mediapipe_min_detection_confidence", 0.5),
        min_tracking_confidence=prep.get("mediapipe_min_tracking_confidence", 0.5),
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
            )
        else:
            kp = np.zeros((N_LANDMARKS, 2), dtype=np.float32)
        frames.append(kp)

    cap.release()
    pose.close()

    if not frames:
        return np.zeros((1, N_LANDMARKS, 2), dtype=np.float32)
    return np.stack(frames, axis=0)


def normalise(keypoints: np.ndarray) -> np.ndarray:
    """Hip-centering + inter-shoulder scale normalisation. (T, 33, 2) -> (T, 33, 2)."""
    kp = keypoints.copy()
    for t in range(kp.shape[0]):
        if not np.any(kp[t] != 0):
            continue
        mid_hip = (kp[t, LEFT_HIP] + kp[t, RIGHT_HIP]) / 2.0
        kp[t] -= mid_hip
        shoulder_dist = float(np.linalg.norm(kp[t, LEFT_SHOULDER] - kp[t, RIGHT_SHOULDER]))
        shoulder_dist = max(shoulder_dist, 1e-5)
        kp[t] /= shoulder_dist
    return kp


def pad_or_truncate(keypoints: np.ndarray, target_len: int = T_MAX) -> np.ndarray:
    """Pad or truncate to exactly target_len frames."""
    T = keypoints.shape[0]
    if T >= target_len:
        return keypoints[:target_len]
    pad = np.zeros((target_len - T, N_LANDMARKS, 2), dtype=np.float32)
    return np.concatenate([keypoints, pad], axis=0)


# ── Kinematic feature extraction ──────────────────────────────────────────────

def compute_kinematics(positions: np.ndarray):
    """
    Compute per-frame velocity and acceleration from normalised position array.

    Matches SpatialEncoder runtime computation (vel scaled ×10, acc scaled ×5).

    Args:
        positions: (T, 33, 2) float32

    Returns:
        velocities:     (T, 33, 2) float32
        accelerations:  (T, 33, 2) float32
    """
    T = positions.shape[0]
    valid_mask = np.abs(positions).sum(axis=(1, 2)) > 1e-4  # (T,)

    velocities = np.zeros_like(positions)
    for t in range(1, T):
        if valid_mask[t] and valid_mask[t - 1]:
            velocities[t] = (positions[t] - positions[t - 1]) * 10.0

    accelerations = np.zeros_like(positions)
    for t in range(2, T):
        if valid_mask[t] and valid_mask[t - 1] and valid_mask[t - 2]:
            accelerations[t] = (velocities[t] - velocities[t - 1]) * 5.0

    return velocities, accelerations


# ── Attribution ───────────────────────────────────────────────────────────────

def compute_attribution(model: ASDMotionModel, x: torch.Tensor,
                        device: torch.device) -> dict:
    """
    Gradient × Input attribution for body-region and kinematic-stream decomposition.

    Returns dict with:
        body_region_attribution: {head, arms, torso, legs} -> float (fraction)
        kinematic_stream_attribution: {position, velocity, acceleration} -> float (fraction)
        joint_attribution: list of 33 floats (per-joint importance)
    """
    model.eval()
    x_req = x.clone().requires_grad_(True)

    probs, logits = model(x_req)
    logits.sum().backward()

    if x_req.grad is None:
        return {
            "body_region_attribution": {r: 0.0 for r in BODY_REGIONS},
            "kinematic_stream_attribution": {"position": 0.0, "velocity": 0.0, "acceleration": 0.0},
            "joint_attribution": [0.0] * N_LANDMARKS,
        }

    # Gradient × Input: (B=1, T, 33, 2)
    grad_input = (x_req.grad * x_req).squeeze(0).detach().cpu().numpy()  # (T, 33, 2)
    importance = np.abs(grad_input).sum(axis=(0, 2))  # (33,) per-joint
    total = importance.sum() + 1e-10

    # Body-region attribution
    body_attr = {}
    for region, joints in BODY_REGIONS.items():
        body_attr[region] = float(importance[joints].sum() / total)

    # Kinematic-stream attribution requires recomputing with velocity/acceleration streams
    # We approximate by computing attribution on the three sub-tensors inside SpatialEncoder:
    # The SpatialEncoder concatenates [pos | vel | acc] at dim=-1 over (B, T, 33, 2) inputs.
    # Since the input to infer.py is positions only, we derive vel/acc same way as the model.
    T = x.shape[1]
    pos_np = x.squeeze(0).detach().cpu().numpy()  # (T, 33, 2)
    valid = np.abs(pos_np).sum(axis=(1, 2)) > 1e-4  # (T,)

    vel_np = np.zeros_like(pos_np)
    for t in range(1, T):
        if valid[t] and valid[t-1]:
            vel_np[t] = (pos_np[t] - pos_np[t-1]) * 10.0

    acc_np = np.zeros_like(vel_np)
    for t in range(2, T):
        if valid[t] and valid[t-1] and valid[t-2]:
            acc_np[t] = (vel_np[t] - vel_np[t-1]) * 5.0

    grad_np = x_req.grad.squeeze(0).detach().cpu().numpy()  # (T, 33, 2)
    pos_attr = float(np.abs(grad_np * pos_np).sum())
    vel_attr = float(np.abs(grad_np * vel_np).sum())  # approximate
    acc_attr = float(np.abs(grad_np * acc_np).sum())  # approximate
    stream_total = pos_attr + vel_attr + acc_attr + 1e-10

    return {
        "body_region_attribution": {k: round(v, 4) for k, v in body_attr.items()},
        "kinematic_stream_attribution": {
            "position":     round(pos_attr / stream_total, 4),
            "velocity":     round(vel_attr / stream_total, 4),
            "acceleration": round(acc_attr / stream_total, 4),
        },
        "joint_attribution": [round(float(importance[j] / total), 6) for j in range(N_LANDMARKS)],
    }


# ── Asymmetry computation ─────────────────────────────────────────────────────

def compute_bilateral_asymmetry(positions: np.ndarray) -> dict:
    """
    Compute descriptive bilateral movement asymmetry from left/right landmark pairs.
    This is a descriptive movement measure, NOT a validated ASD biomarker.

    Left/right pairs (MediaPipe Pose landmarks):
        shoulder (11,12), elbow (13,14), wrist (15,16),
        hip (23,24), knee (25,26), ankle (27,28)

    Returns:
        dict with per-joint-pair mean absolute left-right movement difference.
    """
    LR_PAIRS = [
        ("shoulder", 11, 12),
        ("elbow",    13, 14),
        ("wrist",    15, 16),
        ("hip",      23, 24),
        ("knee",     25, 26),
        ("ankle",    27, 28),
    ]
    T = positions.shape[0]
    valid = np.abs(positions).sum(axis=(1, 2)) > 1e-4  # (T,)
    valid_positions = positions[valid]  # (T_valid, 33, 2)

    asymmetry = {}
    if len(valid_positions) < 2:
        for name, _, _ in LR_PAIRS:
            asymmetry[name] = 0.0
        return {"pairs": asymmetry, "mean_asymmetry": 0.0,
                "note": "Descriptive bilateral movement asymmetry — not a validated ASD biomarker"}

    for name, left_idx, right_idx in LR_PAIRS:
        left_pos  = valid_positions[:, left_idx, :]   # (T_valid, 2)
        right_pos = valid_positions[:, right_idx, :]  # (T_valid, 2)
        # Mirror right X to make positions comparable after hip-centering
        right_mirrored = right_pos.copy()
        right_mirrored[:, 0] = -right_mirrored[:, 0]
        diff = np.abs(left_pos - right_mirrored).mean()
        asymmetry[name] = round(float(diff), 6)

    mean_asym = np.mean(list(asymmetry.values()))
    return {
        "pairs": asymmetry,
        "mean_asymmetry": round(float(mean_asym), 6),
        "note": "Descriptive bilateral movement asymmetry (left-right movement difference) — not a validated ASD biomarker",
    }


# ── Visualization ─────────────────────────────────────────────────────────────

def save_visualizations(output_dir: str, positions: np.ndarray,
                        velocities: np.ndarray, accelerations: np.ndarray,
                        block_scores, selected_indices, result: dict) -> None:
    """Generate and save diagnostic visualization PNGs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    T = positions.shape[0]
    t_axis = np.arange(T)
    valid_mask = np.abs(positions).sum(axis=(1, 2)) > 1e-4

    # ── 1. Kinematics plot ────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("PACE-ASD: Kinematic Trajectories", fontsize=14, fontweight="bold")

    # Position — mean across all valid joints
    pos_mag = np.linalg.norm(positions, axis=-1)  # (T, 33)
    pos_mean = np.where(valid_mask[:, None], pos_mag, np.nan).mean(axis=1)
    axes[0].plot(t_axis, pos_mean, color="#2196F3", linewidth=1.2, label="Mean position magnitude")
    axes[0].set_ylabel("Position (normalised)")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].set_title("Position Trajectories")

    # Velocity — mean magnitude
    vel_mag = np.linalg.norm(velocities, axis=-1).mean(axis=1)  # (T,)
    vel_mag[~valid_mask] = np.nan
    axes[1].plot(t_axis, vel_mag, color="#4CAF50", linewidth=1.2, label="Mean velocity magnitude")
    axes[1].set_ylabel("Velocity (scaled)")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].set_title("Velocity Trajectories")

    # Acceleration — mean magnitude
    acc_mag = np.linalg.norm(accelerations, axis=-1).mean(axis=1)  # (T,)
    acc_mag[~valid_mask] = np.nan
    axes[2].plot(t_axis, acc_mag, color="#E91E63", linewidth=1.2, label="Mean acceleration magnitude")
    axes[2].set_ylabel("Acceleration (scaled)")
    axes[2].set_xlabel("Frame")
    axes[2].legend(loc="upper right", fontsize=9)
    axes[2].set_title("Acceleration Trajectories")

    # Mark selected events if available
    if selected_indices is not None:
        for ax in axes:
            for idx in selected_indices:
                if 0 <= idx < T:
                    ax.axvline(idx, color="orange", alpha=0.15, linewidth=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "kinematics.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: visualizations/kinematics.png")

    # ── 2. Event saliency plot ────────────────────────────────────────────────
    if block_scores is not None:
        fig, ax = plt.subplots(figsize=(12, 4))
        fig.suptitle("PACE-ASD: Block-ESG Event Saliency", fontsize=14, fontweight="bold")

        block_scores_np = np.array(block_scores)
        n_blocks = len(block_scores_np)
        block_centers = np.arange(n_blocks)
        bars = ax.bar(block_centers, block_scores_np,
                      color=["#FF9800" if s >= np.sort(block_scores_np)[-min(8, n_blocks)]
                             else "#90CAF9" for s in block_scores_np],
                      edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Block index (15-frame blocks)")
        ax.set_ylabel("Gate saliency score")
        ax.set_title("Block-ESG Saliency Scores (orange = selected top-M blocks)")
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "event_saliency.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: visualizations/event_saliency.png")

    # ── 3. Evidence summary (multi-panel) ────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"PACE-ASD Evidence Summary\n"
        f"Raw P(ASD) = {result['raw_probability']:.3f}  |  "
        f"Calibrated P(ASD) = {result['calibrated_probability']:.3f}  |  "
        f"Prediction = {'ASD' if result['calibrated_probability'] >= 0.5 else 'TD'}",
        fontsize=13, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # Panel A: position mag
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(t_axis, pos_mean, color="#2196F3", linewidth=1.0)
    ax_a.set_title("(A) Position trajectory", fontsize=10)
    ax_a.set_xlabel("Frame"); ax_a.set_ylabel("Magnitude")
    if selected_indices is not None:
        for idx in selected_indices:
            if 0 <= idx < T:
                ax_a.axvline(idx, color="orange", alpha=0.2, linewidth=0.7)

    # Panel B: velocity
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.plot(t_axis, vel_mag, color="#4CAF50", linewidth=1.0)
    ax_b.set_title("(B) Velocity trajectory", fontsize=10)
    ax_b.set_xlabel("Frame"); ax_b.set_ylabel("Magnitude")

    # Panel C: body-region attribution
    ax_c = fig.add_subplot(gs[1, 0])
    if "body_region_attribution" in result:
        regions = list(result["body_region_attribution"].keys())
        values  = list(result["body_region_attribution"].values())
        colors  = ["#E91E63", "#FF9800", "#9C27B0", "#2196F3"]
        ax_c.barh(regions, values, color=colors[:len(regions)])
        ax_c.set_xlim(0, 1)
        ax_c.set_xlabel("Attribution fraction")
        ax_c.set_title("(C) Body-region attribution", fontsize=10)

    # Panel D: kinematic-stream attribution
    ax_d = fig.add_subplot(gs[1, 1])
    if "kinematic_stream_attribution" in result:
        streams = list(result["kinematic_stream_attribution"].keys())
        svals   = list(result["kinematic_stream_attribution"].values())
        stream_colors = ["#2196F3", "#4CAF50", "#E91E63"]
        ax_d.bar(streams, svals, color=stream_colors)
        ax_d.set_ylim(0, 1)
        ax_d.set_ylabel("Attribution fraction")
        ax_d.set_title("(D) Kinematic-stream attribution", fontsize=10)

    plt.savefig(os.path.join(viz_dir, "evidence_summary.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: visualizations/evidence_summary.png")


# ── Main inference pipeline ───────────────────────────────────────────────────

def load_checkpoint(checkpoint_path: str, config: dict, device: torch.device):
    """Load model and optional PlattScaler from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = ckpt.get("config", config)

    # Determine variant from checkpoint config or inference config
    inf_cfg = config.get("inference", {})
    use_gate        = inf_cfg.get("use_gate", True)
    use_transformer = inf_cfg.get("use_transformer", True)

    model = ASDMotionModel(model_cfg, use_gate=use_gate, use_transformer=use_transformer)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    scaler = None
    if "scaler" in ckpt and ckpt["scaler"] is not None:
        try:
            scaler = pickle.loads(ckpt["scaler"])
        except Exception as e:
            print(f"  [WARN] Could not load Platt scaler: {e}")

    return model, scaler, ckpt


def run_inference(args):
    """Main inference entry point."""
    t_start = time.time()

    # ── Load configuration ────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Device
    inf_cfg = cfg.get("inference", {})
    device_str = inf_cfg.get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"\n  Device: {device}")

    os.makedirs(args.output, exist_ok=True)

    # ── Feature extraction ────────────────────────────────────────────────────
    if args.input_npy:
        print(f"  Loading pre-extracted features: {args.input_npy}")
        positions = np.load(args.input_npy).astype(np.float32)  # (300, 33, 2)
        if positions.shape != (T_MAX, N_LANDMARKS, 2):
            raise ValueError(f"Expected shape ({T_MAX}, {N_LANDMARKS}, 2), got {positions.shape}")
        clip_id = os.path.splitext(os.path.basename(args.input_npy))[0]
        input_source = args.input_npy
    elif args.input:
        print(f"  Extracting pose from video: {args.input}")
        raw_kp  = extract_pose_from_video(args.input, cfg)
        normd   = normalise(raw_kp)
        positions = pad_or_truncate(normd)
        clip_id = os.path.splitext(os.path.basename(args.input))[0]
        input_source = args.input
    else:
        raise ValueError("Provide --input (video) or --input_npy (.npy file)")

    print(f"  Clip ID: {clip_id}")
    print(f"  Feature shape: {positions.shape}")

    # ── Compute kinematics ────────────────────────────────────────────────────
    velocities, accelerations = compute_kinematics(positions)

    # ── Save kinematics ───────────────────────────────────────────────────────
    if inf_cfg.get("save_kinematics", True):
        kinem_path = os.path.join(args.output, "kinematics.npz")
        np.savez_compressed(
            kinem_path,
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
            frame_indices=np.arange(T_MAX),
        )
        print(f"  Saved: kinematics.npz (positions, velocities, accelerations — shape {positions.shape})")

    # ── Bilateral asymmetry ───────────────────────────────────────────────────
    asymmetry = compute_bilateral_asymmetry(positions)

    # ── Model inference ───────────────────────────────────────────────────────
    print(f"\n  Loading checkpoint: {args.checkpoint}")
    model, scaler, ckpt = load_checkpoint(args.checkpoint, cfg, device)

    x = torch.from_numpy(positions).unsqueeze(0).to(device)  # (1, T, 33, 2)

    with torch.no_grad():
        probs, logits = model(x)
        raw_logit = float(logits[0].item())
        raw_prob  = float(probs[0].item())

    # Platt calibration
    if scaler is not None:
        cal_prob = float(scaler.calibrate(np.array([raw_logit]))[0])
    else:
        cal_prob = raw_prob
        print("  [WARN] No Platt scaler found — using raw sigmoid probability")

    threshold = inf_cfg.get("threshold", 0.5)
    prediction = "ASD" if cal_prob >= threshold else "TD"

    print(f"\n  -- Prediction --")
    print(f"  Raw probability     : {raw_prob:.4f}")
    print(f"  Calibrated P(ASD)   : {cal_prob:.4f}")
    print(f"  Decision (thr={threshold}) : {prediction}")

    # ── Block-ESG event information ───────────────────────────────────────────
    block_scores = None
    selected_indices_np = None
    selected_events = {}

    if model.saliency_gate is not None:
        with torch.no_grad():
            _, raw_ind, raw_bs = model.get_attention_maps(x)
            if raw_bs is not None:
                block_scores = raw_bs[0].cpu().numpy().tolist()
            if raw_ind is not None:
                selected_indices_np = raw_ind[0].cpu().numpy()

    if selected_indices_np is not None:
        L = model.saliency_gate.block_size if model.saliency_gate else 15
        M = model.saliency_gate.top_m if model.saliency_gate else 8
        # Recover block boundaries from selected frame indices
        block_starts = sorted(set(int(idx // L) * L for idx in selected_indices_np))
        selected_events = {
            "block_size_frames":   L,
            "num_selected_blocks": M,
            "selected_frame_indices": selected_indices_np.tolist(),
            "selected_block_starts": block_starts,
            "selected_block_ranges": [[s, s + L - 1] for s in block_starts],
        }
        if block_scores is not None:
            selected_events["all_block_saliency_scores"] = block_scores

        if inf_cfg.get("save_events", True):
            ev_path = os.path.join(args.output, "selected_events.json")
            with open(ev_path, "w") as f:
                json.dump(selected_events, f, indent=2)
            print(f"  Saved: selected_events.json ({len(block_starts)} blocks selected)")

    # ── Attribution ───────────────────────────────────────────────────────────
    attribution = {}
    if inf_cfg.get("save_attribution", True):
        try:
            x_attr = torch.from_numpy(positions).unsqueeze(0).to(device)
            attribution = compute_attribution(model, x_attr, device)
            attr_path = os.path.join(args.output, "attribution.json")
            with open(attr_path, "w") as f:
                json.dump(attribution, f, indent=2)
            print(f"  Saved: attribution.json")
            print(f"  Body-region attribution: {attribution['body_region_attribution']}")
            print(f"  Kinematic-stream attribution: {attribution['kinematic_stream_attribution']}")
        except Exception as e:
            print(f"  [WARN] Attribution failed: {e}")

    # ── Compile result ────────────────────────────────────────────────────────
    result = {
        "clip_id":               clip_id,
        "input_source":          input_source,
        "checkpoint":            os.path.abspath(args.checkpoint),
        "prediction":            prediction,
        "raw_probability":       round(raw_prob, 6),
        "calibrated_probability": round(cal_prob, 6),
        "threshold":             threshold,
        "has_platt_scaler":      scaler is not None,
        "body_region_attribution": attribution.get("body_region_attribution", {}),
        "kinematic_stream_attribution": attribution.get("kinematic_stream_attribution", {}),
        "bilateral_asymmetry":   asymmetry,
        "selected_events_summary": {
            "num_blocks_selected": len(selected_events.get("selected_block_starts", [])),
            "block_size_frames":   selected_events.get("block_size_frames", 15),
        },
        "runtime_seconds": round(time.time() - t_start, 2),
    }

    result_path = os.path.join(args.output, "result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: result.json")

    # ── Visualizations ────────────────────────────────────────────────────────
    if inf_cfg.get("save_visualizations", True):
        print("\n  Generating visualizations...")
        save_visualizations(
            args.output, positions, velocities, accelerations,
            block_scores, selected_indices_np, result,
        )

    print(f"\n  Done in {result['runtime_seconds']:.1f}s")
    print(f"  -- Result --")
    print(f"  Calibrated P(ASD): {cal_prob:.4f}")
    print(f"  Prediction:        {prediction}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="PACE-ASD: Run full inference pipeline on a video or feature file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pre-extracted .npy file (no MediaPipe required):
  python scripts/infer.py --input_npy processed/features/asd_1.npy \\
      --checkpoint models/A1/fold1_seed42.pt \\
      --config configs/inference.yaml \\
      --output outputs/asd1_result

  # Raw video file:
  python scripts/infer.py --input path/to/video.mp4 \\
      --checkpoint models/A1/fold1_seed42.pt \\
      --config configs/inference.yaml \\
      --output outputs/video_result
"""
    )
    parser.add_argument("--input",      type=str, default=None,
                        help="Path to raw video file (.mp4, .avi, etc.)")
    parser.add_argument("--input_npy",  type=str, default=None,
                        help="Path to pre-extracted feature file (.npy, shape 300×33×2)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint (.pt file)")
    parser.add_argument("--config",     type=str, default="configs/inference.yaml",
                        help="Path to inference config (default: configs/inference.yaml)")
    parser.add_argument("--output",     type=str, default="outputs/result",
                        help="Output directory (default: outputs/result)")
    args = parser.parse_args()

    if args.input is None and args.input_npy is None:
        parser.error("Provide --input (video) or --input_npy (.npy feature file)")

    run_inference(args)


if __name__ == "__main__":
    main()
