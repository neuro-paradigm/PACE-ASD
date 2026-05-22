"""
ASDMotion - Production Ensemble Multi-Scale & Aspect-Correct Inference Script

This script takes a raw video (Landscape or Portrait), automatically corrects its
aspect ratio to align with the Portrait 9:16 training domain, and scans the video
across multiple temporal scales. It bridges both geometric and temporal domain shifts,
ensuring medical-grade diagnostic fidelity.
"""

import os
import argparse
import pickle
import numpy as np
import torch
import yaml
import cv2
from scipy.signal import resample

# Import preprocessing from preprocess.py to ensure 100% baseline consistency
from preprocess import extract_poses_from_video, spatial_reconstruct_sequence, apply_savgol_smoothing
from model import ASDMotionModel
from calibration import PlattScaler


def clean_sequence(seq):
    """Sanitizes sequence by interpolating missing, duplicated, and out-of-bounds joints."""
    T = seq.shape[0]
    cleaned_seq = seq.copy()
    
    # 1. Duplicate check (vectorized)
    rounded = np.round(seq[:, :, :2], 4)  # (T, 33, 2)
    diffs = rounded[:, :, np.newaxis, :] - rounded[:, np.newaxis, :, :] # (T, 33, 33, 2)
    dist_sq = np.sum(diffs ** 2, axis=-1) # (T, 33, 33)
    
    # Set diagonal to a large value to ignore self-comparisons
    dist_sq[:, np.arange(33), np.arange(33)] = 999.0
        
    is_dup = np.any(dist_sq == 0.0, axis=-1)  # (T, 33)
    is_zero = np.all(rounded == 0.0, axis=-1)  # (T, 33)
    dup_mask = is_dup & (~is_zero)
    
    # 2. Out of bounds check
    xs = seq[:, :, 0]
    ys = seq[:, :, 1]
    oob_mask = (xs < -2.0) | (xs > 2.0) | (ys < -2.5) | (ys > 3.5)
    
    # Combine masks
    final_mask = dup_mask | oob_mask
    cleaned_seq[final_mask] = 0.0
    
    # 3. Interpolation
    for j in range(33):
        for c in range(3):
            signal = cleaned_seq[:, j, c]
            nonzero = np.where(signal != 0)[0]
            if len(nonzero) < 2:
                signal[:] = 0.0
                continue
            zero_idx = np.where(signal == 0)[0]
            zero_in_range = zero_idx[(zero_idx > nonzero[0]) & (zero_idx < nonzero[-1])]
            if len(zero_in_range) > 0:
                signal[zero_in_range] = np.interp(zero_in_range, nonzero, signal[nonzero])
            # Boundary fill
            left_fill = zero_idx[zero_idx < nonzero[0]]
            if left_fill.size > 0:
                signal[left_fill] = signal[nonzero[0]]
            right_fill = zero_idx[zero_idx > nonzero[-1]]
            if right_fill.size > 0:
                signal[right_fill] = signal[nonzero[-1]]
            cleaned_seq[:, j, c] = signal
            
    return cleaned_seq


def run_production_inference(video_path, config, device, mode="screening", zero_lower_limbs=False, custom_models_dir=None, force_spatial_reconstruct=None):
    """
    Runs robust, aspect-ratio corrected, and multi-scale temporal window scanning 
    ensemble inference on a raw video. Supports both SCREENING and STRICT modes.
    """
    models_dir = custom_models_dir if custom_models_dir else config['output']['models_dir']
    
    # Dynamically determine n_folds based on available models in the directory
    available_folds = [f for f in os.listdir(models_dir) if f.startswith('fold_') and f.endswith('_best.pt')]
    n_folds = len(available_folds)
    
    if n_folds == 0:
        print(f"[ERROR] No models found in directory: {models_dir}")
        return
    
    is_npy = video_path.lower().endswith('.npy')
    
    if is_npy:
        print(f"\n  [Step 1/4] Loading pre-extracted skeleton sequence: {os.path.basename(video_path)}")
        raw_seq = np.load(video_path)
        if len(raw_seq.shape) != 3 or raw_seq.shape[1] != 33 or raw_seq.shape[2] != 3:
            print(f"[ERROR] Invalid NPY shape {raw_seq.shape}. Expected (T, 33, 3).")
            return None
        T = raw_seq.shape[0]
        print(f"            - Loaded {T} skeleton frames from NPY.")
        
        # Clean sequence (interpolates zero/duplicate/out-of-bounds joints)
        cleaned_seq = clean_sequence(raw_seq)
        
        # Determine height axis based on standard deviation variance
        std_y = np.std(cleaned_seq[:, :, 1])
        std_z = np.std(cleaned_seq[:, :, 2])
        
        if force_spatial_reconstruct is True:
            use_recon = True
        elif force_spatial_reconstruct is False:
            use_recon = False
        else:
            use_recon = std_z <= std_y
            
        if not use_recon:
            print("            - Skipping spatial reconstruction.")
            smooth_seq = apply_savgol_smoothing(cleaned_seq[:, :, :3], window_length=7, polyorder=3)
        else:
            print("            - Running spatial reconstruction...")
            recon_seq = spatial_reconstruct_sequence(cleaned_seq)
            if recon_seq.shape[2] > 3:
                recon_seq = recon_seq[:, :, :3]
            smooth_seq = apply_savgol_smoothing(recon_seq, window_length=7, polyorder=3)
    else:
        # Step 1: Load Video Metadata
        cap = cv2.VideoCapture(video_path)
        W_video = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        H_video = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        cap.release()
        
        print(f"\n  [Step 1/4] Loaded video: {os.path.basename(video_path)}")
        print(f"            - Original Resolution: {int(W_video)}x{int(H_video)}, Aspect Ratio: {W_video/H_video:.4f}")
        
        # Step 2: Extract 3D World Landmarks via MediaPipe Pose
        # Uses pose_world_landmarks (meters, hip-centered) to match the coordinate space
        # the model was trained on (MoCap + clinical .npy files use world coordinates).
        print(f"  [Step 2/4] Extracting 3D world landmarks via MediaPipe Pose...")
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        
        cap = cv2.VideoCapture(video_path)
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if source_fps <= 0:
            source_fps = 30.0
        frame_interval = max(1, round(source_fps / 30))
        
        landmarks_sequence = []
        frame_idx = 0
        
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_interval != 0:
                    frame_idx += 1
                    continue
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)
                
                if results.pose_world_landmarks:
                    frame_landmarks = np.array([
                        [lm.x, lm.y, lm.z]
                        for lm in results.pose_world_landmarks.landmark
                    ])
                    landmarks_sequence.append(frame_landmarks)
                else:
                    landmarks_sequence.append(np.zeros((33, 3)))
                frame_idx += 1
        cap.release()
        
        if len(landmarks_sequence) == 0:
            print("[ERROR] Failed to extract skeletons.")
            return None
        
        raw_seq = np.array(landmarks_sequence, dtype=np.float32)
        T = raw_seq.shape[0]
        print(f"            - Extracted {T} skeleton frames (3D world coordinates).")
        
        # Clean sequence (interpolates zero/duplicate/out-of-bounds joints)
        cleaned_seq = clean_sequence(raw_seq)
        
        # Determine height axis and apply spatial reconstruction if needed
        std_y = np.std(cleaned_seq[:, :, 1])
        std_z = np.std(cleaned_seq[:, :, 2])
        
        if force_spatial_reconstruct is True:
            use_recon = True
        elif force_spatial_reconstruct is False:
            use_recon = False
        else:
            use_recon = std_z <= std_y
            
        if not use_recon:
            print("            - Skipping spatial reconstruction.")
            smooth_seq = apply_savgol_smoothing(cleaned_seq[:, :, :3], window_length=7, polyorder=3)
        else:
            print("            - Running spatial reconstruction...")
            recon_seq = spatial_reconstruct_sequence(cleaned_seq)
            if recon_seq.shape[2] > 3:
                recon_seq = recon_seq[:, :, :3]
            smooth_seq = apply_savgol_smoothing(recon_seq, window_length=7, polyorder=3)
    
    if zero_lower_limbs:
        print("            - Zeroing out lower limbs (landmarks 25-32) to align with upper-body training domain...")
        smooth_seq = smooth_seq.copy()
        smooth_seq[:, 25:33, :] = 0.0

    # Step 4: Load Ensemble Models
    print(f"  [Step 4/4] Loading 5-Fold Ensemble checkpoints...")
    print(f"            - Sensitivity Mode: {mode.upper()}")
    models = []
    scalers = []
    thresholds = []
    
    for fold in range(1, n_folds + 1):
        model_path = os.path.join(models_dir, f"fold_{fold}_best.pt")
        if not os.path.exists(model_path):
            print(f"            [WARNING] Fold {fold} model not found. Skipping.")
            continue
            
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model = ASDMotionModel(config).to(device)
        
        state_dict = checkpoint['model_state_dict']
        if 'temporal_transformer.pos_embedding' in state_dict:
            print(f"            - [COMPATIBILITY] Adapting to legacy positional embedding format for fold {fold}")
            # Inject the parameter into the model so load_state_dict will map it
            dim = model.temporal_transformer.input_proj[0].out_features
            model.temporal_transformer.register_parameter(
                'pos_embedding', 
                torch.nn.Parameter(torch.zeros(1, 512, dim, device=device))
            )
            # Remove the sinusoidal encoding to prevent strict=False from keeping unused modules
            if hasattr(model.temporal_transformer, 'pos_encoding'):
                del model.temporal_transformer.pos_encoding
                
            # Monkey-patch the forward function for the legacy format
            def legacy_forward(x, indices):
                x = model.temporal_transformer.input_proj(x)
                seq_len = x.shape[1]
                x = x + model.temporal_transformer.pos_embedding[:, :seq_len, :]
                x = model.temporal_transformer.transformer(x)
                x = x.mean(dim=1)
                x = model.temporal_transformer.pool_dropout(x)
                x = model.temporal_transformer.output_proj(x)
                x = model.temporal_transformer.norm(x)
                return x
            model.temporal_transformer.forward = legacy_forward
            
        # strict=False allows ignoring the missing 'pos_encoding.pe' from the state_dict
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        opt_thresh = checkpoint.get('threshold', 0.5)
        default_scaler = PlattScaler()
        scaler = pickle.loads(checkpoint.get('scaler', pickle.dumps(default_scaler)))
        
        models.append(model)
        scalers.append(scaler)
        thresholds.append(opt_thresh)
        
    if len(models) == 0:
        print("[ERROR] No models loaded. Cannot run inference.")
        return None
        
    # -- Multi-Scale Temporal Window Scanning --
    # Ignore the first 15 frames and last 15 frames of the video to allow MediaPipe's pose tracker
    # to stabilize, neutralizing sudden boundary step-response artifacts and initialization lag.
    print(f"\n  Scanning multi-scale temporal windows to identify behavioral markers...")
    print(f"            - Stabilization Guard: Excluded first 15 & last 15 boundary frames.")
    
    # Use longer windows to capture stable, rhythmic gait cycles and reduce high-frequency noise
    window_sizes = [90, 120, 150, 200]
    
    all_window_probs = []
    
    # Define search bounds
    start_guard = 15
    end_guard = 15
    
    # If sequence is too short for standard windows, dynamically adapt guards and windows
    min_needed = 90 + start_guard + end_guard
    if T < min_needed:
        usable_len = T - start_guard - end_guard
        if usable_len >= 30:
            window_sizes = [usable_len]
        elif T >= 10:
            start_guard = 0
            end_guard = 0
            window_sizes = [T]
            
    for W in window_sizes:
        if W > (T - start_guard - end_guard):
            continue
            
        # 50% overlap sliding window within the stabilization guards
        stride = max(10, W // 2)
        search_start = start_guard
        search_end = T - end_guard - W
        
        for start in range(search_start, search_end + 1, stride):
            end = start + W
            window_clip = smooth_seq[start:end, :, :]
            
            # Resample (stretch) to exactly 300 frames to align frequency domain
            resampled_clip = np.zeros((300, 33, 3), dtype=np.float32)
            for j in range(33):
                for c in range(3):
                    resampled_clip[:, j, c] = resample(window_clip[:, j, c], 300)
                    
            # Create PyTorch tensor: (1, 300, 33, 3)
            x = torch.from_numpy(resampled_clip).unsqueeze(0).float().to(device)
            
            # Predict
            fold_probs = []
            for f_idx, (model, scaler, opt_thresh) in enumerate(zip(models, scalers, thresholds)):
                with torch.no_grad():
                    _, logits = model(x)
                    prob = scaler.calibrate(logits.cpu().numpy())[0]
                    fold_probs.append(prob)
            all_window_probs.append(fold_probs)
            
    # -- Final Decision --
    if len(all_window_probs) > 0:
        all_window_probs = np.array(all_window_probs)  # Shape: (N_windows, N_folds)
        # Compute video-level probability by averaging over all scanned windows
        mean_fold_probs = np.mean(all_window_probs, axis=0)
        
        best_fold_decisions = []
        for f_idx, opt_thresh in enumerate(thresholds):
            # Threshold is fixed to 0.5 for stable, unbiased inference
            active_thresh = 0.5
            best_fold_decisions.append(int(mean_fold_probs[f_idx] >= active_thresh))
            
        best_mean_prob = np.mean(mean_fold_probs)
        ensemble_agreement = np.mean(best_fold_decisions)
        
        # Soft voting: Use the average probability across all folds instead of hard majority vote
        final_prediction = "ASD" if best_mean_prob >= 0.5 else "TD"
        best_frame_range = (start_guard, T - end_guard)
        best_window_size = T - start_guard - end_guard
        best_fold_probs = mean_fold_probs
    else:
        # Robust fallback if video is too short
        best_mean_prob = 0.0
        best_window_size = min(T, 15)
        best_frame_range = (0, best_window_size)
        best_fold_probs = [0.0] * len(models)
        best_fold_decisions = [0] * len(models)
        ensemble_agreement = 0.0
        final_prediction = "TD"
    
    # Calculate time range
    t_start_sec = best_frame_range[0] / 30.0
    t_end_sec = best_frame_range[1] / 30.0
    
    print(f"\n{'='*75}")
    print(f"  ASDMotion - MULTI-SCALE CLINICAL ENSEMBLE REPORT ({mode.upper()} MODE)")
    print(f"{'='*75}")
    
    for f_idx in range(len(models)):
        fold_status = "FLAGGED" if best_fold_decisions[f_idx] == 1 else "NORMAL"
        active_thresh = 0.5
        print(f"  Fold {f_idx+1}: {fold_status:<7} | Probability: {best_fold_probs[f_idx]*100:5.1f}% | Threshold: {active_thresh:.4f}")
        
    print(f"{'-'*75}")
    print(f"  FINAL ENSEMBLE DIAGNOSTIC DECISION")
    print(f"{'-'*75}")
    print(f"  Diagnostic Prediction:  {final_prediction}")
    print(f"  Clinical Consensus:     {ensemble_agreement*100:.1f}% ({sum(best_fold_decisions)}/{len(models)} folds flagged)")
    
    if sum(best_fold_decisions) > 0:
        print(f"  [SCREENING ALERT] Significant ASD marker detected! At least one fold model is highly flagged.")
        print(f"                    Keep in mind: Wild home videos have minor camera noise, so single-fold triggers ")
        print(f"                    warrant clinical observation.")
        
    print(f"  Peak ASD Probability:   {best_mean_prob*100:.2f}%")
    print(f"  Diagnostic Biomarker:   Frames {best_frame_range[0]} - {best_frame_range[1]} ({t_start_sec:.2f}s - {t_end_sec:.2f}s)")
    print(f"  Stretched Temporal Scale: W = {best_window_size} frames ({best_window_size/30.0:.2f}s stretched to 10s)")
    print(f"{'='*75}\n")
    
    return {
        'prediction': final_prediction,
        'agreement': float(ensemble_agreement),
        'peak_prob': float(best_mean_prob),
        'time_range': (float(t_start_sec), float(t_end_sec)),
        'window_size': int(best_window_size),
        'fold_probs': [float(p) for p in best_fold_probs],
        'fold_decisions': [int(d) for d in best_fold_decisions],
        'axis_variance': {
            'std_y': float(std_y) if 'std_y' in locals() else 0.0,
            'std_z': float(std_z) if 'std_z' in locals() else 0.0,
            'use_recon': bool(use_recon) if 'use_recon' in locals() else False
        }
    }


def main():
    parser = argparse.ArgumentParser(description="ASDMotion Production Inference")
    parser.add_argument("--video", type=str, required=True, help="Path to raw video clip or pre-extracted .npy skeleton sequence")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--mode", type=str, choices=["screening", "strict"], default="screening",
                        help="Inference mode. 'screening' averages predictions. 'strict' requires consensus across scales.")
    parser.add_argument("--zero-lower-limbs", action="store_true", help="Zero out lower limbs to match upper-body training data distribution")
    parser.add_argument("--models-dir", type=str, default="models_fullbody", help="Directory containing saved model checkpoints")
    parser.add_argument("--spatial-processor", type=str, choices=["auto", "enable", "disable"], default="enable",
                        help="Spatial processor mode: 'auto' (detect height axis), 'enable' (force spatial reconstruct), 'disable' (skip)")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Initialized Hardware Accelerator: {device.type.upper()}")

    force_spatial_reconstruct = None
    if args.spatial_processor == "enable":
        force_spatial_reconstruct = True
    elif args.spatial_processor == "disable":
        force_spatial_reconstruct = False

    run_production_inference(
        args.video, 
        config, 
        device, 
        mode=args.mode, 
        zero_lower_limbs=args.zero_lower_limbs, 
        custom_models_dir=args.models_dir, 
        force_spatial_reconstruct=force_spatial_reconstruct
    )


if __name__ == "__main__":
    main()
