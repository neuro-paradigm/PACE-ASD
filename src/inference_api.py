"""
PACE-ASD — Python Inference API

Provides PACEASDPredictor: a high-level interface for running PACE-ASD
inference from Python code.

Example usage:
    import sys
    sys.path.insert(0, 'src')
    from inference_api import PACEASDPredictor

    predictor = PACEASDPredictor(
        checkpoint='models/A1/fold1_seed42.pt',
        config='configs/inference.yaml',
    )

    # From pre-extracted .npy
    result = predictor.predict_npy('processed/features/asd_1.npy')

    # From raw video
    result = predictor.predict_video('path/to/video.mp4')

    print(result['calibrated_probability'])
    print(result['body_region_attribution'])
"""

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model import ASDMotionModel
from calibration import PlattScaler
from asymmetry import compute_bilateral_asymmetry


N_LANDMARKS    = 33
T_MAX          = 300
LEFT_HIP       = 23
RIGHT_HIP      = 24
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


class PACEASDPredictor:
    """
    High-level inference interface for PACE-ASD.

    Args:
        checkpoint: path to trained checkpoint (.pt file)
        config:     path to config file (inference.yaml or config.yaml)
        device:     'auto' | 'cpu' | 'cuda' (default: 'auto')
    """

    def __init__(self, checkpoint: str, config: str, device: str = "auto"):
        with open(config) as f:
            self.cfg = yaml.safe_load(f)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._load_model(checkpoint)

    def _load_model(self, checkpoint_path: str) -> None:
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model_cfg = ckpt.get("config", self.cfg)

        inf_cfg = self.cfg.get("inference", {})
        use_gate        = inf_cfg.get("use_gate", True)
        use_transformer = inf_cfg.get("use_transformer", True)

        self.model = ASDMotionModel(model_cfg, use_gate=use_gate,
                                    use_transformer=use_transformer)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()
        self.ckpt_path = checkpoint_path
        self.threshold = self.cfg.get("inference", {}).get("threshold", 0.5)

        self.scaler = None
        if "scaler" in ckpt and ckpt["scaler"] is not None:
            try:
                self.scaler = pickle.loads(ckpt["scaler"])
            except Exception:
                pass

    def predict_npy(self, npy_path: str) -> dict:
        """
        Run inference on a pre-extracted .npy feature file.

        Args:
            npy_path: path to (300, 33, 2) float32 numpy file

        Returns:
            dict with prediction results
        """
        positions = np.load(npy_path).astype(np.float32)
        if positions.shape != (T_MAX, N_LANDMARKS, 2):
            raise ValueError(f"Expected ({T_MAX}, {N_LANDMARKS}, 2), got {positions.shape}")
        clip_id = Path(npy_path).stem
        return self._run(positions, clip_id=clip_id, source=npy_path)

    def predict_video(self, video_path: str) -> dict:
        """
        Run inference on a raw video file (requires MediaPipe + OpenCV).

        Args:
            video_path: path to video file (.mp4, .avi, etc.)

        Returns:
            dict with prediction results
        """
        import warnings
        import os as _os
        _os.environ["GLOG_minloglevel"] = "2"
        warnings.filterwarnings("ignore")
        import mediapipe as mp
        import cv2

        prep = self.cfg.get("preprocessing", {})
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
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if res.pose_landmarks:
                kp = np.array([[lm.x, lm.y] for lm in res.pose_landmarks.landmark],
                               dtype=np.float32)
            else:
                kp = np.zeros((N_LANDMARKS, 2), dtype=np.float32)
            frames.append(kp)
        cap.release()
        pose.close()

        if not frames:
            raise RuntimeError("No frames extracted from video")

        raw = np.stack(frames, axis=0)  # (T, 33, 2)
        # Normalize
        kp = raw.copy()
        for t in range(kp.shape[0]):
            if not np.any(kp[t] != 0):
                continue
            mid_hip = (kp[t, LEFT_HIP] + kp[t, RIGHT_HIP]) / 2.0
            kp[t] -= mid_hip
            sd = max(float(np.linalg.norm(kp[t, LEFT_SHOULDER] - kp[t, RIGHT_SHOULDER])), 1e-5)
            kp[t] /= sd
        # Pad/truncate
        T = kp.shape[0]
        if T >= T_MAX:
            positions = kp[:T_MAX]
        else:
            pad = np.zeros((T_MAX - T, N_LANDMARKS, 2), dtype=np.float32)
            positions = np.concatenate([kp, pad], axis=0)

        clip_id = Path(video_path).stem
        return self._run(positions, clip_id=clip_id, source=video_path)

    # backward-compatible alias
    def predict(self, source: str) -> dict:
        """
        Automatically detect source type and run inference.
        Accepts .npy files or video files (.mp4, .avi, etc.).
        """
        if source.endswith(".npy"):
            return self.predict_npy(source)
        else:
            return self.predict_video(source)

    def _run(self, positions: np.ndarray, clip_id: str, source: str) -> dict:
        """Core inference logic shared by predict_npy and predict_video."""
        x = torch.from_numpy(positions).unsqueeze(0).to(self.device)  # (1, T, 33, 2)

        # Forward pass
        with torch.no_grad():
            probs, logits = self.model(x)
            raw_logit = float(logits[0].item())
            raw_prob  = float(probs[0].item())

        # Calibration
        if self.scaler is not None:
            cal_prob = float(self.scaler.calibrate(np.array([raw_logit]))[0])
        else:
            cal_prob = raw_prob

        prediction = "ASD" if cal_prob >= self.threshold else "TD"

        # Block-ESG events
        selected_events = []
        block_saliency  = None
        if self.model.saliency_gate is not None:
            with torch.no_grad():
                _, raw_ind, raw_bs = self.model.get_attention_maps(x)
            if raw_bs is not None:
                block_saliency = raw_bs[0].cpu().numpy().tolist()
            if raw_ind is not None:
                ind_np = raw_ind[0].cpu().numpy()
                L = self.model.saliency_gate.block_size
                block_starts = sorted(set(int(i // L) * L for i in ind_np))
                selected_events = [{"start_frame": s, "end_frame": s + L - 1}
                                    for s in block_starts]

        # Attribution (gradient × input)
        body_region_attr    = {}
        kinematic_stream_attr = {}
        joint_attribution   = []
        try:
            x_attr = torch.from_numpy(positions).unsqueeze(0).to(self.device)
            x_attr.requires_grad_(True)
            p2, l2 = self.model(x_attr)
            l2.sum().backward()
            if x_attr.grad is not None:
                gi = (x_attr.grad * x_attr).squeeze(0).detach().cpu().numpy()  # (T, 33, 2)
                importance = np.abs(gi).sum(axis=(0, 2))  # (33,)
                total = importance.sum() + 1e-10
                body_region_attr = {r: round(float(importance[j_list].sum() / total), 4)
                                    for r, j_list in BODY_REGIONS.items()}
                joint_attribution = [round(float(importance[j] / total), 6)
                                     for j in range(N_LANDMARKS)]
        except Exception:
            pass

        # Bilateral asymmetry
        asymmetry = compute_bilateral_asymmetry(positions)

        return {
            "clip_id":                      clip_id,
            "source":                       source,
            "prediction":                   prediction,
            "raw_probability":              round(raw_prob, 6),
            "calibrated_probability":       round(cal_prob, 6),
            "threshold":                    self.threshold,
            "has_platt_scaler":             self.scaler is not None,
            "selected_events":              selected_events,
            "event_saliency":               block_saliency,
            "body_region_attribution":      body_region_attr,
            "kinematic_stream_attribution": kinematic_stream_attr,
            "joint_attribution":            joint_attribution,
            "bilateral_asymmetry":          asymmetry,
        }
