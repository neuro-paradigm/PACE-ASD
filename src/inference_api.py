"""
PACE-ASD — Python inference API.

PACEASDPredictor scores one recording with one checkpoint or an ensemble of
checkpoints and returns, besides the probability, everything needed to check
how it was obtained: the prepared landmark sequence and its kinematics, the
blocks each ensemble member selected (in the frame numbering of the input),
how often members agree on each block, and descriptor-level attribution
shares.

    import sys; sys.path.insert(0, "src")
    from inference_api import PACEASDPredictor

    predictor = PACEASDPredictor("models/release")          # directory, file or list
    result = predictor.predict_npy("processed/features/asd_1.npy")
    result = predictor.predict_video("walk.mp4")             # MediaPipe + OpenCV
    print(result["probability"], result["probability_sd"])

The input goes through exactly the preparation used in training
(src/sequence.py): frames with implausible normalised coordinates are treated
as undetected, and the sequence is shifted so its first detected frame is
frame 0. Outputs report frames in the numbering of the input, with the shift
recorded as `onset_frame`.
"""

import glob
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from calibration import PlattScaler                       # noqa: E402
from model import ASDMotionModel                          # noqa: E402
from sequence import DEFAULT_MAX_ABS, prepare_sequence, valid_mask  # noqa: E402

N_LANDMARKS = 33
T_MAX = 300
PACE_FLAGS = {"PACE": (True, True), "PACE-nogate": (False, True), "PACE-frames": (True, True),
              "PACE-noattn": (True, False), "PACE-M4": (True, True), "PACE-nomask": (True, True),
              "PACE-noalign": (True, True), "PACE-valloss": (True, True), "A1": (True, True),
              "A2": (False, True), "A3": (True, True), "A4": (True, False)}
REGIONS = {"head": list(range(0, 11)), "arms_hands": list(range(11, 23)),
           "hips": [23, 24], "legs_feet": list(range(25, 33))}
STREAMS = {"position": slice(0, 66), "velocity": slice(66, 132), "acceleration": slice(132, 198)}


def _scaler_from(obj):
    if obj is None:
        return None
    if isinstance(obj, (bytes, bytearray)):
        return pickle.loads(obj)
    s = PlattScaler()
    with torch.no_grad():
        s.log_temperature.fill_(float(np.log(obj["temperature"])))
        s.bias.fill_(float(obj["bias"]))
    return s


def checkpoint_paths(spec) -> list:
    if isinstance(spec, (list, tuple)):
        return [str(p) for p in spec]
    if os.path.isdir(spec):
        return sorted(glob.glob(os.path.join(spec, "*.pt")))
    return [str(spec)]


class PACEASDPredictor:
    """
    Args:
        checkpoints: a .pt file, a directory of .pt files (ensemble) or a list.
        config:      optional YAML; only `inference.threshold`, `inference.device`
                     and the MediaPipe settings under `preprocessing` are read.
        device:      'auto' | 'cpu' | 'cuda'
    """

    def __init__(self, checkpoints=None, config: str | None = None, device: str = "auto",
                 checkpoint=None):
        checkpoints = checkpoints if checkpoints is not None else checkpoint
        self.cfg = yaml.safe_load(open(config)) if config else {}
        inf = self.cfg.get("inference", {})
        if device == "auto":
            device = inf.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() else "cpu"
        self.device = torch.device(device)
        self.threshold = float(inf.get("threshold", 0.5))
        self.paths = checkpoint_paths(checkpoints)
        if not self.paths:
            raise FileNotFoundError(f"no checkpoint found at {checkpoints}")
        self.members = [self._load(p) for p in self.paths]
        cfg0 = self.members[0][0].cfg_used
        self.max_abs = cfg0.get("data", {}).get("max_abs_coord", DEFAULT_MAX_ABS)
        self.align = bool(cfg0.get("data", {}).get("align_onset",
                                                   cfg0["model"].get("align_onset", False)))

    def _load(self, path):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        cfg = ck["config"]
        arm = ck.get("arm", self.cfg.get("inference", {}).get("model_variant", "A1"))
        gate, transformer = PACE_FLAGS.get(arm, (True, True))
        model = ASDMotionModel(cfg, use_gate=gate, use_transformer=transformer)
        model.load_state_dict(ck["state_dict"])
        model.to(self.device).eval()
        model.cfg_used = cfg
        return model, _scaler_from(ck.get("scaler"))

    # ── inputs ────────────────────────────────────────────────────────────────

    def predict_npy(self, npy_path: str, attribution: bool = True) -> dict:
        positions = np.load(npy_path).astype(np.float32)
        if positions.shape != (T_MAX, N_LANDMARKS, 2):
            raise ValueError(f"expected ({T_MAX}, {N_LANDMARKS}, 2), got {positions.shape}")
        return self.predict_array(positions, Path(npy_path).stem, npy_path, attribution=attribution)

    def predict_video(self, video_path: str, attribution: bool = True) -> dict:
        from preprocess import extract_keypoints_from_video, normalise, pad_or_truncate
        prep = self.cfg.get("preprocessing", {})
        t0 = time.perf_counter()
        raw, meta = extract_keypoints_from_video(video_path, return_meta=True, **{
            k: prep[k] for k in ("model_complexity", "smooth_landmarks",
                                 "min_detection_confidence", "min_tracking_confidence")
            if k in prep})
        meta["pose_seconds"] = round(time.perf_counter() - t0, 3)
        positions = pad_or_truncate(normalise(raw, meta["width"], meta["height"]))
        meta["truncated_frames"] = max(0, len(raw) - T_MAX)
        return self.predict_array(positions, Path(video_path).stem, video_path,
                                  video=meta, attribution=attribution)

    def predict(self, source: str, **kw) -> dict:
        return self.predict_npy(source, **kw) if source.endswith(".npy") else \
            self.predict_video(source, **kw)

    # ── core ──────────────────────────────────────────────────────────────────

    def prepared(self, positions: np.ndarray):
        """Return (prepared sequence, onset frame in the input numbering)."""
        cleaned = prepare_sequence(positions, self.max_abs, align=False)
        vm = valid_mask(cleaned)
        onset = int(np.argmax(vm)) if (vm.any() and self.align) else 0
        return prepare_sequence(positions, self.max_abs, self.align), onset

    def predict_array(self, positions: np.ndarray, clip_id: str = "clip", source: str = "",
                      video: dict | None = None, attribution: bool = True) -> dict:
        t0 = time.perf_counter()
        seq, onset = self.prepared(positions)
        x = torch.from_numpy(seq)[None].to(self.device)
        vm = valid_mask(seq)
        n_rejected = int(valid_mask(positions).sum() - valid_mask(
            prepare_sequence(positions, self.max_abs, align=False)).sum())

        probs, logits, selections = [], [], []
        for model, scaler in self.members:
            with torch.no_grad():
                _, lg = model(x)
            lg = float(lg[0])
            logits.append(lg)
            probs.append(float(scaler.calibrate(np.array([lg]))[0]) if scaler else
                         float(1 / (1 + np.exp(-lg))))
            if model.saliency_gate is not None:
                with torch.no_grad():
                    _, idx, scores = model.get_attention_maps(x)
                L = model.saliency_gate.block_size
                blocks = sorted({int(i) // L for i in idx[0].cpu().numpy()})
                selections.append({"block_size": L, "blocks": blocks,
                                   "scores": scores[0].cpu().numpy().tolist()})
        model_seconds = time.perf_counter() - t0

        p = float(np.mean(probs))
        out = {
            "clip_id": clip_id, "source": source, "n_models": len(self.members),
            "probability": round(p, 6),
            "probability_sd": round(float(np.std(probs)), 6),
            "member_probabilities": [round(v, 6) for v in probs],
            "member_logits": [round(v, 6) for v in logits],
            "threshold": self.threshold,
            "above_threshold": bool(p >= self.threshold),
            "n_detected_frames": int(valid_mask(positions).sum()),
            "n_frames_rejected_implausible": n_rejected,
            "onset_frame": onset,
            "model_seconds": round(model_seconds, 4),
            "checkpoints": [os.path.relpath(pp) for pp in self.paths],
        }
        if video:
            out["video"] = video
        if selections:
            out["selection"] = self._selection_summary(selections, vm, onset)
        if attribution:
            out["attribution"] = self._attribution(x, vm)
        out["_prepared"] = seq
        return out

    @staticmethod
    def _selection_summary(selections, vm, onset):
        L = selections[0]["block_size"]
        n_blocks = int(np.ceil(len(vm) / L))
        counts = np.zeros(n_blocks)
        for s in selections:
            counts[[b for b in s["blocks"] if b < n_blocks]] += 1
        blocks = []
        for b in range(n_blocks):
            n_valid = int(vm[b * L:(b + 1) * L].sum())
            if n_valid == 0 and counts[b] == 0:
                continue
            blocks.append({"block": b, "input_frames": [onset + b * L, onset + (b + 1) * L - 1],
                           "valid_frames": n_valid,
                           "selected_by_fraction_of_models": round(float(counts[b] / len(selections)), 4),
                           "mean_gate_score": round(float(np.mean([s["scores"][b] for s in selections])), 4)})
        n_valid_blocks = int(sum(1 for blk in blocks if blk["valid_frames"] > 0))
        top_m = len(selections[0]["blocks"])
        return {"block_size": L, "budget_blocks": top_m, "n_valid_blocks": n_valid_blocks,
                "selection_is_trivial": n_valid_blocks <= top_m, "blocks": blocks,
                "per_model": [s["blocks"] for s in selections]}

    def _attribution(self, x, vm):
        from attribution import descriptor_gradxinput
        gx = np.mean([descriptor_gradxinput(m, x)[0] for m, _ in self.members], 0)[vm]   # (n, 198)
        tot = gx.sum() + 1e-12
        per_lm = gx.reshape(-1, 3, 33, 2).sum(axis=(0, 3))                             # (3, 33)
        motion = per_lm[1:].sum(0)
        return {
            "stream_share": {k: round(float(gx[:, s].sum() / tot), 4) for k, s in STREAMS.items()},
            "region_share_velocity_acceleration": {
                k: round(float(motion[j].sum() / (motion.sum() + 1e-12)), 4) for k, j in REGIONS.items()},
            "landmark_share": [round(float(v), 5) for v in per_lm.sum(0) / (per_lm.sum() + 1e-12)],
            "note": ("Descriptor-level |gradient x input| averaged over ensemble members: the "
                     "sensitivity of the trained models for this clip, not a measurement of "
                     "the movement."),
        }
