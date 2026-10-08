"""
PACE-ASD — Dataset (Dryad-Only, Protocol Section 1.2)

Two dataset classes:
  ASDMotionDataset       — loads all clips; used for val / test
  SubjectSampledDataset  — samples N clips per subject per epoch; used for training

No domain labels, no domain samplers — Move4AS is completely dropped.
"""

import os
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from sequence import DEFAULT_MAX_ABS, prepare_sequence


# ── Subject ID extraction ─────────────────────────────────────────────────────

def extract_subject_id(clip_id: str) -> str:
    """
    Return subject_id from clip_id.
    Convention:
        asd_{N}        → asd_{N}
        td_{N}         → td_{N}
        severe_{case}_v{i} → severe_{case}
    """
    if clip_id.startswith("severe_"):
        # severe_case2_v1 → severe_case2
        parts = clip_id.rsplit("_v", 1)
        return parts[0]
    return clip_id


# ── Augmentation helpers ──────────────────────────────────────────────────────

_SWAP_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
    (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]


def _flip_horizontal(seq: np.ndarray) -> np.ndarray:
    """Mirror X and swap left/right joint pairs."""
    seq = seq.copy()
    seq[:, :, 0] = -seq[:, :, 0]
    for left, right in _SWAP_PAIRS:
        seq[:, left, :], seq[:, right, :] = (
            seq[:, right, :].copy(),
            seq[:, left, :].copy(),
        )
    return seq


def _time_warp(seq: np.ndarray, vm: np.ndarray, speed: float) -> np.ndarray:
    """Resample the span from the first to the last valid frame by `speed`,
    keeping its start frame. An output frame is valid only when both source
    frames it interpolates between are valid, so detection gaps are carried
    over instead of being blended into neighbouring poses."""
    idx = np.flatnonzero(vm)
    first, last = int(idx[0]), int(idx[-1])
    span = last - first + 1
    new_span = int(round(span * speed))
    new_span = max(10, min(new_span, len(seq) - first))
    src = np.linspace(first, last, new_span)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, last)
    w = (src - lo).astype(np.float32)[:, None, None]
    ok = vm[lo] & (vm[hi] | (w[:, 0, 0] == 0))
    warped = (1.0 - w) * seq[lo] + w * seq[hi]
    warped[~ok] = 0.0
    out = np.zeros_like(seq)
    out[first: first + new_span] = warped
    return out


def augment_sequence(seq: np.ndarray) -> np.ndarray:
    """
    Light augmentation applied during training only.
    seq: (T, 33, 2) float32

    Every operation acts on frames with a detected pose only, wherever they
    lie in the array; undetected frames (leading, internal or trailing) stay
    zero and the first valid frame keeps its position.
      1. Horizontal reflection with left/right landmark exchange (p=0.5)
      2. Global scaling by U(0.95, 1.05) (p=0.5)
      3. Gaussian coordinate noise, sigma=0.002 (p=0.5)
      4. Time warping of the valid span by U(0.92, 1.08) (p=0.3)
      5. Zeroing of 1-3 random landmarks in every valid frame (p=0.3)
    """
    vm = np.abs(seq).sum(axis=(1, 2)) > 1e-4
    n_valid = int(vm.sum())
    if n_valid < 5:
        return seq
    seq = seq.copy()

    # 1. Horizontal reflection (zero frames are unchanged by it)
    if np.random.rand() < 0.5:
        seq = _flip_horizontal(seq)

    # 2. Global scale
    if np.random.rand() < 0.5:
        seq[vm] *= np.random.uniform(0.95, 1.05)

    # 3. Coordinate noise
    if np.random.rand() < 0.5:
        seq[vm] += np.random.normal(0.0, 0.002, size=(n_valid, 33, 2)).astype(np.float32)

    # 4. Time warp of the valid span
    if np.random.rand() < 0.3 and n_valid > 15:
        warped = _time_warp(seq, vm, np.random.uniform(0.92, 1.08))
        vm_w = np.abs(warped).sum(axis=(1, 2)) > 1e-4
        if vm_w.sum() >= 5:
            seq, vm = warped, vm_w

    # 5. Landmark dropout: 1-3 landmarks zeroed in every valid frame
    if np.random.rand() < 0.3:
        n_drop = np.random.randint(1, 4)
        drop_joints = np.random.choice(33, size=n_drop, replace=False)
        rows = np.flatnonzero(vm)
        seq[np.ix_(rows, drop_joints)] = 0.0

    return seq


# ── Dataset classes ───────────────────────────────────────────────────────────

class ASDMotionDataset(Dataset):
    """
    Standard dataset — returns every clip once.
    Used for validation, test, and supplementary evaluation.
    """

    def __init__(self, clip_ids: list, labels: list,
                 features_dir: str, augment: bool = False,
                 max_abs: float | None = DEFAULT_MAX_ABS, align: bool = True):
        self.clip_ids     = clip_ids
        self.labels       = labels
        self.features_dir = features_dir
        self.augment      = augment
        self.max_abs      = max_abs
        self.align        = align

    def __len__(self) -> int:
        return len(self.clip_ids)

    def __getitem__(self, idx: int):
        clip_id = self.clip_ids[idx]
        label   = self.labels[idx]
        path    = os.path.join(self.features_dir, f"{clip_id}.npy")
        seq     = np.load(path).astype(np.float32)   # (300, 33, 2)
        seq     = prepare_sequence(seq, self.max_abs, self.align)

        if self.augment:
            seq = augment_sequence(seq)

        return (
            torch.from_numpy(seq),
            torch.tensor(label, dtype=torch.float32),
        )


class SubjectSampledDataset(Dataset):
    """
    Training dataset with subject-level clip sampling.

    Each call to _resample() picks N clips per subject at random.
    With the Dryad-only dataset (1 raw clip per subject), this class
    acts identically to ASDMotionDataset when clips_per_subject=1,
    but remains useful if augmented clip variants are added later.

    Resampled at the start of every epoch via train_loader.dataset._resample().
    """

    def __init__(self, clip_ids: list, labels: list, subject_ids: list,
                 features_dir: str, augment: bool = True,
                 clips_per_subject: int = 1,
                 max_abs: float | None = DEFAULT_MAX_ABS, align: bool = True):
        self.features_dir      = features_dir
        self.augment           = augment
        self.clips_per_subject = clips_per_subject
        self.max_abs           = max_abs
        self.align             = align

        # Group clips by subject
        self.subject_clips: dict = defaultdict(list)
        self.subject_label: dict = {}
        for cid, lbl, sid in zip(clip_ids, labels, subject_ids):
            self.subject_clips[sid].append((cid, int(lbl)))
            self.subject_label[sid] = int(lbl)

        self.subjects = sorted(self.subject_clips.keys())
        self._resample()

    def _resample(self) -> None:
        """Pick clips_per_subject clips per subject. Called each epoch."""
        samples = []
        for sid in self.subjects:
            clips = self.subject_clips[sid]
            n     = self.clips_per_subject
            if len(clips) >= n:
                chosen = random.sample(clips, n)
            else:
                # Repeat if fewer clips than requested (rare with current data)
                chosen = clips * (n // len(clips) + 1)
                chosen = chosen[:n]
            samples.extend(chosen)
        random.shuffle(samples)
        self.epoch_samples = samples

    def __len__(self) -> int:
        return len(self.epoch_samples)

    def __getitem__(self, idx: int):
        clip_id, label = self.epoch_samples[idx]
        path = os.path.join(self.features_dir, f"{clip_id}.npy")
        seq  = np.load(path).astype(np.float32)
        seq  = prepare_sequence(seq, self.max_abs, self.align)

        if self.augment:
            seq = augment_sequence(seq)

        return (
            torch.from_numpy(seq),
            torch.tensor(label, dtype=torch.float32),
        )


# ── Labels loader ─────────────────────────────────────────────────────────────

def load_labels(processed_dir: str) -> pd.DataFrame:
    """Load processed/labels.csv. Returns DataFrame."""
    path = os.path.join(processed_dir, "labels.csv")
    return pd.read_csv(path)


# ── DataLoader factory ────────────────────────────────────────────────────────

def create_dataloaders(
    train_ids, train_labels, train_subjects,
    val_ids, val_labels,
    features_dir: str,
    batch_size: int = 16,
    num_workers: int = 0,
    clips_per_subject: int = 1,
):
    """
    Build train and validation DataLoaders.

    Training uses SubjectSampledDataset (augment=True).
    Validation uses ASDMotionDataset (augment=False, all clips evaluated).
    """
    train_ds = SubjectSampledDataset(
        train_ids, train_labels, train_subjects,
        features_dir, augment=True,
        clips_per_subject=clips_per_subject,
    )
    val_ds = ASDMotionDataset(
        val_ids, val_labels, features_dir, augment=False
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )
    return train_loader, val_loader
