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


def augment_sequence(seq: np.ndarray) -> np.ndarray:
    """
    Light augmentation applied during training only.
    seq: (T, 33, 2) float32

    Augmentations:
      1. Horizontal flip (50%)
      2. Gentle scale variation x U(0.95, 1.05) (50%)
      3. Subtle Gaussian coordinate noise sigma=0.002 on valid frames (50%)
      4. Smooth temporal speed jitter via linear interpolation (30%)
    """
    non_zero_mask = np.any(seq != 0, axis=(1, 2))
    actual_len = int(non_zero_mask.sum())
    if actual_len < 5:
        return seq

    # 1. Horizontal flip
    if np.random.rand() < 0.5:
        seq = _flip_horizontal(seq)

    # 2. Gentle scale
    if np.random.rand() < 0.5:
        scale = np.random.uniform(0.95, 1.05)
        seq[:actual_len] = seq[:actual_len] * scale

    # 3. Subtle coordinate noise
    if np.random.rand() < 0.5:
        noise = np.random.normal(0.0, 0.002, size=(actual_len, 33, 2)).astype(np.float32)
        seq[:actual_len] = seq[:actual_len] + noise

    # 4. Smooth temporal speed variation (linear interpolation)
    if np.random.rand() < 0.3 and actual_len > 15:
        speed = np.random.uniform(0.92, 1.08)
        new_len = int(round(actual_len * speed))
        new_len = max(10, min(new_len, len(seq)))
        orig_indices = np.linspace(0, actual_len - 1, actual_len)
        warp_indices = np.linspace(0, actual_len - 1, new_len)
        warped = np.zeros((new_len, 33, 2), dtype=np.float32)
        for j in range(33):
            for c in range(2):
                warped[:, j, c] = np.interp(warp_indices, orig_indices, seq[:actual_len, j, c])
        # Place warped back into seq
        seq_new = np.zeros_like(seq)
        put_len = min(new_len, len(seq))
        seq_new[:put_len] = warped[:put_len]
        seq = seq_new

    # 5. Joint Dropout (30%): randomly drop 1-3 joints to prevent reliance on single landmark tracking noise
    if np.random.rand() < 0.3 and actual_len > 5:
        n_drop = np.random.randint(1, 4)
        drop_joints = np.random.choice(33, size=n_drop, replace=False)
        seq[:actual_len, drop_joints] = 0.0

    return seq


# ── Dataset classes ───────────────────────────────────────────────────────────

class ASDMotionDataset(Dataset):
    """
    Standard dataset — returns every clip once.
    Used for validation, test, and supplementary evaluation.
    """

    def __init__(self, clip_ids: list, labels: list,
                 features_dir: str, augment: bool = False):
        self.clip_ids     = clip_ids
        self.labels       = labels
        self.features_dir = features_dir
        self.augment      = augment

    def __len__(self) -> int:
        return len(self.clip_ids)

    def __getitem__(self, idx: int):
        clip_id = self.clip_ids[idx]
        label   = self.labels[idx]
        path    = os.path.join(self.features_dir, f"{clip_id}.npy")
        seq     = np.load(path).astype(np.float32)   # (300, 33, 2)

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
                 clips_per_subject: int = 1):
        self.features_dir      = features_dir
        self.augment           = augment
        self.clips_per_subject = clips_per_subject

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
