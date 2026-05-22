import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

class ASDMotionDataset(Dataset):
    """
    Standard dataset that returns individual clips.
    Used for validation and testing to evaluate on ALL samples.
    """
    def __init__(self, video_ids, labels, features_dir, augment=False):
        self.video_ids = video_ids
        self.labels = labels
        self.features_dir = features_dir
        self.augment = augment

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        label = self.labels[idx]
        npy_path = os.path.join(self.features_dir, f"{video_id}.npy")
        sequence = np.load(npy_path).astype(np.float32)

        if self.augment:
            sequence = self._augment(sequence)

        return torch.from_numpy(sequence), torch.tensor(label, dtype=torch.float32)

    def _augment(self, sequence):
        # 1. Horizontal Flip (Mirroring) - 50%
        if np.random.rand() < 0.5:
            sequence = self._flip_horizontal(sequence)

        # 2. Random Scaling (Global size invariance) - 50%
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.9, 1.1)
            sequence = sequence * scale

        # 3. Gaussian noise (per-joint jitter) - 50%
        if np.random.rand() < 0.5:
            noise = np.random.normal(0, 0.005, size=sequence.shape).astype(np.float32)
            mask = np.any(sequence != 0, axis=(1, 2), keepdims=True)
            sequence = sequence + noise * mask
        
        # 4. Temporal Jitter (existing logic)
        if np.random.rand() < 0.3:
            non_zero = np.any(sequence != 0, axis=(1, 2))
            actual_len = int(np.sum(non_zero))
            if actual_len > 10:
                n_drop = max(1, int(actual_len * 0.05))
                drop_idx = np.random.choice(actual_len, size=n_drop, replace=False)
                keep_idx = np.setdiff1d(np.arange(actual_len), drop_idx)
                resample_idx = np.sort(np.concatenate([
                    keep_idx, np.random.choice(keep_idx, size=n_drop, replace=True)
                ]))[:actual_len]
                sequence[:actual_len] = sequence[resample_idx]
        return sequence

    def _flip_horizontal(self, sequence):
        """Flip skeleton across the X-axis and swap left/right joints."""
        # Flip X coordinates
        sequence[:, :, 0] = -sequence[:, :, 0]
        # Swap left/right pairs
        swap_pairs = [
            (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
            (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
            (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)
        ]
        for left, right in swap_pairs:
            temp = sequence[:, left, :].copy()
            sequence[:, left, :] = sequence[:, right, :]
            sequence[:, right, :] = temp
        return sequence


class SubjectSampledDataset(Dataset):
    """
    Subject-level sampling dataset for training.
    
    Each epoch iterates over unique subjects (not individual samples).
    For each subject, N clips are randomly selected per epoch.
    This ensures:
      - Every subject gets equal representation (N clips each)
      - Subjects with many clips don't dominate training
      - Different clips are sampled each epoch (diversity)

    Length = number of unique subjects * clips_per_subject
    """

    def __init__(self, video_ids, labels, subject_ids, features_dir,
                 augment=True, clips_per_subject=5):
        self.features_dir = features_dir
        self.augment = augment
        # clips_per_subject can be an int or a dict {label: count}
        self.clips_per_subject = clips_per_subject

        # Group video_ids and labels by subject
        self.subject_clips = defaultdict(list)
        self.subject_label = {}
        for vid, lbl, subj in zip(video_ids, labels, subject_ids):
            self.subject_clips[subj].append((vid, lbl))
            self.subject_label[subj] = int(lbl)

        self.subjects = sorted(self.subject_clips.keys())
        self._resample()

    def _resample(self):
        """Randomly pick N clips per subject. Called at start of each epoch."""
        self.epoch_samples = []
        for subj in self.subjects:
            clips = self.subject_clips[subj]
            label = self.subject_label[subj]
            
            # Determine count for this subject's class
            if isinstance(self.clips_per_subject, dict):
                n_target = self.clips_per_subject.get(label, 5)
            else:
                n_target = self.clips_per_subject

            n = min(n_target, len(clips))
            if n < n_target:
                # Subject has fewer clips than requested - use all + repeat
                chosen = clips.copy()
                while len(chosen) < n_target:
                    chosen.append(random.choice(clips))
            else:
                chosen = random.sample(clips, n_target)
            self.epoch_samples.extend(chosen)
        random.shuffle(self.epoch_samples)

    def __len__(self):
        return len(self.epoch_samples)

    def __getitem__(self, idx):
        video_id, label = self.epoch_samples[idx]
        npy_path = os.path.join(self.features_dir, f"{video_id}.npy")
        sequence = np.load(npy_path).astype(np.float32)

        if self.augment:
            sequence = self._augment(sequence)

        return torch.from_numpy(sequence), torch.tensor(label, dtype=torch.float32)

    def _augment(self, sequence):
        # Same as ASDMotionDataset but accessed differently if needed
        return ASDMotionDataset._augment(self, sequence)

    def _flip_horizontal(self, sequence):
        return ASDMotionDataset._flip_horizontal(self, sequence)


def load_labels(processed_dir):
    return pd.read_csv(os.path.join(processed_dir, 'labels.csv'))


def create_dataloaders(train_ids, train_labels, val_ids, val_labels,
                       features_dir, batch_size=16, num_workers=0,
                       train_subject_ids=None, val_subject_ids=None,
                       clips_per_subject=5):
    """
    Create DataLoaders.

    If subject_ids are provided, uses SubjectSampledDataset
    (N clips per subject per epoch) for that split.
    """
    if train_subject_ids is not None:
        train_ds = SubjectSampledDataset(
            train_ids, train_labels, train_subject_ids,
            features_dir, augment=True,
            clips_per_subject=clips_per_subject
        )
    else:
        train_ds = ASDMotionDataset(train_ids, train_labels, features_dir, augment=True)

    # Validation and Test always use ALL samples for robust evaluation
    val_ds = ASDMotionDataset(val_ids, val_labels, features_dir, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=False)
    return train_loader, val_loader
