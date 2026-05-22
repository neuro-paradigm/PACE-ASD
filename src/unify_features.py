"""
Unify Training Features — Eliminates format-dependent skeleton differences.

This script:
1. Loads all .npy features from processed/features/
2. Zeros out joints that were approximated differently across data sources:
   - Landmarks 17-22 (pinky, index, thumb) — different hand extension formulas
   - Landmarks 29-32 (heel, foot_index) — real detection in td_data vs approximated in asdpose
3. Downsamples the majority class (asdpose ASD) to balance with TD count
4. Saves unified features to processed_unified/features/ with updated labels.csv
"""

import os
import sys
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from train import extract_subject_id


# Joints to zero out (format-dependent approximations that differ between converters)
# Added 25, 26, 27, 28 (knees and ankles) because 40% of asdpose has static/missing legs
ZERO_JOINTS = [17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 31, 32]


def get_source(video_id):
    if video_id.startswith('asdpose_'):
        return 'asdpose'
    elif video_id.startswith('td_'):
        return 'td_data'
    elif video_id.startswith('m4a_'):
        return 'move4as'
    else:
        return 'other'


def main():
    src_processed = 'processed'
    dst_processed = 'processed_unified'
    src_features = os.path.join(src_processed, 'features')
    dst_features = os.path.join(dst_processed, 'features')
    src_labels = os.path.join(src_processed, 'labels.csv')
    dst_labels = os.path.join(dst_processed, 'labels.csv')

    # Load labels
    df = pd.read_csv(src_labels)
    df['source'] = df['video_id'].apply(get_source)
    df['subject_id'] = df['video_id'].apply(extract_subject_id)

    print("=" * 70)
    print("  TRAINING DATA UNIFICATION")
    print("=" * 70)

    # --- Step 1: Class balance via subject-level downsampling ---
    print("\n[Step 1] Balancing classes via subject-level downsampling...")

    # Current counts
    asd_clips = df[df['label'] == 1]
    td_clips = df[df['label'] == 0]
    print(f"  Before: ASD={len(asd_clips)} clips, TD={len(td_clips)} clips")

    # Target: roughly match TD clip count
    # Strategy: downsample asdpose ASD subjects, keep all move4as and other
    target_asd_total = len(td_clips)  # ~22,608

    # Keep all non-asdpose ASD clips
    asd_non_asdpose = asd_clips[asd_clips['source'] != 'asdpose']
    asd_asdpose = asd_clips[asd_clips['source'] == 'asdpose']

    # How many asdpose ASD clips do we need?
    asd_needed_from_asdpose = target_asd_total - len(asd_non_asdpose)

    if asd_needed_from_asdpose < len(asd_asdpose):
        # Downsample at subject level to preserve diversity
        asd_subjects = asd_asdpose['subject_id'].unique()
        np.random.seed(42)

        # Calculate how many clips per subject to keep
        clips_per_subj = asd_asdpose.groupby('subject_id').size()
        total_asdpose_clips = clips_per_subj.sum()
        keep_ratio = asd_needed_from_asdpose / total_asdpose_clips

        # Sample proportionally from each subject
        sampled_indices = []
        for subj in asd_subjects:
            subj_df = asd_asdpose[asd_asdpose['subject_id'] == subj]
            n_keep = max(1, int(len(subj_df) * keep_ratio))
            sampled = subj_df.sample(n=min(n_keep, len(subj_df)), random_state=42)
            sampled_indices.extend(sampled.index.tolist())

        asd_asdpose_sampled = asd_asdpose.loc[sampled_indices]
        print(f"  Downsampled asdpose ASD: {len(asd_asdpose)} -> {len(asd_asdpose_sampled)} clips")
    else:
        asd_asdpose_sampled = asd_asdpose
        print(f"  No downsampling needed for asdpose ASD")

    # Combine: downsampled asdpose ASD + all other ASD + all TD
    df_balanced = pd.concat([
        asd_asdpose_sampled,
        asd_non_asdpose,
        td_clips
    ]).sort_index()

    asd_final = len(df_balanced[df_balanced['label'] == 1])
    td_final = len(df_balanced[df_balanced['label'] == 0])
    print(f"  After:  ASD={asd_final} clips, TD={td_final} clips (ratio {asd_final/max(td_final,1):.2f}:1)")

    # --- Step 2: Create unified features ---
    print(f"\n[Step 2] Creating unified features (zeroing joints {ZERO_JOINTS})...")
    os.makedirs(dst_features, exist_ok=True)

    success = 0
    missing = 0
    for _, row in tqdm(df_balanced.iterrows(), total=len(df_balanced), desc="  Processing"):
        vid = row['video_id']
        src_path = os.path.join(src_features, f'{vid}.npy')

        if not os.path.exists(src_path):
            missing += 1
            continue

        seq = np.load(src_path)  # (300, 33, 3)

        # Zero out format-dependent joints
        seq[:, ZERO_JOINTS, :] = 0.0

        # Save
        dst_path = os.path.join(dst_features, f'{vid}.npy')
        np.save(dst_path, seq)
        success += 1

    print(f"  Processed: {success} clips, Missing: {missing} clips")

    # --- Step 3: Save labels ---
    print(f"\n[Step 3] Saving unified labels...")
    # Only keep clips that exist
    existing_vids = set()
    for f in os.listdir(dst_features):
        if f.endswith('.npy'):
            existing_vids.add(f[:-4])

    df_final = df_balanced[df_balanced['video_id'].isin(existing_vids)]
    # Save only the original columns (drop source and subject_id)
    df_final[['video_id', 'label']].to_csv(dst_labels, index=False)

    print(f"  Saved {len(df_final)} labels to {dst_labels}")

    # --- Summary ---
    df_final_src = df_final.copy()
    df_final_src['source'] = df_final_src['video_id'].apply(get_source)
    print("\n" + "=" * 70)
    print("  UNIFICATION COMPLETE")
    print("=" * 70)
    print(f"  Output directory: {dst_processed}/")
    print(f"  Total clips: {len(df_final)}")
    print(f"  ASD: {len(df_final[df_final['label']==1])}")
    print(f"  TD:  {len(df_final[df_final['label']==0])}")
    print(f"\n  Per-source breakdown:")
    for src in ['asdpose', 'td_data', 'move4as', 'other']:
        sub = df_final_src[df_final_src['source'] == src]
        asd = len(sub[sub['label'] == 1])
        td = len(sub[sub['label'] == 0])
        print(f"    {src:<10}: ASD={asd:<6} TD={td:<6}")
    print(f"\n  Zeroed joints: {ZERO_JOINTS}")
    print(f"  Active joints: {[j for j in range(33) if j not in ZERO_JOINTS]}")
    print("=" * 70)


if __name__ == '__main__':
    main()
