import os
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm

def pad_or_truncate(seq, max_frames=300):
    length = seq.shape[0]
    if length > max_frames:
        return seq[:max_frames]
    elif length < max_frames:
        # Zero padding
        pad_size = max_frames - length
        pad = np.zeros((pad_size, seq.shape[1], seq.shape[2]), dtype=seq.dtype)
        return np.concatenate((seq, pad), axis=0)
    return seq

def main():
    src_processed = 'processed'
    p2_dir = 'processed_2'
    dst_processed = 'processed_fullbody'
    
    dst_features = os.path.join(dst_processed, 'features')
    dst_labels = os.path.join(dst_processed, 'labels.csv')
    
    os.makedirs(dst_features, exist_ok=True)
    
    print("=" * 70)
    print("  MERGING FULL-BODY CLINICAL DATA (MOVE4AS + PROCESSED_2)")
    print("=" * 70)

    # 1. Handle move4as (from original processed folder)
    print("[1/2] Processing move4as clips...")
    df_orig = pd.read_csv(os.path.join(src_processed, 'labels.csv'))
    df_m4a = df_orig[df_orig['video_id'].str.startswith('m4a_')].copy()
    
    m4a_records = []
    success_m4a = 0
    for _, row in tqdm(df_m4a.iterrows(), total=len(df_m4a), desc="  Copying move4as"):
        vid = row['video_id']
        src_path = os.path.join(src_processed, 'features', f'{vid}.npy')
        dst_path = os.path.join(dst_features, f'{vid}.npy')
        if os.path.exists(src_path):
            # Just copy, they are already 300 frames
            shutil.copy2(src_path, dst_path)
            m4a_records.append({'video_id': vid, 'label': row['label']})
            success_m4a += 1
            
    print(f"  -> Merged {success_m4a} move4as clips.")

    # 2. Handle processed_2
    print("\n[2/2] Processing processed_2 clips...")
    p2_records = []
    success_p2 = 0
    
    for cls_name, label in [('asd', 1), ('td', 0)]:
        cls_dir = os.path.join(p2_dir, cls_name)
        if not os.path.exists(cls_dir):
            continue
            
        files = [f for f in os.listdir(cls_dir) if f.endswith('.npy')]
        for f in tqdm(files, desc=f"  Padding processed_2/{cls_name}"):
            vid = f"p2_{cls_name}_{f[:-4]}"
            src_path = os.path.join(cls_dir, f)
            dst_path = os.path.join(dst_features, f'{vid}.npy')
            
            seq = np.load(src_path)
            seq_padded = pad_or_truncate(seq, max_frames=300)
            np.save(dst_path, seq_padded)
            
            p2_records.append({'video_id': vid, 'label': label})
            success_p2 += 1
            
    print(f"  -> Merged {success_p2} processed_2 clips.")

    # 3. Save combined labels
    df_combined = pd.DataFrame(m4a_records + p2_records)
    df_combined.to_csv(dst_labels, index=False)
    
    total = len(df_combined)
    asd_count = len(df_combined[df_combined['label'] == 1])
    td_count = len(df_combined[df_combined['label'] == 0])

    print("\n" + "=" * 70)
    print("  MERGE COMPLETE!")
    print("=" * 70)
    print(f"  Output directory: {dst_processed}/")
    print(f"  Total clips: {total} (ASD: {asd_count}, TD: {td_count})")
    print("=" * 70)

if __name__ == '__main__':
    main()
