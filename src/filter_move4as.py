import os
import shutil
import pandas as pd
from tqdm import tqdm

def main():
    src_processed = 'processed_unified'
    dst_processed = 'processed_move4as'
    src_features = os.path.join(src_processed, 'features')
    dst_features = os.path.join(dst_processed, 'features')
    src_labels = os.path.join(src_processed, 'labels.csv')
    dst_labels = os.path.join(dst_processed, 'labels.csv')

    print("=" * 70)
    print("  FILTERING PURE CLINICAL DATA (MOVE4AS)")
    print("=" * 70)

    os.makedirs(dst_features, exist_ok=True)

    df = pd.read_csv(src_labels)
    
    # Filter only move4as clips
    df_m4a = df[df['video_id'].str.startswith('m4a_')].copy()
    
    asd_clips = df_m4a[df_m4a['label'] == 1]
    td_clips = df_m4a[df_m4a['label'] == 0]
    
    print(f"  Found move4as clips - ASD: {len(asd_clips)}, TD: {len(td_clips)}")
    
    # The dataloader handles balanced sampling via BalancedSubjectSampler, 
    # but we can also optionally downsample TD to make epochs naturally balanced.
    # Let's keep all valid move4as data since we want every drop of signal.
    df_final = df_m4a
    
    success = 0
    missing = 0
    for _, row in tqdm(df_final.iterrows(), total=len(df_final), desc="  Copying features"):
        vid = row['video_id']
        src_path = os.path.join(src_features, f'{vid}.npy')
        dst_path = os.path.join(dst_features, f'{vid}.npy')
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            success += 1
        else:
            missing += 1

    df_final[['video_id', 'label']].to_csv(dst_labels, index=False)

    print("\n" + "=" * 70)
    print("  DONE!")
    print("=" * 70)
    print(f"  Output directory: {dst_processed}/")
    print(f"  Total clips: {success}")
    print(f"  Missing: {missing}")
    print("=" * 70)

if __name__ == '__main__':
    main()
