import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def main():
    target_dir = 'processed_fullbody'
    features_dir = os.path.join(target_dir, 'features')
    labels_file = os.path.join(target_dir, 'labels.csv')
    
    df = pd.read_csv(labels_file)
    initial_count = len(df)
    
    print("=" * 70)
    print("  CLEANING FROZEN CLIPS FROM DATASET")
    print("=" * 70)
    
    frozen_vids = []
    
    files = [f for f in os.listdir(features_dir) if f.endswith('.npy')]
    for f in tqdm(files, desc="  Scanning for frozen clips"):
        path = os.path.join(features_dir, f)
        data = np.load(path)
        
        var = data.var(axis=0) # (33, 3)
        active_var = var.sum(axis=1) # (33,)
        
        # If less than 10 joints are moving, it's considered frozen/static
        moving_joints = np.sum(active_var > 1e-6)
        if moving_joints < 10:
            vid = f.replace('.npy', '')
            frozen_vids.append(vid)
            # Delete the frozen numpy file
            os.remove(path)
            
    # Filter the dataframe
    df_clean = df[~df['video_id'].isin(frozen_vids)]
    
    # Save the cleaned labels
    df_clean.to_csv(labels_file, index=False)
    
    final_count = len(df_clean)
    removed = initial_count - final_count
    
    print("\n" + "=" * 70)
    print("  CLEANING COMPLETE!")
    print("=" * 70)
    print(f"  Frozen clips detected & deleted: {len(frozen_vids)}")
    print(f"  Labels removed from CSV      : {removed}")
    print(f"  Final active clips remaining : {final_count}")
    print("=" * 70)

if __name__ == '__main__':
    main()
