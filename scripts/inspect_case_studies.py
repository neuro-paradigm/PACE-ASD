import sys, yaml
import numpy as np
sys.path.insert(0, 'src')
from interpretability import load_and_aggregate_runs, compute_population_profiles

with open('configs/config.yaml') as f:
    config = yaml.safe_load(f)

runs = load_and_aggregate_runs('A1', config)
print(f"Loaded runs: {len(runs)}")

pop_test, cross_chk = compute_population_profiles(runs, 'test', n_frames=300)

for gname in ['TP', 'TN', 'FP', 'FN']:
    rep = pop_test[gname]['representative']
    if rep:
        prof = rep['frame_profile']
        p_max = float(np.max(prof))
        p_mean = float(np.mean(prof))
        # Find peak frame regions
        peaks = np.where(prof > 0.005)[0]
        print(f"=== {gname} ===")
        print(f"Subject: {rep['subject_id']}, Probability: {rep['prob']:.3f}")
        print(f"Profile max: {p_max:.4f}, mean: {p_mean:.4f}")
        print(f"Frames with attention > 0.005: {len(peaks)} frames")
        if len(peaks) > 0:
            print(f"Frame ranges: min={peaks[0]}, max={peaks[-1]}")
            # print top 5 peaks
            top_idx = np.argsort(prof)[-5:][::-1]
            print(f"Top 5 frame indices: {top_idx}, values: {[round(float(prof[i]), 4) for i in top_idx]}")
