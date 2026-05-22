import json
import numpy as np

with open('D:\\A_filtered\\A\\A001_GMS_1_1_1.json', 'r') as f:
    data = json.load(f)['data']

print(f'Number of frames: {len(data)}')
for i, frame in enumerate(data[:5]):
    skels = frame.get('skeleton', [])
    print(f'Frame {i} has {len(skels)} skeletons')
    for j, sk in enumerate(skels):
        pose = sk.get('pose', [])
        score = sk.get('score', [])
        print(f'  Skel {j}: pose len {len(pose)}, score len {len(score)}, mean score {np.mean(score)}')
