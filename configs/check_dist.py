import pandas as pd
import numpy as np
import re

df = pd.read_csv('d:/ASD/data/processed/labels.csv')
labels = df['label'].values
video_ids = df['video_id'].values

def extract_subj(vid):
    vid = re.sub(r'_c\d+$', '', vid)
    if vid.startswith('m4a_'):
        match = re.search(r'mdata_((?:P|S)\d+)', vid)
        if match: return f'm4a_{match.group(1)}'
    if vid.startswith('asdpose_'):
        return f'asdpose_{vid.split("_")[1]}'
    if vid.startswith('td_'):
        return f'td_{vid.split("_")[1]}'
    return vid.split('_')[0]

subjs = [extract_subj(v) for v in video_ids]
subj_dict = {}
for s, l in zip(subjs, labels):
    subj_dict[s] = l
    
asd_subj = sum(1 for v in subj_dict.values() if v == 1)
td_subj = sum(1 for v in subj_dict.values() if v == 0)

print(f'Total clips: {len(labels)}')
print(f'Clip ASD ratio: {np.mean(labels):.2%}')
print(f'Total subjects: {len(subj_dict)}')
print(f'ASD subjects: {asd_subj}')
print(f'TD subjects: {td_subj}')
print(f'Subject ASD ratio: {asd_subj / len(subj_dict):.2%}')
