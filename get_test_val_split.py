import numpy as np
import pandas as pd
from collections import defaultdict
import random

# ── config ────────────────────────────────────────────────────────────────
VIDEOS_NPY  = "ucf_shanghai_files/ucf_crime_videos.npy"
LABELS_NPY  = "ucf_shanghai_files/ucf_crime_labels.npy"
VAL_SIZE    = 50      # total validation videos
RANDOM_SEED = 42
VAL_OUT     = "ucf_val_videos.txt"
TEST_OUT    = "ucf_test_videos.txt"
# ──────────────────────────────────────────────────────────────────────────

videos = np.load(VIDEOS_NPY, allow_pickle=True)
labels = np.load(LABELS_NPY, allow_pickle=True)

# get unique videos and their category
unique_videos = np.unique(videos)
cat_to_vids = defaultdict(list)
for vid in unique_videos:
    cat = ''.join([c for c in vid.split("_x264")[0] if not c.isdigit()])
    cat_to_vids[cat].append(vid)

print("Videos per category:")
for cat, vids in sorted(cat_to_vids.items()):
    print(f"  {cat:<20} {len(vids)}")

# stratified sample — proportional to category size
random.seed(RANDOM_SEED)
val_videos  = []
test_videos = []

for cat, vids in sorted(cat_to_vids.items()):
    vids_shuffled = sorted(vids)
    random.shuffle(vids_shuffled)

    if cat == "Normal_Videos_":
        # sample ~15 normal videos for val
        n_val = 15
    else:
        # ~3-4 per crime category to fill 50 total
        n_val = max(1, round(len(vids) * (35 / 481)))  # 35 crime + 15 normal = 50

    val_videos  += vids_shuffled[:n_val]
    test_videos += vids_shuffled[n_val:]

print(f"\nValidation set : {len(val_videos)} videos")
print(f"Test set       : {len(test_videos)} videos")

with open(VAL_OUT, 'w') as f:
    for v in sorted(val_videos):
        f.write(v + "\n")

with open(TEST_OUT, 'w') as f:
    for v in sorted(test_videos):
        f.write(v + "\n")

print(f"Saved val  list to: {VAL_OUT}")
print(f"Saved test list to: {TEST_OUT}")