import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

CSV_PATH  = "ucf_crime_teacher_filt_r300_mu300.csv"
SCORE_COL = "score_teacher_filt"
TEST_LIST      = "/user/home/gk23779/evaluations/ucf_test_videos.txt"

df = pd.read_csv(CSV_PATH)

# filter to test set only if TEST_LIST is provided
if TEST_LIST.strip():
    with open(TEST_LIST) as f:
        test_set = set(line.strip() for line in f if line.strip())
    df = df[df["video"].isin(test_set)].reset_index(drop=True)
    print(f"Filtered to {df['video'].nunique()} test videos")

df["category"] = df["video"].apply(lambda x: ''.join([c for c in x.split("_x264")[0] if not c.isdigit()]))

print(f"{'Video':<35} {'Type':<15} {'Mean score':>10} {'Max score':>10} {'Frames':>7}")
print("-" * 80)

for vid, g in df.groupby("video"):
    if g["label"].nunique() >= 2:
        continue

    all_anom = g["label"].mean() == 1.0
    vid_type = "all-anomalous" if all_anom else "all-normal"
    mean_s   = g[SCORE_COL].mean()
    max_s    = g[SCORE_COL].max()
    n        = len(g)

    print(f"{vid:<35} {vid_type:<15} {mean_s:>10.4f} {max_s:>10.4f} {n:>7}")

# -----------------------------
# AUC for normal videos
# -----------------------------
normal_aucs = []

for vid, g in df.groupby("video"):
    if not vid.startswith("Normal"):
        continue
    if g["label"].nunique() < 2:
        continue
    auc = roc_auc_score(g["label"], g[SCORE_COL])
    normal_aucs.append(auc)

if normal_aucs:
    print(f"\nNormal videos with both classes present: {len(normal_aucs)}")
    print(f"Mean AUC   : {np.mean(normal_aucs):.4f}")
    print(f"Median AUC : {np.median(normal_aucs):.4f}")
    print(f"Std AUC    : {np.std(normal_aucs):.4f}")
else:
    print("\nNo normal videos had both classes present --> all are truly all-normal.")