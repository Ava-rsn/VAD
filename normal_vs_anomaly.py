import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

CSV_PATH  = "ucf_avenue_files/ucf_crime_debug_scores.csv"
SCORE_COL = "score_st_raw"
OUT_PNG   = "score_distribution.png"

df = pd.read_csv(CSV_PATH)

TEST_LIST      = ""
# filter to test set only if TEST_LIST is provided
if TEST_LIST.strip():
    with open(TEST_LIST) as f:
        test_set = set(line.strip() for line in f if line.strip())
    df = df[df["video"].isin(test_set)].reset_index(drop=True)
    print(f"Filtered to {df['video'].nunique()} test videos")

# compute per-video mean score
video_stats = []
for vid, g in df.groupby("video"):
    is_normal  = vid.startswith("Normal")
    mean_score = g[SCORE_COL].mean()
    max_score  = g[SCORE_COL].max()
    frac_anom  = g["label"].mean()
    video_stats.append({
        "video"      : vid,
        "is_normal"  : is_normal,
        "mean_score" : mean_score,
        "max_score"  : max_score,
        "frac_anom"  : frac_anom,
    })

stats_df = pd.DataFrame(video_stats)

normal_means = stats_df[stats_df["is_normal"]]["mean_score"]
crime_means  = stats_df[~stats_df["is_normal"]]["mean_score"]

print(f"Normal videos  - mean: {normal_means.mean():.4f}  median: {normal_means.median():.4f}  std: {normal_means.std():.4f}")
print(f"Crime videos   - mean: {crime_means.mean():.4f}  median: {crime_means.median():.4f}  std: {crime_means.std():.4f}")

# -----------------------------
# Plot
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# left — histogram of per-video mean scores
axes[0].hist(normal_means, bins=30, alpha=0.6, label="Normal", color="#2196F3")
axes[0].hist(crime_means,  bins=30, alpha=0.6, label="Crime",  color="#F44336")
axes[0].set_xlabel("Per-video mean anomaly score")
axes[0].set_ylabel("Number of videos")
axes[0].set_title("Distribution of mean anomaly scores\n(Avenue --> UCF-Crime, st raw)")
axes[0].legend()
axes[0].axvline(normal_means.mean(), color="#2196F3", linestyle="--", linewidth=1.5, label="Normal mean")
axes[0].axvline(crime_means.mean(),  color="#F44336", linestyle="--", linewidth=1.5, label="Crime mean")

# right — histogram of per-video max scores
normal_max = stats_df[stats_df["is_normal"]]["max_score"]
crime_max  = stats_df[~stats_df["is_normal"]]["max_score"]

axes[1].hist(normal_max, bins=30, alpha=0.6, label="Normal", color="#2196F3")
axes[1].hist(crime_max,  bins=30, alpha=0.6, label="Crime",  color="#F44336")
axes[1].set_xlabel("Per-video max anomaly score")
axes[1].set_ylabel("Number of videos")
axes[1].set_title("Distribution of max anomaly scores\n(Avenue --> UCF-Crime, st raw)")
axes[1].legend()

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200)
plt.close()
print(f"Saved to: {OUT_PNG}")