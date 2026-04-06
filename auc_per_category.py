import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc as sklearn_auc

CSV_PATH  = "~/evaluations/ucf_avenue_files/ucf_crime_debug_scores.csv"
SCORE_COL = "score_st_raw"
OUT_PNG   = "ucf_st_raw_per_category_avenue.png"
TEST_LIST = ""

df = pd.read_csv(CSV_PATH)

if TEST_LIST.strip():
    with open(TEST_LIST) as f:
        test_set = set(line.strip() for line in f if line.strip())
    df = df[df["video"].isin(test_set)].reset_index(drop=True)
    print(f"Filtered to {df['video'].nunique()} test videos")

df["category"] = df["video"].apply(lambda x: ''.join([c for c in x.split("_x264")[0] if not c.isdigit()]))

rows_skip     = []   # skip one-class videos
rows_anchored = []   # anchor all videos

for vid, g in df.groupby("video"):
    lbl  = g["label"].values
    pred = g[SCORE_COL].values
    cat  = g["category"].iloc[0]

    # anchored — always computable
    lbl_a  = np.array([0] + list(lbl)  + [1])
    pred_a = np.array([0] + list(pred) + [1])
    fpr, tpr, _ = roc_curve(lbl_a, pred_a)
    rows_anchored.append({"video": vid, "category": cat, "auc": sklearn_auc(fpr, tpr)})

    # skip — only if both classes present
    if g["label"].nunique() < 2:
        continue
    rows_skip.append({"video": vid, "category": cat, "auc": roc_auc_score(lbl, pred)})

video_df_skip     = pd.DataFrame(rows_skip)
video_df_anchored = pd.DataFrame(rows_anchored)

# per-category stats (skip variant — used for plot)
stats = video_df_skip.groupby("category")["auc"].agg(
    count="count",
    mean="mean",
    median="median",
    std="std",
    min="min",
    max="max"
).round(4).reset_index().sort_values("mean", ascending=True)

print("Per-category AUC (skip variant):")
print(stats.to_string(index=False))
print(f"\nOverall Macro AUC (skip,     n={len(video_df_skip):3d}): {video_df_skip['auc'].mean():.4f}  median: {video_df_skip['auc'].median():.4f}  std: {video_df_skip['auc'].std():.4f}")
print(f"Overall Macro AUC (anchored, n={len(video_df_anchored):3d}): {video_df_anchored['auc'].mean():.4f}  median: {video_df_anchored['auc'].median():.4f}  std: {video_df_anchored['auc'].std():.4f}")

stats.to_csv("ucf_per_category_auc.csv", index=False)

# ── plot ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

bars = ax.bar(stats["category"], stats["mean"], color="#1f77b4")

ax.errorbar(
    stats["category"], stats["mean"],
    yerr=stats["std"],
    fmt="none", color="black", capsize=4, linewidth=1.2
)

for bar, val in zip(bars, stats["mean"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=8)

ax.axhline(y=0.5, color="red", linestyle="--", linewidth=1, label="Random (AUC=0.5)")
ax.set_xlabel("Category", fontsize=12)
ax.set_ylabel("Mean AUC per video", fontsize=12)
ax.set_title("Model performance per anomaly category (Avenue --> UCF-Crime)", fontsize=13)
ax.set_ylim(0, max(stats["mean"] + stats["std"]) + 0.08)
ax.tick_params(axis="x", rotation=45)
ax.legend()
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200)
plt.close()
print(f"Saved plot to: {OUT_PNG}")