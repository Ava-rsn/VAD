import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

CSV_PATH = "ucf_crime_teacher_filt_r1000_mu350.csv"

# change this to whichever one you want to inspect
SCORE_COL = "score_teacher_filt"
OUT_DIR = f"ucf_shanghai_files/plots_filt_ucf_shanghai_{SCORE_COL}"

os.makedirs(OUT_DIR, exist_ok=True)


def extract_frame_number(x):
    if pd.isna(x):
        return np.nan
    s = str(x).split("/")[-1]   # e.g. "000123" or "000123.jpg"
    s = s.split(".")[0]
    return int(s)


def shade_abnormal_regions(ax, frames, labels):
    frames = np.asarray(frames)
    labels = np.asarray(labels)

    in_seg = False
    start = None

    for i in range(len(labels)):
        if labels[i] == 1 and not in_seg:
            start = frames[i]
            in_seg = True
        elif labels[i] == 0 and in_seg:
            ax.axvspan(start, frames[i - 1], alpha=0.2)
            in_seg = False

    if in_seg:
        ax.axvspan(start, frames[-1], alpha=0.2)


df = pd.read_csv(CSV_PATH)

required_cols = ["video", "frame", "label", SCORE_COL]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing required column: {c}")

df["frame_num"] = df["frame"].apply(extract_frame_number)
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df[SCORE_COL] = pd.to_numeric(df[SCORE_COL], errors="coerce")

df = df.dropna(subset=["video", "frame_num", "label", SCORE_COL]).copy()
df["frame_num"] = df["frame_num"].astype(int)
df["label"] = df["label"].astype(int)

df = df.sort_values(["video", "frame_num"]).reset_index(drop=True)

print("Rows:", len(df))
print("Videos:", df["video"].nunique())
print("Using score column:", SCORE_COL)

# -----------------------------
# Global histogram
# -----------------------------
plt.figure(figsize=(8, 5))
plt.hist(df[df["label"] == 0][SCORE_COL], bins=100, alpha=0.6, label="Normal")
plt.hist(df[df["label"] == 1][SCORE_COL], bins=100, alpha=0.6, label="Abnormal")
plt.xlabel(SCORE_COL)
plt.ylabel("Count")
plt.title(f"Global score distribution: {SCORE_COL}")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "global_histogram.png"), dpi=200)
plt.close()

# -----------------------------
# Per-video AUC stats
# -----------------------------
rows = []
for vid, g in df.groupby("video"):
    row = {
        "video": vid,
        "n_frames": len(g),
        "mean_score": g[SCORE_COL].mean(),
        "std_score": g[SCORE_COL].std(),
        "min_score": g[SCORE_COL].min(),
        "max_score": g[SCORE_COL].max(),
        "frac_abnormal": g["label"].mean(),
    }

    if g["label"].nunique() >= 2:
        row["auc"] = roc_auc_score(g["label"], g[SCORE_COL])
    else:
        row["auc"] = np.nan

    rows.append(row)

video_stats = pd.DataFrame(rows).sort_values("auc", na_position="last")
video_stats.to_csv(os.path.join(OUT_DIR, "video_stats.csv"), index=False)

print("\nWorst 20 abnormal videos by AUC:")
print(video_stats[video_stats["auc"].notna()].head(20)[["video", "auc", "mean_score", "max_score", "frac_abnormal"]])

print("\nBest 20 abnormal videos by AUC:")
print(video_stats[video_stats["auc"].notna()].tail(20)[["video", "auc", "mean_score", "max_score", "frac_abnormal"]])

# -----------------------------
# Per-video AUC histogram
# -----------------------------
valid_auc = video_stats["auc"].dropna()

plt.figure(figsize=(8, 5))
plt.hist(valid_auc, bins=30)
plt.xlabel("Per-video AUC")
plt.ylabel("Number of videos")
plt.title(f"Per-video AUC distribution: {SCORE_COL}")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "per_video_auc_hist.png"), dpi=200)
plt.close()

# -----------------------------
# Per-video scale scatter
# -----------------------------
plt.figure(figsize=(8, 5))
plt.scatter(video_stats["mean_score"], video_stats["max_score"])
plt.xlabel("Mean score per video")
plt.ylabel("Max score per video")
plt.title(f"Per-video score scale: {SCORE_COL}")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "per_video_scale_scatter.png"), dpi=200)
plt.close()

# -----------------------------
# Plot worst and best abnormal videos
# -----------------------------
worst_dir = os.path.join(OUT_DIR, "worst_abnormal_videos")
best_dir = os.path.join(OUT_DIR, "best_abnormal_videos")
normal_dir = os.path.join(OUT_DIR, "sample_normal_videos")
os.makedirs(worst_dir, exist_ok=True)
os.makedirs(best_dir, exist_ok=True)
os.makedirs(normal_dir, exist_ok=True)

abnormal_stats = video_stats[video_stats["auc"].notna()].copy()
normal_stats = video_stats[video_stats["auc"].isna()].copy()

worst_vids = abnormal_stats.head(12)["video"].tolist()
best_vids = abnormal_stats.tail(12)["video"].tolist()

normal_stats_true = video_stats[(video_stats["auc"].isna()) & 
                                 (video_stats["frac_abnormal"] == 0)].copy()
all_anom_stats    = video_stats[(video_stats["auc"].isna()) & 
                                 (video_stats["frac_abnormal"] == 1)].copy()

sample_normals   = normal_stats_true.head(12)["video"].tolist()
sample_all_anoms = all_anom_stats.head(12)["video"].tolist()

all_anom_dir = os.path.join(OUT_DIR, "all_anomalous_videos")
os.makedirs(all_anom_dir, exist_ok=True)


def save_video_plot(vid, out_folder):
    g = df[df["video"] == vid].sort_values("frame_num")
    auc_vals = video_stats.loc[video_stats["video"] == vid, "auc"].values
    auc_val = auc_vals[0] if len(auc_vals) > 0 else np.nan

    plt.figure(figsize=(12, 4))
    plt.plot(g["frame_num"], g[SCORE_COL])
    shade_abnormal_regions(plt.gca(), g["frame_num"], g["label"])
    plt.xlabel("Frame")
    plt.ylabel(SCORE_COL)
    plt.title(f"{vid} | AUC={auc_val}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, f"{vid}.png"), dpi=200)
    plt.close()

for vid in worst_vids:
    save_video_plot(vid, worst_dir)

for vid in best_vids:
    save_video_plot(vid, best_dir)

for vid in sample_normals:
    save_video_plot(vid, normal_dir)

for vid in sample_all_anoms:
    save_video_plot(vid, all_anom_dir)


# -----------------------------
# Summary
# -----------------------------
micro_auc = roc_auc_score(df["label"], df[SCORE_COL])

with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
    f.write(f"Score column: {SCORE_COL}\n")
    f.write(f"Rows: {len(df)}\n")
    f.write(f"Videos: {df['video'].nunique()}\n")
    f.write(f"Micro AUC from CSV: {micro_auc:.6f}\n")
    f.write(f"Mean per-video AUC: {valid_auc.mean():.6f}\n")
    f.write(f"Median per-video AUC: {valid_auc.median():.6f}\n")

print(f"\nSaved everything to: {OUT_DIR}")
print(f"Micro AUC from CSV ({SCORE_COL}): {micro_auc:.6f}")