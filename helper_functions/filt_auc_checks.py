import os
import numpy as np
import pandas as pd
from sklearn import metrics
from util.abnormal_utils import filt

# ── config ────────────────────────────────────────────────────────────────
DATA_DIR       = "/user/home/gk23779/evaluations/ucf_shanghai_files"
DATASET_PREFIX = "ucf_crime"
SCORE_TYPE     = "teacher"   # teacher, st, combined
RANGE          = 1000
MU             = 350
NORMALIZE      = True
TEST_LIST      = "ucf_og_test.txt"
OUT_CSV        = f"{DATASET_PREFIX}_{SCORE_TYPE}_filt_r{RANGE}_mu{MU}.csv"
# ──────────────────────────────────────────────────────────────────────────

scores = np.load(os.path.join(DATA_DIR, f"{DATASET_PREFIX}_{SCORE_TYPE}_raw.npy"), allow_pickle=True)
labels = np.load(os.path.join(DATA_DIR, f"{DATASET_PREFIX}_labels.npy"),           allow_pickle=True)
videos = np.load(os.path.join(DATA_DIR, f"{DATASET_PREFIX}_videos.npy"),           allow_pickle=True)
frames = np.load(os.path.join(DATA_DIR, f"{DATASET_PREFIX}_frames.npy"),           allow_pickle=True)

scores = np.asarray(scores, dtype=np.float32)
labels = np.asarray(labels)
videos = np.asarray(videos)
frames = np.asarray(frames)

if DATASET_PREFIX == "ucf_crime" and TEST_LIST.strip():
    with open(TEST_LIST) as f:
        test_set = set(line.strip() for line in f if line.strip())
    test_mask = np.isin(videos, list(test_set))
    scores = scores[test_mask]
    labels = labels[test_mask]
    frames = frames[test_mask]
    videos = videos[test_mask]
    print(f"Filtered to {len(np.unique(videos))} test videos ({test_mask.sum()} frames)")

filtered_scores = np.zeros_like(scores, dtype=np.float32)

# per-video loop — collect data for both macro variants
aucs_anchored = []   # Ristea et al. style: [0]+lbl+[1]
aucs_skip     = []   # skip videos with only one class

for vid in np.unique(videos):
    mask = (videos == vid)
    pred = scores[mask].copy()
    lbl  = labels[mask]

    pred = filt(pred, range=RANGE, mu=MU)
    pred = np.nan_to_num(pred, nan=0.0)

    if len(pred) != mask.sum():
        min_len = min(len(pred), mask.sum())
        pred = pred[:min_len]
        lbl  = lbl[:min_len]

    if NORMALIZE:
        mn, mx = np.min(pred), np.max(pred)
        if mx - mn > 1e-8:
            pred = (pred - mn) / (mx - mn)
        else:
            pred = np.zeros_like(pred)

    filtered_scores[mask] = pred

    # anchored macro (all videos, Ristea style)
    lbl_a  = np.array([0] + list(lbl)  + [1])
    pred_a = np.array([0] + list(pred) + [1])
    fpr, tpr, _ = metrics.roc_curve(lbl_a, pred_a)
    aucs_anchored.append(metrics.auc(fpr, tpr))

    # skip macro (only videos with both classes)
    if np.unique(lbl).size >= 2:
        fpr, tpr, _ = metrics.roc_curve(lbl, pred)
        aucs_skip.append(metrics.auc(fpr, tpr))

macro_anchored = np.nanmean(aucs_anchored) if aucs_anchored else np.nan
macro_skip     = np.nanmean(aucs_skip)     if aucs_skip     else np.nan

fpr, tpr, _ = metrics.roc_curve(labels, filtered_scores)
micro_auc = metrics.auc(fpr, tpr)

print(f"score={SCORE_TYPE}  range={RANGE}  mu={MU}  normalize={NORMALIZE}")
print(f"Micro AUC                          : {micro_auc:.6f}")
print(f"Macro AUC (anchored, n={len(aucs_anchored):3d} videos) : {macro_anchored:.6f}")
print(f"Macro AUC (skip,     n={len(aucs_skip):3d} videos) : {macro_skip:.6f}")

out_df = pd.DataFrame({
    "video":                    videos,
    "frame":                    frames,
    "label":                    labels,
    f"score_{SCORE_TYPE}_filt": filtered_scores,
})
out_df = out_df.sort_values(["video", "frame"]).reset_index(drop=True)
out_df.to_csv(OUT_CSV, index=False)
print(f"Saved per-frame CSV to: {OUT_CSV}")