import os
import numpy as np
import pandas as pd
from sklearn import metrics
from scipy.stats import rankdata
from util.abnormal_utils import filt

# ── config ────────────────────────────────────────────────────────────────
AVENUE_DIR     = "ucf_avenue_files"
SHANGHAI_DIR   = "ucf_shanghai_files"
DATASET_PREFIX = "ucf_crime"
TEST_LIST      = ""
OUT_DIR        = "fusion_results"

SHANGHAI_RANGE = 1000
SHANGHAI_MU    = 350
SHANGHAI_NORM  = True
# ──────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)

labels = np.load(os.path.join(SHANGHAI_DIR, f"{DATASET_PREFIX}_labels.npy"), allow_pickle=True)
videos = np.load(os.path.join(SHANGHAI_DIR, f"{DATASET_PREFIX}_videos.npy"), allow_pickle=True)
frames = np.load(os.path.join(SHANGHAI_DIR, f"{DATASET_PREFIX}_frames.npy"), allow_pickle=True)

# avenue: raw st scores (no filtering — best as found)
scores_av = np.load(os.path.join(AVENUE_DIR,   f"{DATASET_PREFIX}_st_raw.npy"),      allow_pickle=True).astype(np.float32)
# shanghai: teacher raw scores
scores_sh = np.load(os.path.join(SHANGHAI_DIR, f"{DATASET_PREFIX}_teacher_raw.npy"), allow_pickle=True).astype(np.float32)

if TEST_LIST.strip():
    with open(TEST_LIST) as f:
        test_set = set(line.strip() for line in f if line.strip())
    mask     = np.isin(videos, list(test_set))
    labels   = labels[mask]
    videos   = videos[mask]
    frames   = frames[mask]
    scores_av = scores_av[mask]
    scores_sh = scores_sh[mask]
    print(f"Filtered to {len(np.unique(videos))} test videos ({mask.sum()} frames)")


def normalise_only(scores, videos):
    out = np.zeros_like(scores, dtype=np.float32)
    for vid in np.unique(videos):
        vmask = (videos == vid)
        pred  = scores[vmask].astype(np.float32)
        mn, mx = np.min(pred), np.max(pred)
        if mx - mn > 1e-8:
            pred = (pred - mn) / (mx - mn)
        else:
            pred = np.zeros_like(pred)
        out[vmask] = pred
    return out


def preprocess_scores(scores, videos, range_val, mu_val, normalize):
    out = np.zeros_like(scores, dtype=np.float32)
    for vid in np.unique(videos):
        vmask = (videos == vid)
        pred  = scores[vmask].copy()
        pred  = filt(pred, range=range_val, mu=mu_val)
        pred  = np.nan_to_num(pred, nan=0.0)
        if len(pred) != vmask.sum():
            min_len = min(len(pred), vmask.sum())
            pred    = pred[:min_len]
        if normalize:
            mn, mx = np.min(pred), np.max(pred)
            if mx - mn > 1e-8:
                pred = (pred - mn) / (mx - mn)
            else:
                pred = np.zeros_like(pred)
        out[vmask] = pred
    return out


def evaluate(scores, labels, videos, name):
    # micro
    fpr, tpr, _ = metrics.roc_curve(labels, scores)
    micro = metrics.auc(fpr, tpr)

    # macro skip (only videos with both classes)
    aucs_skip = []
    # macro anchored (all videos)
    aucs_anchored = []
    for vid in np.unique(videos):
        vmask  = (videos == vid)
        lbl    = labels[vmask]
        pred   = scores[vmask]
        lbl_a  = np.array([0] + list(lbl)  + [1])
        pred_a = np.array([0] + list(pred) + [1])
        fpr, tpr, _ = metrics.roc_curve(lbl_a, pred_a)
        aucs_anchored.append(metrics.auc(fpr, tpr))
        if np.unique(lbl).size >= 2:
            fpr, tpr, _ = metrics.roc_curve(lbl, pred)
            aucs_skip.append(metrics.auc(fpr, tpr))

    macro_skip     = float(np.nanmean(aucs_skip))     if aucs_skip     else float('nan')
    macro_anchored = float(np.nanmean(aucs_anchored)) if aucs_anchored else float('nan')

    print(f"{name:<45} Micro={micro:.4f}  MacroSkip={macro_skip:.4f}  MacroAnchored={macro_anchored:.4f}")
    return micro, macro_skip, macro_anchored


def rank_fuse(a, b, alpha=0.5):
    ra = rankdata(a) / len(a)
    rb = rankdata(b) / len(b)
    return alpha * ra + (1 - alpha) * rb


# ── prepare scores ────────────────────────────────────────────────────────
# avenue: raw st (best as found), normalise only for fusion
av_raw  = scores_av
av_norm = normalise_only(scores_av, videos)

# shanghai: filtered + normalised (best as found)
sh_filt = preprocess_scores(scores_sh, videos, SHANGHAI_RANGE, SHANGHAI_MU, SHANGHAI_NORM)
sh_norm = sh_filt  # already normalised

print("\n" + "="*70)
print("FUSION RESULTS")
print("="*70)

results = {}

micro, ms, ma = evaluate(av_raw,  labels, videos, "Avenue only (raw st)")
results["Avenue only"] = (micro, ms, ma)

micro, ms, ma = evaluate(sh_filt, labels, videos, "Shanghai only (filt+norm)")
results["Shanghai only"] = (micro, ms, ma)

# simple average
simple_avg = 0.5 * av_norm + 0.5 * sh_norm
micro, ms, ma = evaluate(simple_avg, labels, videos, "Simple average (both normalised)")
results["Simple average"] = (micro, ms, ma)

# weighted average sweep — optimise for MacroSkip
print("\nWeighted average sweep (optimising MacroSkip):")
best_alpha_skip, best_macro_skip = 0.5, 0.0
best_alpha_anch, best_macro_anch = 0.5, 0.0
for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    fused = alpha * av_norm + (1 - alpha) * sh_norm
    _, ms, ma = evaluate(fused, labels, videos, f"  Weighted alpha(av)={alpha:.1f}")
    if ms > best_macro_skip:
        best_macro_skip  = ms
        best_alpha_skip  = alpha
    if ma > best_macro_anch:
        best_macro_anch  = ma
        best_alpha_anch  = alpha

print(f"\nBest alpha for MacroSkip     = {best_alpha_skip} -> {best_macro_skip:.4f}")
print(f"Best alpha for MacroAnchored = {best_alpha_anch} -> {best_macro_anch:.4f}")

best_weighted_skip = best_alpha_skip * av_norm + (1 - best_alpha_skip) * sh_norm
micro, ms, ma = evaluate(best_weighted_skip, labels, videos, f"Weighted (best MacroSkip a={best_alpha_skip})")
results[f"Weighted (best skip a={best_alpha_skip})"] = (micro, ms, ma)

best_weighted_anch = best_alpha_anch * av_norm + (1 - best_alpha_anch) * sh_norm
micro, ms, ma = evaluate(best_weighted_anch, labels, videos, f"Weighted (best MacroAnch a={best_alpha_anch})")
results[f"Weighted (best anch a={best_alpha_anch})"] = (micro, ms, ma)

# rank fusion sweep
print("\nRank fusion sweep:")
best_alpha_rank_skip, best_rank_skip = 0.5, 0.0
for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    fused = rank_fuse(av_raw, sh_filt, alpha=alpha)
    _, ms, ma = evaluate(fused, labels, videos, f"  Rank fusion alpha(av)={alpha:.1f}")
    if ms > best_rank_skip:
        best_rank_skip       = ms
        best_alpha_rank_skip = alpha

best_rank_fused = rank_fuse(av_raw, sh_filt, alpha=best_alpha_rank_skip)
micro, ms, ma = evaluate(best_rank_fused, labels, videos, f"Rank fusion (best a={best_alpha_rank_skip})")
results[f"Rank fusion (best a={best_alpha_rank_skip})"] = (micro, ms, ma)

# max fusion
max_fused = np.maximum(av_norm, sh_norm)
micro, ms, ma = evaluate(max_fused, labels, videos, "Max fusion")
results["Max fusion"] = (micro, ms, ma)

# ── summary ───────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"{'Method':<45} {'Micro':>8} {'MacroSkip':>10} {'MacroAnch':>10}")
print("-"*73)
for name, (micro, ms, ma) in results.items():
    print(f"{name:<45} {micro:>8.4f} {ms:>10.4f} {ma:>10.4f}")

# save best fusion CSV
best_fused_scores = best_weighted_skip
best_name = f"weighted_skip_a{best_alpha_skip}"

out_df = pd.DataFrame({
    "video":        videos,
    "frame":        frames,
    "label":        labels,
    "score_av":     av_raw,
    "score_sh":     sh_filt,
    "score_fused":  best_fused_scores,
})
out_df = out_df.sort_values(["video", "frame"]).reset_index(drop=True)
csv_path = os.path.join(OUT_DIR, f"ucf_fusion_{best_name}.csv")
out_df.to_csv(csv_path, index=False)
print(f"\nSaved best fusion CSV to: {csv_path}")