"""
Tune alpha/beta weighting on Avenue in-domain.
Uses already-saved raw score arrays — no model inference needed.

Usage: python tune_alpha_beta.py
"""

import numpy as np
from sklearn import metrics
from util.abnormal_utils import filt

# ── config ────────────────────────────────────────────────────────────────
DATA_DIR       = "/user/home/gk23779/evaluations/avenue_cls_head_files"  # adjust to your Avenue path
DATASET_PREFIX = "avenue"
RANGE          = 38
MU             = 11
NORMALIZE      = False
# ──────────────────────────────────────────────────────────────────────────

# Load raw scores
teacher = np.load(f"{DATA_DIR}/{DATASET_PREFIX}_teacher_raw.npy", allow_pickle=True).astype(np.float32)
st      = np.load(f"{DATA_DIR}/{DATASET_PREFIX}_st_raw.npy",      allow_pickle=True).astype(np.float32)
labels  = np.load(f"{DATA_DIR}/{DATASET_PREFIX}_labels.npy",      allow_pickle=True)
videos  = np.load(f"{DATA_DIR}/{DATASET_PREFIX}_videos.npy",      allow_pickle=True)

labels = np.asarray(labels)
videos = np.asarray(videos)


def evaluate_weighted(scores, labels, videos, range_val, mu_val, normalize):
    """Compute micro, macro anchored, macro skip for a given score array."""
    filtered_scores = np.zeros_like(scores, dtype=np.float32)
    aucs_anchored = []
    aucs_skip = []

    for vid in np.unique(videos):
        mask = (videos == vid)
        pred = scores[mask].copy()
        lbl = labels[mask]

        pred = filt(pred, range=range_val, mu=mu_val)
        pred = np.nan_to_num(pred, nan=0.0)

        if len(pred) != mask.sum():
            min_len = min(len(pred), mask.sum())
            pred = pred[:min_len]
            lbl = lbl[:min_len]

        if normalize:
            mn, mx = np.min(pred), np.max(pred)
            if mx - mn > 1e-8:
                pred = (pred - mn) / (mx - mn)
            else:
                pred = np.zeros_like(pred)

        filtered_scores[mask] = pred

        # anchored macro
        lbl_a = np.array([0] + list(lbl) + [1])
        pred_a = np.array([0] + list(pred) + [1])
        fpr, tpr, _ = metrics.roc_curve(lbl_a, pred_a)
        aucs_anchored.append(metrics.auc(fpr, tpr))

        # skip macro
        if np.unique(lbl).size >= 2:
            fpr, tpr, _ = metrics.roc_curve(lbl, pred)
            aucs_skip.append(metrics.auc(fpr, tpr))

    macro_anchored = np.nanmean(aucs_anchored) if aucs_anchored else np.nan
    macro_skip = np.nanmean(aucs_skip) if aucs_skip else np.nan

    fpr, tpr, _ = metrics.roc_curve(labels, filtered_scores)
    micro_auc = metrics.auc(fpr, tpr)

    return micro_auc, macro_anchored, macro_skip


# ── Grid search over alpha/beta ──────────────────────────────────────────
print(f"Tuning alpha/beta on {DATASET_PREFIX} (range={RANGE}, mu={MU}, norm={NORMALIZE})")
print(f"{'alpha':>6s}  {'beta':>6s}  {'Micro':>8s}  {'Macro(anch)':>12s}  {'Macro(skip)':>12s}")
print("-" * 52)

best_micro = 0
best_alpha = 1.0
best_beta = 0.0
best_result = None

results = []

for alpha_int in range(0, 11):
    for beta_int in range(0, 11):
        alpha = alpha_int / 10.0
        beta = beta_int / 10.0
        if alpha == 0 and beta == 0:
            continue

        combined = alpha * teacher + beta * st
        micro, macro_anch, macro_skip = evaluate_weighted(
            combined, labels, videos, RANGE, MU, NORMALIZE
        )

        results.append((alpha, beta, micro, macro_anch, macro_skip))

        if micro > best_micro:
            best_micro = micro
            best_alpha = alpha
            best_beta = beta
            best_result = (micro, macro_anch, macro_skip)

# Print all results sorted by micro AUC
results.sort(key=lambda x: -x[2])
for alpha, beta, micro, macro_anch, macro_skip in results[:15]:
    marker = " <-- BEST" if (alpha == best_alpha and beta == best_beta) else ""
    print(f"{alpha:6.1f}  {beta:6.1f}  {micro:8.4f}  {macro_anch:12.4f}  {macro_skip:12.4f}{marker}")

print(f"\nBest: alpha={best_alpha:.1f}, beta={best_beta:.1f}")
print(f"  Micro AUC:      {best_result[0]:.6f}")
print(f"  Macro (anchored): {best_result[1]:.6f}")
print(f"  Macro (skip):     {best_result[2]:.6f}")

# Compare against baselines
print("\n-Baselines -")
for name, a, b in [("Teacher only", 1.0, 0.0), ("ST only", 0.0, 1.0), ("Equal sum", 1.0, 1.0)]:
    combined = a * teacher + b * st
    micro, macro_anch, macro_skip = evaluate_weighted(
        combined, labels, videos, RANGE, MU, NORMALIZE
    )
    print(f"{name:15s}  (a={a:.1f}, b={b:.1f})  Micro={micro:.4f}  Macro(anch)={macro_anch:.4f}  Macro(skip)={macro_skip:.4f}")