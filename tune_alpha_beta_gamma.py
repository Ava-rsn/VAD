"""
Tune alpha/beta/gamma weighting on Avenue in-domain (including cls head).
Uses already-saved raw score arrays — no model inference needed.

Usage: python tune_alpha_beta_gamma.py
"""

import numpy as np
from sklearn import metrics
from util.abnormal_utils import filt

# ── config ────────────────────────────────────────────────────────────────
DATA_DIR       = "/user/home/gk23779/evaluations/avenue_avenue_files"  # adjust path
DATA_DIR_CLS   = "/user/home/gk23779/evaluations/avenue_cls_head_files"  # adjust path for cls checkpoint
DATASET_PREFIX = "avenue"
RANGE          = 38
MU             = 11
NORMALIZE      = False
# ──────────────────────────────────────────────────────────────────────────

# Load raw scores from non-cls checkpoint
teacher = np.load(f"{DATA_DIR}/{DATASET_PREFIX}_teacher_raw.npy", allow_pickle=True).astype(np.float32)
st      = np.load(f"{DATA_DIR}/{DATASET_PREFIX}_st_raw.npy",      allow_pickle=True).astype(np.float32)

# Load raw scores from cls checkpoint
teacher_cls = np.load(f"{DATA_DIR_CLS}/{DATASET_PREFIX}_teacher_raw.npy", allow_pickle=True).astype(np.float32)
st_cls      = np.load(f"{DATA_DIR_CLS}/{DATASET_PREFIX}_st_raw.npy",      allow_pickle=True).astype(np.float32)

# Load labels/videos (same for both checkpoints)
labels  = np.load(f"{DATA_DIR_CLS}/{DATASET_PREFIX}_labels.npy", allow_pickle=True)
videos  = np.load(f"{DATA_DIR_CLS}/{DATASET_PREFIX}_videos.npy", allow_pickle=True)

labels = np.asarray(labels)
videos = np.asarray(videos)

# Verify arrays match in length
assert len(teacher_cls) == len(teacher), f"Length mismatch: cls={len(teacher_cls)}, non-cls={len(teacher)}"
print(f"Loaded {len(teacher)} frames, {len(np.unique(videos))} videos")


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

        lbl_a = np.array([0] + list(lbl) + [1])
        pred_a = np.array([0] + list(pred) + [1])
        fpr, tpr, _ = metrics.roc_curve(lbl_a, pred_a)
        aucs_anchored.append(metrics.auc(fpr, tpr))

        if np.unique(lbl).size >= 2:
            fpr, tpr, _ = metrics.roc_curve(lbl, pred)
            aucs_skip.append(metrics.auc(fpr, tpr))

    macro_anchored = np.nanmean(aucs_anchored) if aucs_anchored else np.nan
    macro_skip = np.nanmean(aucs_skip) if aucs_skip else np.nan

    fpr, tpr, _ = metrics.roc_curve(labels, filtered_scores)
    micro_auc = metrics.auc(fpr, tpr)

    return micro_auc, macro_anchored, macro_skip


# ── Grid search over alpha/beta/gamma ─────────────────────────────────────
# Using cls checkpoint's teacher and st scores, plus treating the
# non-cls teacher as the "base" and cls as "cls-enhanced"
#
# Strategy: alpha * teacher_cls + beta * st_cls + gamma * (teacher_cls - teacher)
# OR simpler: just weight the three raw signals from the cls checkpoint
# since that checkpoint already has the cls head baked in.
#
# Actually, the cls head output is embedded in the teacher_cls scores
# if the model was trained with it. Let's just do the straightforward
# three-signal combination from the cls checkpoint:

print(f"Tuning alpha/beta/gamma on {DATASET_PREFIX}")
print(f"Post-processing: range={RANGE}, mu={MU}, norm={NORMALIZE}")
print(f"\n{'alpha':>6s}  {'beta':>6s}  {'gamma':>6s}  {'Micro':>8s}  {'Macro(anch)':>12s}  {'Macro(skip)':>12s}")
print("-" * 65)

best_micro = 0
best_params = (1.0, 0.0, 0.0)
best_result = None

results = []

# Search: alpha weights teacher_cls, beta weights st_cls,
#         gamma weights a "cls boost" = difference between cls and non-cls teacher
cls_boost = teacher_cls - teacher  # the additional signal from having the cls head

for alpha_int in range(0, 11):
    for beta_int in range(0, 11):
        for gamma_int in range(0, 11):
            alpha = alpha_int / 10.0
            beta = beta_int / 10.0
            gamma = gamma_int / 10.0
            if alpha == 0 and beta == 0 and gamma == 0:
                continue

            combined = alpha * teacher_cls + beta * st_cls + gamma * cls_boost

            micro, macro_anch, macro_skip = evaluate_weighted(
                combined, labels, videos, RANGE, MU, NORMALIZE
            )

            results.append((alpha, beta, gamma, micro, macro_anch, macro_skip))

            if micro > best_micro:
                best_micro = micro
                best_params = (alpha, beta, gamma)
                best_result = (micro, macro_anch, macro_skip)

# Print top 20
results.sort(key=lambda x: -x[2])
for alpha, beta, gamma, micro, macro_anch, macro_skip in results[:20]:
    marker = " <-- BEST" if (alpha, beta, gamma) == best_params else ""
    print(f"{alpha:6.1f}  {beta:6.1f}  {gamma:6.1f}  {micro:8.4f}  {macro_anch:12.4f}  {macro_skip:12.4f}{marker}")

print(f"\nBest: alpha={best_params[0]:.1f}, beta={best_params[1]:.1f}, gamma={best_params[2]:.1f}")
print(f"  Micro AUC:        {best_result[0]:.6f}")
print(f"  Macro (anchored): {best_result[1]:.6f}")
print(f"  Macro (skip):     {best_result[2]:.6f}")

# ── Baselines ─────────────────────────────────────────────────────────────
print("\n- Baselines -")

baselines = [
    ("Teacher (no cls)",     teacher,                   "non-cls checkpoint"),
    ("Teacher (cls)",        teacher_cls,               "cls checkpoint"),
    ("ST (no cls)",          st,                        "non-cls checkpoint"),
    ("ST (cls)",             st_cls,                    "cls checkpoint"),
    ("T+ST (no cls)",        teacher + st,              "non-cls checkpoint"),
    ("T+ST (cls)",           teacher_cls + st_cls,      "cls checkpoint"),
    ("T_cls + cls_boost",    teacher_cls + cls_boost,   "double cls weight"),
]

for name, scores_bl, desc in baselines:
    micro, macro_anch, macro_skip = evaluate_weighted(
        scores_bl, labels, videos, RANGE, MU, NORMALIZE
    )
    print(f"{name:22s}  Micro={micro:.4f}  Macro(anch)={macro_anch:.4f}  Macro(skip)={macro_skip:.4f}  [{desc}]")