#!/usr/bin/env python3
"""
Temporal bias analysis for post-processing in VAD.

Goal:
Check whether strong temporal post-processing is boosting AUC partly by
imposing a bell-shaped temporal prior that favours anomalies near the
middle of a video.

What this script adds:
1) GT temporal distribution:
   - start vs duration heatmap
   - midpoint histogram
   - midpoint vs duration heatmap

2) Performance vs anomaly location:
   - raw model AUC vs anomaly midpoint
   - post-processed model AUC vs anomaly midpoint
   - null-model AUC vs anomaly midpoint

3) Null-model comparison:
   - actual vs null per-video AUC
   - raw vs post vs null micro/macro comparison

4) Quantitative stats:
   - Spearman correlation between midpoint and AUC
   - middle-vs-edge AUC comparison
   - count of videos where null beats post, etc.

Usage example:
python temporal_bias_analysis_improved.py \
  --labels  shanghai_shanghai_files/shanghai_labels.npy \
  --videos  shanghai_shanghai_files/shanghai_videos.npy \
  --frames  shanghai_shanghai_files/shanghai_frames.npy \
  --scores  shanghai_shanghai_files/shanghai_teacher_raw.npy \
  --range_val 1000 --mu_val 350 --normalize \
  --dataset shanghai \
  --outdir shanghai_improved_temporal_bias_plots

True no-filter baseline:
python temporal_bias_analysis_improved.py \
  --labels  ucf_shanghai_files/ucf_crime_labels.npy \
  --videos  ucf_shanghai_files/ucf_crime_videos.npy \
  --frames  ucf_shanghai_files/ucf_crime_frames.npy \
  --scores  ucf_shanghai_files/ucf_crime_teacher_raw.npy \
  --range_val 1000 --mu_val 350 --normalize \
  --dataset ucf_crime \
  --outdir ucf_shanghai_improved_temporal_bias_plots


python temporal_bias_analysis_improved.py \
  --labels  avenue_avenue_files/avenue_labels.npy \
  --videos  avenue_avenue_files/avenue_videos.npy \
  --frames  avenue_avenue_files/avenue_frames.npy \
  --scores  avenue_avenue_files/avenue_teacher_raw.npy \
  --range_val 38 --mu_val 4 \
  --dataset avenue \
  --outdir avenue_improved_temporal_bias_plots
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from scipy.stats import binned_statistic_2d, spearmanr


# -----------------------------
# Repo-compatible filter
# -----------------------------
def gaussian_filter_repo(support, sigma):
    """
    Same behaviour as the repo:
    - centre is forced to the middle of support
    - passed sigma controls spread, not centre
    """
    mu = support[len(support) // 2 - 1]
    filt = (
        1.0 / (sigma * np.sqrt(2 * np.pi))
        * np.exp(-0.5 * ((support - mu) / sigma) ** 2)
    )
    return filt


def filt_repo(input_scores, dim=9, range_val=302, mu_val=21):
    """
    Same temporal smoothing logic as repo.
    The 3D filter is unused in the original code, so preserved only for compatibility.
    """
    _filter_3d = np.ones((dim, dim, dim)) / (dim ** 3)  # unused, kept for compatibility
    filter_2d = gaussian_filter_repo(np.arange(1, range_val), mu_val)

    frame_scores = input_scores
    padding_size = len(filter_2d) // 2
    padded = np.concatenate(
        (np.zeros(padding_size), frame_scores, np.zeros(padding_size))
    )
    out = np.correlate(padded, filter_2d, "valid")
    return out


def maybe_filter_scores(scores_1d, range_val, mu_val, do_filter):
    """
    True no-filter support.
    """
    scores_1d = np.asarray(scores_1d, dtype=np.float64)
    if not do_filter:
        return scores_1d.copy()
    if range_val <= 1:
        raise ValueError("range_val must be > 1 when filtering is enabled.")
    return filt_repo(scores_1d, range_val=range_val, mu_val=mu_val)


def preprocess_scores(scores_flat, videos_flat, range_val, mu_val, normalize, do_filter):
    """
    Per-video filtering and optional per-video min-max normalization.
    """
    out = np.zeros_like(scores_flat, dtype=np.float32)

    for vid in np.unique(videos_flat):
        mask = videos_flat == vid
        pred = scores_flat[mask].astype(np.float64).copy()

        pred = maybe_filter_scores(pred, range_val, mu_val, do_filter)
        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

        if normalize:
            mn, mx = pred.min(), pred.max()
            if mx - mn > 1e-8:
                pred = (pred - mn) / (mx - mn)
            else:
                pred = np.zeros_like(pred)

        out[mask] = pred.astype(np.float32)

    return out


def generate_null_profile(n_frames, range_val, mu_val, do_filter, normalize):
    """
    Null model:
    start from a constant score sequence and apply the same post-processing.
    This isolates the temporal prior induced by filtering/padding/normalization.
    """
    flat = np.ones(int(n_frames), dtype=np.float64)
    prof = maybe_filter_scores(flat, range_val, mu_val, do_filter)
    prof = np.nan_to_num(prof, nan=0.0, posinf=0.0, neginf=0.0)

    if normalize:
        mn, mx = prof.min(), prof.max()
        if mx - mn > 1e-8:
            prof = (prof - mn) / (mx - mn)
        else:
            prof = np.zeros_like(prof)

    return prof.astype(np.float32)


# -----------------------------
# Metrics and segment utilities
# -----------------------------
def compute_per_video_auc(scores, labels, videos):
    """
    Macro-skip style per-video AUC:
    skip videos that contain only one class.
    """
    auc_dict = {}
    for vid in np.unique(videos):
        mask = videos == vid
        lbl = labels[mask].astype(int)
        pred = scores[mask].astype(np.float32)

        if len(np.unique(lbl)) < 2:
            continue

        fpr, tpr, _ = roc_curve(lbl, pred)
        auc_dict[vid] = auc(fpr, tpr)
    return auc_dict


def compute_micro_auc(scores, labels):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float32)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    fpr, tpr, _ = roc_curve(labels, scores)
    return auc(fpr, tpr)


def get_anomaly_segments(labels_vid):
    """
    Return list of contiguous anomaly segments as (start_idx, end_idx), inclusive.
    """
    labels_vid = np.asarray(labels_vid).astype(int)
    segments = []
    in_seg = False
    start = 0

    for i, l in enumerate(labels_vid):
        if l == 1 and not in_seg:
            start = i
            in_seg = True
        elif l == 0 and in_seg:
            segments.append((start, i - 1))
            in_seg = False

    if in_seg:
        segments.append((start, len(labels_vid) - 1))

    return segments


def extract_segment_records(labels, videos, mode="all"):
    """
    Build segment-level records for GT temporal analysis.

    mode:
      - "all": keep every anomaly segment
      - "longest": keep only the longest segment per video
    """
    records = []

    for vid in np.unique(videos):
        mask = videos == vid
        lbl = labels[mask].astype(int)
        n = mask.sum()

        if len(np.unique(lbl)) < 2:
            continue

        segments = get_anomaly_segments(lbl)
        if not segments:
            continue

        if mode == "longest":
            segments = [max(segments, key=lambda s: s[1] - s[0])]

        for (s, e) in segments:
            start_n = s / n
            end_n = e / n
            dur_n = (e - s + 1) / n
            mid_n = (start_n + end_n) / 2.0
            records.append(
                {
                    "video": vid,
                    "n_frames": n,
                    "start": start_n,
                    "end": end_n,
                    "duration": dur_n,
                    "midpoint": mid_n,
                }
            )

    return records


def build_video_level_location_summary(labels, videos, mode="weighted_mean"):
    """
    One location summary per video so it can be paired with per-video AUC.

    mode:
      - "longest": use longest anomaly segment
      - "weighted_mean": use duration-weighted mean midpoint/start/duration over segments
    """
    out = {}

    for vid in np.unique(videos):
        mask = videos == vid
        lbl = labels[mask].astype(int)
        n = mask.sum()

        if len(np.unique(lbl)) < 2:
            continue

        segments = get_anomaly_segments(lbl)
        if not segments:
            continue

        if mode == "longest":
            s, e = max(segments, key=lambda seg: seg[1] - seg[0])
            start_n = s / n
            end_n = e / n
            dur_n = (e - s + 1) / n
            mid_n = (start_n + end_n) / 2.0
        else:
            starts = []
            mids = []
            durs = []

            for s, e in segments:
                start_n = s / n
                end_n = e / n
                dur_n = (e - s + 1) / n
                mid_n = (start_n + end_n) / 2.0
                starts.append(start_n)
                mids.append(mid_n)
                durs.append(dur_n)

            durs = np.array(durs, dtype=np.float64)
            weights = durs / durs.sum()
            start_n = float(np.sum(np.array(starts) * weights))
            mid_n = float(np.sum(np.array(mids) * weights))
            dur_n = float(np.sum(durs))

        out[vid] = {
            "start": start_n,
            "midpoint": mid_n,
            "duration": dur_n,
        }

    return out


def compare_middle_vs_edge(midpoints, aucs, middle_band=(0.4, 0.6), edge_band=0.25):
    """
    Compare videos with anomalies near the middle against anomalies near the edges.
    """
    midpoints = np.asarray(midpoints, dtype=np.float64)
    aucs = np.asarray(aucs, dtype=np.float64)
    valid = ~np.isnan(midpoints) & ~np.isnan(aucs)

    midpoints = midpoints[valid]
    aucs = aucs[valid]

    is_middle = (midpoints >= middle_band[0]) & (midpoints <= middle_band[1])
    is_edge = (midpoints <= edge_band) | (midpoints >= 1.0 - edge_band)

    middle_vals = aucs[is_middle]
    edge_vals = aucs[is_edge]

    result = {
        "n_middle": len(middle_vals),
        "n_edge": len(edge_vals),
        "mean_middle": np.nan if len(middle_vals) == 0 else float(np.mean(middle_vals)),
        "mean_edge": np.nan if len(edge_vals) == 0 else float(np.mean(edge_vals)),
        "median_middle": np.nan if len(middle_vals) == 0 else float(np.median(middle_vals)),
        "median_edge": np.nan if len(edge_vals) == 0 else float(np.median(edge_vals)),
    }

    if len(middle_vals) > 0 and len(edge_vals) > 0:
        result["mean_gap_middle_minus_edge"] = float(np.mean(middle_vals) - np.mean(edge_vals))
    else:
        result["mean_gap_middle_minus_edge"] = np.nan

    return result


# -----------------------------
# Plot helpers
# -----------------------------
def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_gt_distribution(segment_records, dataset, outdir):
    starts = np.array([r["start"] for r in segment_records], dtype=np.float64)
    durs = np.array([r["duration"] for r in segment_records], dtype=np.float64)
    mids = np.array([r["midpoint"] for r in segment_records], dtype=np.float64)

    # Figure 1: start-duration heatmap + midpoint-duration heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    h = ax.hist2d(
        starts, durs, bins=25, range=[[0, 1], [0, 1]], cmap="YlOrRd", cmin=0.5
    )
    plt.colorbar(h[3], ax=ax, label="Number of anomaly segments")
    ax.set_xlabel("Normalised anomaly start")
    ax.set_ylabel("Normalised anomaly duration")
    ax.set_title(f"{dataset}: GT anomaly start vs duration")
    clean_axes(ax)

    ax = axes[1]
    h = ax.hist2d(
        mids, durs, bins=25, range=[[0, 1], [0, 1]], cmap="YlGnBu", cmin=0.5
    )
    plt.colorbar(h[3], ax=ax, label="Number of anomaly segments")
    ax.set_xlabel("Normalised anomaly midpoint")
    ax.set_ylabel("Normalised anomaly duration")
    ax.set_title(f"{dataset}: GT anomaly midpoint vs duration")
    clean_axes(ax)

    savefig(fig, os.path.join(outdir, f"{dataset}_gt_heatmaps.png"))

    # Figure 2: midpoint histogram
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.hist(mids, bins=20, range=(0, 1), edgecolor="black")
    ax.axvline(0.5, linestyle="--", linewidth=1.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Normalised anomaly midpoint")
    ax.set_ylabel("Count")
    ax.set_title(f"{dataset}: GT anomaly midpoint distribution")
    clean_axes(ax)

    savefig(fig, os.path.join(outdir, f"{dataset}_gt_midpoint_histogram.png"))


def plot_auc_location_maps(starts_vid, durs_vid, auc_raw, auc_post, auc_null, dataset, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    items = [
        (auc_raw, "Raw mean AUC by anomaly location"),
        (auc_post, "Post-processed mean AUC by anomaly location"),
        (auc_null, "Null-model mean AUC by anomaly location"),
    ]

    for ax, (vals, title) in zip(axes, items):
        stat = binned_statistic_2d(
            starts_vid,
            durs_vid,
            vals,
            statistic="mean",
            bins=15,
            range=[[0, 1], [0, 1]],
        )
        im = ax.imshow(
            stat.statistic.T,
            origin="lower",
            extent=[0, 1, 0, 1],
            aspect="auto",
            cmap="RdYlGn",
            vmin=0.0,
            vmax=1.0,
        )
        plt.colorbar(im, ax=ax, label="Mean per-video AUC")
        ax.set_xlabel("Normalised anomaly start")
        ax.set_ylabel("Normalised anomaly duration")
        ax.set_title(f"{dataset}: {title}")
        clean_axes(ax)

    savefig(fig, os.path.join(outdir, f"{dataset}_auc_location_maps.png"))


def plot_midpoint_vs_auc(midpoints, durations, auc_raw, auc_post, auc_null, dataset, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.4))

    items = [
        (auc_raw, "Raw model"),
        (auc_post, "Post-processed model"),
        (auc_null, "Null model"),
    ]

    x_fit = np.linspace(0, 1, 200)

    for ax, (vals, title) in zip(axes, items):
        vals = np.asarray(vals, dtype=np.float64)
        sc = ax.scatter(
            midpoints,
            vals,
            c=durations,
            cmap="viridis",
            alpha=0.75,
            s=42,
            edgecolors="none",
        )
        plt.colorbar(sc, ax=ax, label="Anomaly duration")
        ax.axhline(0.5, linestyle="--", alpha=0.7)
        ax.axvline(0.5, linestyle=":", alpha=0.9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Normalised anomaly midpoint")
        ax.set_ylabel("Per-video AUC")
        ax.set_title(f"{dataset}: {title} AUC vs midpoint")
        clean_axes(ax)

        valid = ~np.isnan(vals)
        if valid.sum() >= 3:
            coeffs = np.polyfit(midpoints[valid], vals[valid], deg=2)
            y_fit = np.polyval(coeffs, x_fit)
            ax.plot(x_fit, y_fit, linewidth=2.2)

    savefig(fig, os.path.join(outdir, f"{dataset}_midpoint_vs_auc_all.png"))


def plot_actual_vs_null(auc_post_arr, auc_null_arr, dataset, outdir):
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.scatter(auc_null_arr, auc_post_arr, alpha=0.65, s=35, edgecolors="none")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Null-model per-video AUC")
    ax.set_ylabel("Post-processed model per-video AUC")
    ax.set_title(f"{dataset}: Post-processed vs null per-video AUC")
    clean_axes(ax)

    if len(auc_post_arr) >= 2:
        corr = np.corrcoef(auc_null_arr, auc_post_arr)[0, 1]
        ax.text(
            0.05,
            0.93,
            f"Pearson r = {corr:.3f}",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round", alpha=0.2),
        )

    savefig(fig, os.path.join(outdir, f"{dataset}_post_vs_null_scatter.png"))


def plot_bar_comparison(micro_raw, micro_post, micro_null, macro_raw, macro_post, macro_null, dataset, outdir):
    labels = [
        "Raw\nMicro",
        "Post\nMicro",
        "Null\nMicro",
        "Raw\nMacro",
        "Post\nMacro",
        "Null\nMacro",
    ]
    vals = [micro_raw, micro_post, micro_null, macro_raw, macro_post, macro_null]

    fig, ax = plt.subplots(figsize=(9.5, 5.3))
    bars = ax.bar(labels, vals, edgecolor="black", linewidth=0.6)
    ax.axhline(0.5, linestyle="--", alpha=0.7)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("AUC")
    ax.set_title(f"{dataset}: Raw vs post-processed vs null")
    clean_axes(ax)

    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.015,
            f"{v:.3f}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    savefig(fig, os.path.join(outdir, f"{dataset}_raw_post_null_bars.png"))


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--videos", required=True)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--scores", required=True)

    parser.add_argument("--range_val", type=int, default=1000)
    parser.add_argument("--mu_val", type=int, default=350)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--no_filter", action="store_true")

    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--outdir", default="temporal_bias_plots")

    parser.add_argument(
        "--segment_mode",
        choices=["all", "longest"],
        default="all",
        help="For GT segment plots: use all anomaly segments or only longest per video.",
    )
    parser.add_argument(
        "--video_location_mode",
        choices=["weighted_mean", "longest"],
        default="weighted_mean",
        help="How to summarise anomaly location to one point per video for AUC pairing.",
    )

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    labels = np.load(args.labels, allow_pickle=True).astype(int)
    videos = np.load(args.videos, allow_pickle=True)
    _frames = np.load(args.frames, allow_pickle=True)  # kept for compatibility, not used
    scores_raw = np.load(args.scores, allow_pickle=True).astype(np.float32)

    if not (len(labels) == len(videos) == len(scores_raw)):
        raise ValueError("labels, videos, and scores must have the same length.")

    do_filter = not args.no_filter

    print(f"Loaded {len(np.unique(videos))} videos, {len(labels)} frames")
    print(
        f"Filtering={'ON' if do_filter else 'OFF'} | "
        f"range={args.range_val} | mu={args.mu_val} | normalize={args.normalize}"
    )
    print(
        f"GT segment mode={args.segment_mode} | "
        f"Per-video location summary={args.video_location_mode}"
    )

    # -------------------------
    # Score variants
    # -------------------------
    scores_raw_proc = preprocess_scores(
        scores_raw,
        videos,
        range_val=args.range_val,
        mu_val=args.mu_val,
        normalize=False,
        do_filter=False,
    )
    # raw means truly raw in plots/metrics

    scores_post = preprocess_scores(
        scores_raw,
        videos,
        range_val=args.range_val,
        mu_val=args.mu_val,
        normalize=args.normalize,
        do_filter=do_filter,
    )

    scores_null = np.zeros_like(scores_raw, dtype=np.float32)
    for vid in np.unique(videos):
        mask = videos == vid
        n = mask.sum()
        scores_null[mask] = generate_null_profile(
            n,
            range_val=args.range_val,
            mu_val=args.mu_val,
            do_filter=do_filter,
            normalize=args.normalize,
        )

    # -------------------------
    # Metrics
    # -------------------------
    auc_raw_dict = compute_per_video_auc(scores_raw_proc, labels, videos)
    auc_post_dict = compute_per_video_auc(scores_post, labels, videos)
    auc_null_dict = compute_per_video_auc(scores_null, labels, videos)

    micro_raw = compute_micro_auc(scores_raw_proc, labels)
    micro_post = compute_micro_auc(scores_post, labels)
    micro_null = compute_micro_auc(scores_null, labels)

    common_vids_raw = sorted(auc_raw_dict.keys())
    common_vids_post = sorted(auc_post_dict.keys())
    common_vids_null = sorted(auc_null_dict.keys())
    common_vids = sorted(set(common_vids_raw) & set(common_vids_post) & set(common_vids_null))

    macro_raw = float(np.mean([auc_raw_dict[v] for v in common_vids]))
    macro_post = float(np.mean([auc_post_dict[v] for v in common_vids]))
    macro_null = float(np.mean([auc_null_dict[v] for v in common_vids]))

    # -------------------------
    # GT segment-level analysis
    # -------------------------
    segment_records = extract_segment_records(labels, videos, mode=args.segment_mode)
    if len(segment_records) == 0:
        raise RuntimeError("No anomaly segments found.")

    plot_gt_distribution(segment_records, args.dataset, args.outdir)

    # -------------------------
    # Video-level location summary
    # -------------------------
    vid_loc = build_video_level_location_summary(
        labels, videos, mode=args.video_location_mode
    )

    mids = []
    starts = []
    durs = []
    auc_raw_arr = []
    auc_post_arr = []
    auc_null_arr = []
    vids_final = []

    for vid in common_vids:
        if vid not in vid_loc:
            continue
        mids.append(vid_loc[vid]["midpoint"])
        starts.append(vid_loc[vid]["start"])
        durs.append(vid_loc[vid]["duration"])
        auc_raw_arr.append(auc_raw_dict[vid])
        auc_post_arr.append(auc_post_dict[vid])
        auc_null_arr.append(auc_null_dict[vid])
        vids_final.append(vid)

    mids = np.array(mids, dtype=np.float64)
    starts = np.array(starts, dtype=np.float64)
    durs = np.array(durs, dtype=np.float64)
    auc_raw_arr = np.array(auc_raw_arr, dtype=np.float64)
    auc_post_arr = np.array(auc_post_arr, dtype=np.float64)
    auc_null_arr = np.array(auc_null_arr, dtype=np.float64)

    # -------------------------
    # Plots
    # -------------------------
    plot_auc_location_maps(starts, durs, auc_raw_arr, auc_post_arr, auc_null_arr, args.dataset, args.outdir)
    plot_midpoint_vs_auc(mids, durs, auc_raw_arr, auc_post_arr, auc_null_arr, args.dataset, args.outdir)
    plot_actual_vs_null(auc_post_arr, auc_null_arr, args.dataset, args.outdir)
    plot_bar_comparison(micro_raw, micro_post, micro_null, macro_raw, macro_post, macro_null, args.dataset, args.outdir)

    # -------------------------
    # Quantitative analysis
    # -------------------------
    def safe_spearman(x, y):
        if len(x) < 2:
            return np.nan, np.nan
        r, p = spearmanr(x, y, nan_policy="omit")
        return float(r), float(p)

    sp_raw_r, sp_raw_p = safe_spearman(mids, auc_raw_arr)
    sp_post_r, sp_post_p = safe_spearman(mids, auc_post_arr)
    sp_null_r, sp_null_p = safe_spearman(mids, auc_null_arr)

    raw_me = compare_middle_vs_edge(mids, auc_raw_arr)
    post_me = compare_middle_vs_edge(mids, auc_post_arr)
    null_me = compare_middle_vs_edge(mids, auc_null_arr)

    diff_post_minus_null = auc_post_arr - auc_null_arr
    diff_post_minus_raw = auc_post_arr - auc_raw_arr

    # -------------------------
    # Print summary
    # -------------------------
    print("\n" + "=" * 72)
    print(f"TEMPORAL BIAS SUMMARY: {args.dataset}")
    print("=" * 72)

    print("\nOverall AUC")
    print(f"Raw   - Micro: {micro_raw:.4f} | Macro(skip): {macro_raw:.4f}")
    print(f"Post  - Micro: {micro_post:.4f} | Macro(skip): {macro_post:.4f}")
    print(f"Null  - Micro: {micro_null:.4f} | Macro(skip): {macro_null:.4f}")

    print("\nGT anomaly location")
    seg_mids = np.array([r['midpoint'] for r in segment_records], dtype=np.float64)
    seg_durs = np.array([r['duration'] for r in segment_records], dtype=np.float64)
    print(f"Anomaly segments analysed: {len(segment_records)}")
    print(f"Midpoint mean: {seg_mids.mean():.3f} | std: {seg_mids.std():.3f}")
    print(f"Duration mean: {seg_durs.mean():.3f} | std: {seg_durs.std():.3f}")

    print("\nSpearman correlation: anomaly midpoint vs per-video AUC")
    print(f"Raw  : rho = {sp_raw_r:.4f}, p = {sp_raw_p:.4g}")
    print(f"Post : rho = {sp_post_r:.4f}, p = {sp_post_p:.4g}")
    print(f"Null : rho = {sp_null_r:.4f}, p = {sp_null_p:.4g}")

    print("\nMiddle vs edge anomaly comparison")
    print(
        f"Raw  : n_mid={raw_me['n_middle']}, n_edge={raw_me['n_edge']}, "
        f"mean_mid={raw_me['mean_middle']:.4f}, mean_edge={raw_me['mean_edge']:.4f}, "
        f"gap={raw_me['mean_gap_middle_minus_edge']:.4f}"
    )
    print(
        f"Post : n_mid={post_me['n_middle']}, n_edge={post_me['n_edge']}, "
        f"mean_mid={post_me['mean_middle']:.4f}, mean_edge={post_me['mean_edge']:.4f}, "
        f"gap={post_me['mean_gap_middle_minus_edge']:.4f}"
    )
    print(
        f"Null : n_mid={null_me['n_middle']}, n_edge={null_me['n_edge']}, "
        f"mean_mid={null_me['mean_middle']:.4f}, mean_edge={null_me['mean_edge']:.4f}, "
        f"gap={null_me['mean_gap_middle_minus_edge']:.4f}"
    )

    if len(auc_post_arr) >= 2:
        pearson_post_null = float(np.corrcoef(auc_null_arr, auc_post_arr)[0, 1])
        print("\nPost vs null per-video AUC relation")
        print(f"Pearson correlation: {pearson_post_null:.4f}")

    print("\nPost-processed vs null")
    print(f"Videos analysed: {len(auc_post_arr)}")
    print(f"Post > Null: {(diff_post_minus_null > 0).sum()} / {len(diff_post_minus_null)}")
    print(f"Null > Post: {(diff_post_minus_null < 0).sum()} / {len(diff_post_minus_null)}")
    print(f"Mean(Post - Null): {diff_post_minus_null.mean():.4f}")
    print(f"Median(Post - Null): {np.median(diff_post_minus_null):.4f}")
    print(f"Post beats Null by >0.05: {(diff_post_minus_null > 0.05).sum()} / {len(diff_post_minus_null)}")

    print("\nPost-processed vs raw")
    print(f"Post > Raw: {(diff_post_minus_raw > 0).sum()} / {len(diff_post_minus_raw)}")
    print(f"Raw > Post: {(diff_post_minus_raw < 0).sum()} / {len(diff_post_minus_raw)}")
    print(f"Mean(Post - Raw): {diff_post_minus_raw.mean():.4f}")
    print(f"Median(Post - Raw): {np.median(diff_post_minus_raw):.4f}")

    print("\nSaved plots to:")
    print(f"  {os.path.join(args.outdir, f'{args.dataset}_gt_heatmaps.png')}")
    print(f"  {os.path.join(args.outdir, f'{args.dataset}_gt_midpoint_histogram.png')}")
    print(f"  {os.path.join(args.outdir, f'{args.dataset}_auc_location_maps.png')}")
    print(f"  {os.path.join(args.outdir, f'{args.dataset}_midpoint_vs_auc_all.png')}")
    print(f"  {os.path.join(args.outdir, f'{args.dataset}_post_vs_null_scatter.png')}")
    print(f"  {os.path.join(args.outdir, f'{args.dataset}_raw_post_null_bars.png')}")


if __name__ == "__main__":
    main()