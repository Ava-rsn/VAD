#!/usr/bin/env python3
"""
Temporal bias analysis for ShanghaiTech post-processing.
Tests whether the heavy Gaussian smoothing (range=1000, mu=350) 
is simply imposing a bell-curve prior that rewards mid-video anomalies.

Produces 3 figures:
  1. Heatmap of ground-truth anomaly temporal locations (start vs duration)
  2. Scatter of normalised anomaly midpoint vs per-video AUC
  3. Null-model comparison: pure bell curve (no model) vs actual model

Usage:
  python temporal_bias_analysis.py \
    --labels  <path_to_labels.npy> \
    --videos  <path_to_videos.npy> \
    --frames  <path_to_frames.npy> \
    --scores  <path_to_teacher_raw.npy> \
    --range_val 1000 --mu_val 350 --normalize \
    --dataset shanghai \
    --outdir  temporal_bias_plots

python temporal_bias_analysis.py \
  --labels  ucf_avenue_files/ucf_crime_labels.npy \
  --videos  ucf_avenue_files/ucf_crime_videos.npy \
  --frames  ucf_avenue_files/ucf_crime_frames.npy \
  --scores  ucf_avenue_files/ucf_crime_teacher_raw.npy \
  --range_val 1 --mu_val 1 \
  --dataset ucf_crime \
  --outdir  ucf_avenue_temporal_bias_plots
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_curve, auc
from collections import defaultdict


def gaussian_filter(support, sigma):
    mu = support[len(support) // 2 - 1]
    # mu = np.mean(support)
    filter = 1.0 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((support - mu) / sigma) ** 2)
    return filter


def filt(input, dim=9, range=302, mu=21):
    filter_3d = np.ones((dim, dim, dim)) / (dim ** 3)
    filter_2d = gaussian_filter(np.arange(1, range), mu)

    frame_scores = input  # convolve(input, filter_3d)
    # frame_scores = frame_scores.max((1, 2))

    padding_size = len(filter_2d) // 2
    in_ = np.concatenate((np.zeros(padding_size), frame_scores, np.zeros(padding_size)))
    frame_scores = np.correlate(in_, filter_2d, 'valid')
    return frame_scores



def preprocess_scores(scores_flat, videos_flat, range_val, mu_val, normalize):
    """Filter and optionally normalise scores per-video."""
    out = np.zeros_like(scores_flat, dtype=np.float32)
    for vid in np.unique(videos_flat):
        mask = videos_flat == vid
        pred = scores_flat[mask].copy().astype(np.float64)
        pred = filt(pred, range=range_val, mu=mu_val)
        pred = np.nan_to_num(pred, nan=0.0)
        if normalize:
            mn, mx = pred.min(), pred.max()
            if mx - mn > 1e-8:
                pred = (pred - mn) / (mx - mn)
            else:
                pred = np.zeros_like(pred)
        out[mask] = pred.astype(np.float32)
    return out


def compute_per_video_auc(scores, labels, videos):
    """Compute per-video AUC (skip variant)."""
    auc_dict = {}
    for vid in np.unique(videos):
        mask = videos == vid
        lbl = labels[mask]
        pred = scores[mask]
        if len(np.unique(lbl)) < 2:
            continue
        fpr, tpr, _ = roc_curve(lbl, pred)
        auc_dict[vid] = auc(fpr, tpr)
    return auc_dict


def get_anomaly_segments(labels_vid):
    """Extract contiguous anomaly segments from a binary label array."""
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


def generate_bell_curve(n_frames, range_val=1000, mu_val=350):
    """Generate a pure bell curve score (no model), mimicking the Gaussian filter."""
    # Create a flat score of 1.0 for all frames, then filter + normalise
    flat = np.ones(n_frames, dtype=np.float64)
    filtered = filt(flat, range=range_val, mu=mu_val)
    mn, mx = filtered.min(), filtered.max()
    if mx - mn > 1e-8:
        filtered = (filtered - mn) / (mx - mn)
    else:
        filtered = np.zeros_like(filtered)
    return filtered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--videos", required=True)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--range_val", type=int, default=1000)
    parser.add_argument("--mu_val", type=int, default=350)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--dataset", default="shanghai")
    parser.add_argument("--outdir", default="temporal_bias_plots")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    labels = np.load(args.labels, allow_pickle=True)
    videos = np.load(args.videos, allow_pickle=True)
    frames = np.load(args.frames, allow_pickle=True)
    scores_raw = np.load(args.scores, allow_pickle=True).astype(np.float32)

    print(f"Loaded {len(np.unique(videos))} videos, {len(labels)} frames")
    print(f"Post-processing: range={args.range_val}, mu={args.mu_val}, norm={args.normalize}")

    # - Preprocess actual model scores -
    scores_filt = preprocess_scores(scores_raw, videos, args.range_val, args.mu_val, args.normalize)
    auc_actual = compute_per_video_auc(scores_filt, labels, videos)

    # - Generate null model (pure bell curve) -
    scores_null = np.zeros_like(scores_raw, dtype=np.float32)
    for vid in np.unique(videos):
        mask = videos == vid
        n = mask.sum()
        scores_null[mask] = generate_bell_curve(n, args.range_val, args.mu_val).astype(np.float32)
    auc_null = compute_per_video_auc(scores_null, labels, videos)

    # - Extract anomaly temporal locations -
    norm_starts = []
    norm_durations = []
    norm_midpoints = []
    vid_aucs_actual = []
    vid_aucs_null = []
    vid_names = []

    for vid in np.unique(videos):
        mask = videos == vid
        lbl = labels[mask]
        n = mask.sum()
        if len(np.unique(lbl)) < 2:
            continue  # skip all-normal videos

        segments = get_anomaly_segments(lbl)
        if not segments:
            continue

        # Use the longest anomaly segment for the temporal location
        longest = max(segments, key=lambda s: s[1] - s[0])
        seg_start = longest[0] / n
        seg_end = longest[1] / n
        seg_dur = seg_end - seg_start
        seg_mid = (seg_start + seg_end) / 2

        norm_starts.append(seg_start)
        norm_durations.append(seg_dur)
        norm_midpoints.append(seg_mid)
        vid_aucs_actual.append(auc_actual.get(vid, np.nan))
        vid_aucs_null.append(auc_null.get(vid, np.nan))
        vid_names.append(vid)

    norm_starts = np.array(norm_starts)
    norm_durations = np.array(norm_durations)
    norm_midpoints = np.array(norm_midpoints)
    vid_aucs_actual = np.array(vid_aucs_actual)
    vid_aucs_null = np.array(vid_aucs_null)

    print(f"\nVideos with anomalies: {len(norm_starts)}")
    print(f"Anomaly midpoint - mean: {norm_midpoints.mean():.3f}, std: {norm_midpoints.std():.3f}")
    print(f"Anomaly duration - mean: {norm_durations.mean():.3f}, std: {norm_durations.std():.3f}")

    # - PLOT 1: Temporal location heatmap -
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: 2D histogram (like Otani et al.)
    ax = axes[0]
    h = ax.hist2d(norm_starts, norm_durations, bins=30, range=[[0, 1], [0, 1]],
                  cmap='YlOrRd', cmin=0.5)
    plt.colorbar(h[3], ax=ax, label='Number of videos')
    ax.set_xlabel('Normalised anomaly start', fontsize=12)
    ax.set_ylabel('Normalised anomaly duration', fontsize=12)
    ax.set_title(f'{args.dataset} - Ground-truth anomaly locations', fontsize=13)

    # Right: 2D histogram coloured by mean AUC
    ax = axes[1]
    from scipy.stats import binned_statistic_2d
    stat = binned_statistic_2d(norm_starts, norm_durations, vid_aucs_actual,
                                statistic='mean', bins=15, range=[[0, 1], [0, 1]])
    im = ax.imshow(stat.statistic.T, origin='lower', extent=[0, 1, 0, 1],
                   aspect='auto', cmap='RdYlGn', vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, label='Mean per-video AUC')
    ax.set_xlabel('Normalised anomaly start', fontsize=12)
    ax.set_ylabel('Normalised anomaly duration', fontsize=12)
    ax.set_title(f'{args.dataset} - Mean AUC by anomaly location', fontsize=13)

    plt.tight_layout()
    path1 = os.path.join(args.outdir, f"{args.dataset}_anomaly_location_heatmap.png")
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path1}")

    # - PLOT 2: Midpoint vs AUC scatter -
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    sc = ax.scatter(norm_midpoints, vid_aucs_actual, c=norm_durations, cmap='viridis',
                    alpha=0.6, s=30, edgecolors='none')
    plt.colorbar(sc, ax=ax, label='Anomaly duration (normalised)')
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Random chance')
    ax.axvline(0.5, color='grey', linestyle=':', alpha=0.5)
    ax.set_xlabel('Normalised anomaly midpoint', fontsize=12)
    ax.set_ylabel('Per-video AUC (actual model)', fontsize=12)
    ax.set_title(f'{args.dataset} - Actual model AUC vs anomaly midpoint', fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend()

    # Add trend line
    from numpy.polynomial.polynomial import polyfit
    valid = ~np.isnan(vid_aucs_actual)
    coeffs = polyfit(norm_midpoints[valid], vid_aucs_actual[valid], 2)
    x_fit = np.linspace(0, 1, 100)
    y_fit = coeffs[0] + coeffs[1] * x_fit + coeffs[2] * x_fit**2
    ax.plot(x_fit, y_fit, 'k-', linewidth=2, label='Quadratic fit')
    ax.legend()

    ax = axes[1]
    sc = ax.scatter(norm_midpoints, vid_aucs_null, c=norm_durations, cmap='viridis',
                    alpha=0.6, s=30, edgecolors='none')
    plt.colorbar(sc, ax=ax, label='Anomaly duration (normalised)')
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Random chance')
    ax.axvline(0.5, color='grey', linestyle=':', alpha=0.5)
    ax.set_xlabel('Normalised anomaly midpoint', fontsize=12)
    ax.set_ylabel('Per-video AUC (bell curve null model)', fontsize=12)
    ax.set_title(f'{args.dataset} - Null model (pure bell curve) AUC vs midpoint', fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend()

    valid_null = ~np.isnan(vid_aucs_null)
    coeffs_null = polyfit(norm_midpoints[valid_null], vid_aucs_null[valid_null], 2)
    y_fit_null = coeffs_null[0] + coeffs_null[1] * x_fit + coeffs_null[2] * x_fit**2
    ax.plot(x_fit, y_fit_null, 'k-', linewidth=2, label='Quadratic fit')
    ax.legend()

    plt.tight_layout()
    path2 = os.path.join(args.outdir, f"{args.dataset}_midpoint_vs_auc.png")
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path2}")

    # - PLOT 3: Null model comparison -
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter: actual vs null
    common_vids = sorted(set(auc_actual.keys()) & set(auc_null.keys()))
    actual_arr = np.array([auc_actual[v] for v in common_vids])
    null_arr = np.array([auc_null[v] for v in common_vids])

    ax = axes[0]
    ax.scatter(null_arr, actual_arr, alpha=0.5, s=25, edgecolors='none')
    ax.plot([0, 1], [0, 1], 'r--', label='y = x (model = null)')
    ax.set_xlabel('Null model AUC (pure bell curve)', fontsize=12)
    ax.set_ylabel('Actual model AUC', fontsize=12)
    ax.set_title(f'{args.dataset} - Actual vs null model per-video AUC', fontsize=13)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend()

    # Correlation
    corr = np.corrcoef(null_arr, actual_arr)[0, 1]
    ax.text(0.05, 0.92, f'r = {corr:.3f}', transform=ax.transAxes, fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Summary stats
    ax = axes[1]
    macro_actual = np.nanmean(actual_arr)
    macro_null = np.nanmean(null_arr)

    # Also compute micro AUC for null model
    fpr_null, tpr_null, _ = roc_curve(labels, scores_null)
    micro_null = auc(fpr_null, tpr_null)
    fpr_act, tpr_act, _ = roc_curve(labels, scores_filt)
    micro_actual = auc(fpr_act, tpr_act)

    bars = ax.bar(['Actual model\n(Micro)', 'Null model\n(Micro)',
                   'Actual model\n(Macro skip)', 'Null model\n(Macro skip)'],
                  [micro_actual, micro_null, macro_actual, macro_null],
                  color=['#2196F3', '#FF9800', '#2196F3', '#FF9800'],
                  edgecolor='black', linewidth=0.5)
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title(f'{args.dataset} - Model vs null model comparison', fontsize=13)
    ax.set_ylim(0, 1.0)

    for bar, val in zip(bars, [micro_actual, micro_null, macro_actual, macro_null]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    path3 = os.path.join(args.outdir, f"{args.dataset}_null_model_comparison.png")
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path3}")

    # - Print summary -
    print(f"\n{'='*60}")
    print(f"SUMMARY : {args.dataset}")
    print(f"{'='*60}")
    print(f"Actual model  - Micro AUC: {micro_actual:.4f}, Macro AUC (skip): {macro_actual:.4f}")
    print(f"Null model    - Micro AUC: {micro_null:.4f}, Macro AUC (skip): {macro_null:.4f}")
    print(f"Correlation (actual vs null per-video AUC): r = {corr:.4f}")
    print(f"Mean anomaly midpoint: {norm_midpoints.mean():.3f} (std: {norm_midpoints.std():.3f})")
    print(f"Videos where null > actual: {(null_arr > actual_arr).sum()} / {len(common_vids)}")
    print(f"Videos where actual > null: {(actual_arr > null_arr).sum()} / {len(common_vids)}")


if __name__ == "__main__":
    main()
