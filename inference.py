from collections.abc import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn import metrics
import os
import cv2

from util import misc
from util.abnormal_utils import filt


def filter_scores_per_video(scores, videos, range_val=302, mu_val=21, normalize_scores=False):
    """
    Apply filt() video-by-video so temporal smoothing does not leak across videos.
    Optionally min-max normalize within each video after filtering.
    """
    scores = np.asarray(scores, dtype=np.float32)
    videos = np.asarray(videos)

    filtered_scores = np.zeros_like(scores, dtype=np.float32)

    for vid in np.unique(videos):
        mask = (videos == vid)
        pred_vid = scores[mask]

        pred_vid = filt(pred_vid, range=range_val, mu=mu_val)
        pred_vid = np.nan_to_num(pred_vid, nan=0.0)

        if normalize_scores:
            min_v = np.min(pred_vid)
            max_v = np.max(pred_vid)
            denom = max_v - min_v
            if denom > 1e-8:
                pred_vid = (pred_vid - min_v) / denom
            else:
                pred_vid = np.zeros_like(pred_vid, dtype=np.float32)

        filtered_scores[mask] = pred_vid

    return filtered_scores


def evaluate_model(predictions, labels, videos, range=302, mu=21, normalize_scores=False):
    """
    Evaluate raw predictions by first filtering them per video inside this function.
    """
    predictions = np.asarray(predictions, dtype=np.float32)
    labels = np.asarray(labels)
    videos = np.asarray(videos)

    aucs = []
    filtered_preds = []
    filtered_labels = []

    for vid in np.unique(videos):
        mask = (videos == vid)

        pred = predictions[mask]
        lbl = labels[mask]

        pred = filt(pred, range=range, mu=mu)
        pred = np.nan_to_num(pred, nan=0.0)

        if normalize_scores:
            min_v = np.min(pred)
            max_v = np.max(pred)
            denom = max_v - min_v
            if denom > 1e-8:
                pred = (pred - min_v) / denom
            else:
                pred = np.zeros_like(pred, dtype=np.float32)

        filtered_preds.append(pred)
        filtered_labels.append(lbl)

        # Per-video AUC only makes sense if both classes are present
        if np.unique(lbl).size < 2:
            continue

        lbl_auc = np.array([0] + list(lbl) + [1])
        pred_auc = np.array([0] + list(pred) + [1])

        fpr, tpr, _ = metrics.roc_curve(lbl_auc, pred_auc)
        aucs.append(metrics.auc(fpr, tpr))

    macro_auc = np.nanmean(aucs) if len(aucs) > 0 else np.nan

    filtered_preds = np.concatenate(filtered_preds)
    filtered_labels = np.concatenate(filtered_labels)

    fpr, tpr, _ = metrics.roc_curve(filtered_labels, filtered_preds)
    micro_auc = metrics.auc(fpr, tpr)
    micro_auc = np.nan_to_num(micro_auc, nan=1.0)

    print(
        f"MicroAUC: {micro_auc}, MacroAUC: {macro_auc}, "
        f"range:{range}, mu:{mu}, normalize_scores:{normalize_scores}"
    )
    return micro_auc, macro_auc


# ================================================================================================

def tensor_to_rgb_image(x):
    """
    x: [C, H, W], float tensor
    Uses first 3 channels only.
    """
    x = x[:3].detach().cpu().float()
    x = x.permute(1, 2, 0).numpy()
    x = np.nan_to_num(x)

    # If already in [0,1], keep it. Otherwise min-max for display.
    mn, mx = x.min(), x.max()
    if mx > 1.0 or mn < 0.0:
        if mx > mn:
            x = (x - mn) / (mx - mn)
        else:
            x = np.zeros_like(x)

    x = (x * 255).clip(0, 255).astype(np.uint8)
    return x


def map_to_heatmap(x):
    """
    x: [H, W] numpy or tensor -> color heatmap
    """
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.nan_to_num(x.astype(np.float32))
    mn, mx = x.min(), x.max()
    if mx > mn:
        x = (x - mn) / (mx - mn)
    else:
        x = np.zeros_like(x)
    x_u8 = (x * 255).astype(np.uint8)
    return cv2.applyColorMap(x_u8, cv2.COLORMAP_JET)


def overlay_heatmap_on_rgb(rgb, heatmap, alpha=0.45):
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(rgb_bgr, 1 - alpha, heatmap, alpha, 0)

# ================================================================================================

def evaluate_precomputed_scores(predictions, labels, videos, score_name="score"):
    """
    Evaluate scores that have already been filtered / normalized outside.
    """
    predictions = np.asarray(predictions, dtype=np.float32)
    labels = np.asarray(labels)
    videos = np.asarray(videos)

    aucs = []
    all_preds = []
    all_labels = []

    for vid in np.unique(videos):
        mask = (videos == vid)

        pred = np.nan_to_num(predictions[mask], nan=0.0)
        lbl = labels[mask]

        all_preds.append(pred)
        all_labels.append(lbl)

        if np.unique(lbl).size < 2:
            continue

        lbl_auc = np.array([0] + list(lbl) + [1])
        pred_auc = np.array([0] + list(pred) + [1])

        fpr, tpr, _ = metrics.roc_curve(lbl_auc, pred_auc)
        aucs.append(metrics.auc(fpr, tpr))

    macro_auc = np.nanmean(aucs) if len(aucs) > 0 else np.nan

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    fpr, tpr, _ = metrics.roc_curve(all_labels, all_preds)
    micro_auc = metrics.auc(fpr, tpr)
    micro_auc = np.nan_to_num(micro_auc, nan=1.0)

    print(f"[precomputed:{score_name}] MicroAUC: {micro_auc}, MacroAUC: {macro_auc}")
    return micro_auc, macro_auc


def inference(model: torch.nn.Module, data_loader: Iterable,
              device: torch.device, log_writer=None, args=None):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Testing "

    if log_writer is not None:
        print(f"log_dir: {log_writer.log_dir}")

    predictions_teacher = []
    predictions_student_teacher = []
    labels = []
    videos = []
    frames = []

    for data_iter_step, (samples, grads, targets, label, vid, frame_name) in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        videos += list(vid)
        labels += list(label.detach().cpu().numpy())
        frames += list(frame_name)

        samples = samples.to(device)
        grads = grads.to(device)
        targets = targets.to(device)

        model.train_TS = True  # student-teacher reconstruction error

        # Keep your current scoring choice logic
        if args.dataset in ["avenue","ucf_crime"]:
            model.abnormal_score_func_TS = "L2"
        else:
            model.abnormal_score_func_TS = "L1"

        _, _, _, recon_error_st_tc = model(
            samples,
            targets=targets,
            grad_mask=grads,
            mask_ratio=args.mask_ratio
        )

        recon_error_st_tc[0] = recon_error_st_tc[0].detach().cpu().numpy()
        recon_error_st_tc[1] = recon_error_st_tc[1].detach().cpu().numpy()

        predictions_student_teacher += list(recon_error_st_tc[0])
        predictions_teacher += list(recon_error_st_tc[1])

    # Convert to numpy
    predictions_teacher = np.array(predictions_teacher, dtype=np.float32)
    predictions_student_teacher = np.array(predictions_student_teacher, dtype=np.float32)
    predictions_combined = predictions_teacher + predictions_student_teacher
    labels = np.array(labels)
    videos = np.array(videos)
    frames = np.array(frames)

    '''
    # Dataset-specific post-processing params
    if args.dataset == "avenue":
        filt_range = 38
        filt_mu = 4
        filt_norm = False
        eval_raw_choice = "teacher"
    elif args.dataset == "shanghai":
        filt_range = 900
        filt_mu = 282
        filt_norm = True
        eval_raw_choice = "combined"
    elif args.dataset == "ucf_crime":
        filt_range = 640
        filt_mu = 211
        filt_norm = True
        eval_raw_choice = "teacher"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    
    # Filtered versions
    predictions_teacher_filt = filter_scores_per_video(
        predictions_teacher, videos,
        range_val=filt_range, mu_val=filt_mu,
        normalize_scores=filt_norm
    )
    predictions_st_filt = filter_scores_per_video(
        predictions_student_teacher, videos,
        range_val=filt_range, mu_val=filt_mu,
        normalize_scores=filt_norm
    )
    predictions_combined_filt = filter_scores_per_video(
        predictions_combined, videos,
        range_val=filt_range, mu_val=filt_mu,
        normalize_scores=filt_norm
    )
    '''

    # Save raw arrays
    np.save(f"{args.dataset}_teacher_raw.npy", predictions_teacher)
    np.save(f"{args.dataset}_st_raw.npy", predictions_student_teacher)
    np.save(f"{args.dataset}_combined_raw.npy", predictions_combined)

    '''
    # Save filtered arrays
    np.save(f"{args.dataset}_teacher_filt.npy", predictions_teacher_filt)
    np.save(f"{args.dataset}_st_filt.npy", predictions_st_filt)
    np.save(f"{args.dataset}_combined_filt.npy", predictions_combined_filt)
    '''

    # Save metadata
    np.save(f"{args.dataset}_labels.npy", labels)
    np.save(f"{args.dataset}_videos.npy", videos)
    np.save(f"{args.dataset}_frames.npy", frames)

    # Save one debug CSV with everything
    debug_df = pd.DataFrame({
        "video": videos,
        "frame": frames,
        "label": labels,

        "score_teacher_raw": predictions_teacher,
        "score_st_raw": predictions_student_teacher,
        "score_combined_raw": predictions_combined,
    })
    debug_csv_name = f"{args.dataset}_debug_scores.csv"
    debug_df.to_csv(debug_csv_name, index=False)
    print(f"[inference] Saved {debug_csv_name}", flush=True)

    print("[inference] Saved predictions to disk", flush=True)
    '''
    
    # Original-style evaluation: raw input, filtering inside evaluate_model()
    print(f"[inference] Evaluating dataset={args.dataset}", flush=True)

    if eval_raw_choice == "teacher":
        print("[eval] Raw teacher scores with internal filtering")
        evaluate_model(
            predictions_teacher, labels, videos,
            range=filt_range, mu=filt_mu, normalize_scores=filt_norm
        )
    elif eval_raw_choice == "combined":
        print("[eval] Raw combined scores with internal filtering")
        evaluate_model(
            predictions_combined, labels, videos,
            range=filt_range, mu=filt_mu, normalize_scores=filt_norm
        )

    # Extra comparisons on already filtered scores
    print("[eval] Precomputed filtered teacher")
    evaluate_precomputed_scores(predictions_teacher_filt, labels, videos, score_name="teacher_filt")

    print("[eval] Precomputed filtered student-teacher")
    evaluate_precomputed_scores(predictions_st_filt, labels, videos, score_name="st_filt")

    print("[eval] Precomputed filtered combined")
    evaluate_precomputed_scores(predictions_combined_filt, labels, videos, score_name="combined_filt")
    
    '''