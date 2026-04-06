import os
import argparse
import numpy as np
from sklearn.metrics import roc_curve, auc


def load_array(path):
    return np.load(path, allow_pickle=True)


def compute_micro_auc(labels, scores):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float32)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    fpr, tpr, _ = roc_curve(labels, scores)
    return auc(fpr, tpr)


def compute_macro_auc_skip(labels, scores, videos):
    """Macro AUC: skip videos where only one class is present."""
    aucs = []
    for vid in np.unique(videos):
        vmask = (videos == vid)
        lbl   = labels[vmask].astype(int)
        pred  = np.nan_to_num(scores[vmask].astype(np.float32), nan=0.0)
        if np.unique(lbl).size < 2:
            continue
        fpr, tpr, _ = roc_curve(lbl, pred)
        aucs.append(auc(fpr, tpr))
    return float(np.nanmean(aucs)) if aucs else float('nan'), len(aucs)


def compute_macro_auc_anchored(labels, scores, videos):
    """Macro AUC: Ristea et al. style — anchor every video with [0]+lbl+[1]
    and [0]+pred+[1] so all-normal and all-anomalous videos are included."""
    aucs = []
    for vid in np.unique(videos):
        vmask = (videos == vid)
        lbl   = labels[vmask].astype(int)
        pred  = np.nan_to_num(scores[vmask].astype(np.float32), nan=0.0)
        lbl_a  = np.array([0] + list(lbl)  + [1])
        pred_a = np.array([0] + list(pred) + [1])
        fpr, tpr, _ = roc_curve(lbl_a, pred_a)
        aucs.append(auc(fpr, tpr))
    return float(np.nanmean(aucs)) if aucs else float('nan'), len(aucs)


def main():
    parser = argparse.ArgumentParser(description="Compute raw AUC with no filtering.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--dataset_prefix", type=str, required=True)
    args = parser.parse_args()

    labels = load_array(os.path.join(args.data_dir, f"{args.dataset_prefix}_labels.npy"))
    videos = load_array(os.path.join(args.data_dir, f"{args.dataset_prefix}_videos.npy"))

    score_files = {
        "teacher_raw" : os.path.join(args.data_dir, f"{args.dataset_prefix}_teacher_raw.npy"),
        "st_raw"      : os.path.join(args.data_dir, f"{args.dataset_prefix}_st_raw.npy"),
        "combined_raw": os.path.join(args.data_dir, f"{args.dataset_prefix}_combined_raw.npy"),
    }

    print(f"labels shape : {labels.shape}")
    print(f"videos shape : {videos.shape}")
    print(f"Total videos : {len(np.unique(videos))}")
    print()

    for name, path in score_files.items():
        if not os.path.exists(path):
            print(f"{name}: missing ({path})")
            continue

        scores = load_array(path).astype(np.float32)
        print(f"{name} shape: {scores.shape}")

        if len(scores) != len(labels):
            min_len = min(len(scores), len(labels))
            print(f"  [warn] length mismatch: trimming to {min_len}")
            scores      = scores[:min_len]
            labels_eval = labels[:min_len]
            videos_eval = videos[:min_len]
        else:
            labels_eval = labels
            videos_eval = videos

        micro = compute_micro_auc(labels_eval, scores)
        macro_skip, n_skip         = compute_macro_auc_skip(labels_eval, scores, videos_eval)
        macro_anchored, n_anchored = compute_macro_auc_anchored(labels_eval, scores, videos_eval)

        print(f"  Micro AUC                        : {micro:.6f}")
        print(f"  Macro AUC (skip one-class, n={n_skip:3d}) : {macro_skip:.6f}")
        print(f"  Macro AUC (anchored,  n={n_anchored:3d})  : {macro_anchored:.6f}")
        print()


if __name__ == "__main__":
    main()