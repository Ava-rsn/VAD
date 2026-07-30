import os
import argparse
import numpy as np
import pandas as pd
from sklearn import metrics

from util.abnormal_utils import filt


def load_array(path):
    return np.load(path, allow_pickle=True)


def safe_minmax(x):
    x = np.asarray(x, dtype=np.float32)
    mn = np.min(x)
    mx = np.max(x)
    denom = mx - mn
    if denom > 1e-8:
        return (x - mn) / denom
    return np.zeros_like(x, dtype=np.float32)


def make_fast_mu_candidates(range_val):
    raw = [
        5, 11, 21,
        max(1, int(round(range_val / 10))),
        max(1, int(round(range_val / 8))),
        max(1, int(round(range_val / 6))),
        max(1, int(round(range_val / 4))),
        max(1, int(round(range_val / 3))),
        max(1, int(round(range_val / 2))),
    ]

    out = []
    seen = set()
    for v in raw:
        if v not in seen and v <= range_val:
            out.append(v)
            seen.add(v)
    return out

def evaluate_model_micro_only(predictions, labels, videos, range=302, mu=21, normalize_scores=False):
    predictions = np.asarray(predictions, dtype=np.float32)
    labels = np.asarray(labels)
    videos = np.asarray(videos)

    filtered_preds = []
    filtered_labels = []

    for vid in np.unique(videos):
        mask = (videos == vid)

        pred = predictions[mask]
        lbl = labels[mask]

        pred = filt(pred, range=range, mu=mu)
        pred = np.nan_to_num(pred, nan=0.0)

        # safety in case filt changes length for some params
        if len(pred) != len(lbl):
            print(
                f"[warn] length mismatch for vid={vid}: "
                f"filt_pred_len={len(pred)}, label_len={len(lbl)}, "
                f"range={range}, mu={mu}",
                flush=True
            )
            min_len = min(len(pred), len(lbl))
            pred = pred[:min_len]
            lbl = lbl[:min_len]

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

    filtered_preds = np.concatenate(filtered_preds)
    filtered_labels = np.concatenate(filtered_labels)

    fpr, tpr, _ = metrics.roc_curve(filtered_labels, filtered_preds)
    micro_auc = metrics.auc(fpr, tpr)
    micro_auc = np.nan_to_num(micro_auc, nan=1.0)

    return float(micro_auc)

def main():
    parser = argparse.ArgumentParser(description="Fast structured post-processing search for saved VAD scores.")
    parser.add_argument("--data_dir", type=str, default=".", help="Directory containing saved .npy files")
    parser.add_argument("--dataset_prefix", type=str, required=True, help="Example: shanghai, avenue, ucf_crime")
    parser.add_argument(
        "--score_types",
        type=str,
        default="teacher,combined",
        help="Comma-separated subset of: teacher,st,combined",
    )
    parser.add_argument(
        "--ranges",
        type=str,
        default="38,76,150,302,600,900",
        help="Comma-separated range values",
    )
    parser.add_argument(
        "--normalizes",
        type=str,
        default="false,true",
        help="Comma-separated booleans: false,true or just false",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="fast_postproc_results.csv",
        help="Where to save all results",
    )
    parser.add_argument(
        "--out_txt",
        type=str,
        default="fast_postproc_summary.txt",
        help="Where to save summary",
    )
    parser.add_argument(
        "--mus",
        type=str,
        default="",
        help="Optional comma-separated mu values. If empty, use structured mu candidates per range.",
    )
    parser.add_argument(
        "--val_list",
        type=str,
        default="",
        help="Optional path to txt file with video names to restrict evaluation to (e.g. validation split)",
    )


    args = parser.parse_args()
    mu_list = [int(x.strip()) for x in args.mus.split(",") if x.strip()] if args.mus.strip() else None
    score_types = [x.strip() for x in args.score_types.split(",") if x.strip()]
    range_list = [int(x.strip()) for x in args.ranges.split(",") if x.strip()]

    normalize_list = []
    for x in args.normalizes.split(","):
        x = x.strip().lower()
        if x in {"true", "1", "yes", "y"}:
            normalize_list.append(True)
        elif x in {"false", "0", "no", "n"}:
            normalize_list.append(False)
        else:
            raise ValueError(f"Bad normalize value: {x}")

    labels = load_array(os.path.join(args.data_dir, f"{args.dataset_prefix}_labels.npy"))
    videos = load_array(os.path.join(args.data_dir, f"{args.dataset_prefix}_videos.npy"))

    score_map = {}

    if "teacher" in score_types:
        score_map["teacher"] = load_array(os.path.join(args.data_dir, f"{args.dataset_prefix}_teacher_raw.npy"))
    if "st" in score_types:
        score_map["st"] = load_array(os.path.join(args.data_dir, f"{args.dataset_prefix}_st_raw.npy"))
    if "combined" in score_types:
        score_map["combined"] = load_array(os.path.join(args.data_dir, f"{args.dataset_prefix}_combined_raw.npy"))

    if not score_map:
        raise ValueError("No valid score types selected.")

    if args.val_list.strip():
        with open(args.val_list) as f:
            val_set = set(line.strip() for line in f if line.strip())
        val_mask = np.isin(videos, list(val_set))
        labels = labels[val_mask]
        videos = videos[val_mask]
        for k in score_map:
            score_map[k] = score_map[k][val_mask]
        print(f"Filtered to {len(np.unique(videos))} validation videos ({val_mask.sum()} frames)")


    search_plan = []
    for score_name in score_map.keys():
        for range_val in range_list:
            if mu_list is None:
                mu_candidates = make_fast_mu_candidates(range_val)
            else:
                mu_candidates = [mu for mu in mu_list if mu <= range_val]
            for mu_val in mu_candidates:
                for normalize_scores in normalize_list:
                    search_plan.append((score_name, range_val, mu_val, normalize_scores))

    print(f"labels shape: {labels.shape}")
    print(f"videos shape: {videos.shape}")
    for k, v in score_map.items():
        print(f"{k} shape: {v.shape}")
    print(f"Total configs to evaluate: {len(search_plan)}")

    rows = []
    for idx, (score_name, range_val, mu_val, normalize_scores) in enumerate(search_plan, start=1):
        micro_auc= evaluate_model_micro_only(
            predictions=score_map[score_name],
            labels=labels,
            videos=videos,
            range=range_val,
            mu=mu_val,
            normalize_scores=normalize_scores,
        )

        row = {
            "score_type": score_name,
            "range": range_val,
            "mu": mu_val,
            "mu_ratio": mu_val / range_val,
            "normalize_scores": normalize_scores,
            "micro_auc": micro_auc,
        }
        rows.append(row)

        print(
            f"[{idx:03d}/{len(search_plan):03d}] "
            f"score={score_name:8s} "
            f"range={range_val:4d} "
            f"mu={mu_val:4d} "
            f"norm={str(normalize_scores):5s} "
            f"micro={micro_auc:.6f} ",
            flush=True,
        )

    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values(
        by=["micro_auc"],
        ascending=[False]
    ).reset_index(drop=True)

    results_df.to_csv(args.out_csv, index=False)

    best_overall = results_df.iloc[0]

    best_per_score = (
        results_df.sort_values(["score_type", "micro_auc"], ascending=[True, False])
        .groupby("score_type", as_index=False)
        .first()
    )

    with open(args.out_txt, "w") as f:
        f.write("FAST POST-PROCESSING SEARCH SUMMARY\n")
        f.write("==================================\n\n")

        f.write("Best overall:\n")
        f.write(best_overall.to_string())
        f.write("\n\n")

        f.write("Best per score type:\n")
        f.write(best_per_score.to_string(index=False))
        f.write("\n\n")

        f.write("Top 20 overall:\n")
        f.write(results_df.head(20).to_string(index=False))
        f.write("\n")

    print("\nDone.")
    print(f"Saved results CSV to: {args.out_csv}")
    print(f"Saved summary TXT to: {args.out_txt}")
    print("\nBest overall:")
    print(best_overall.to_string())


if __name__ == "__main__":
    main()