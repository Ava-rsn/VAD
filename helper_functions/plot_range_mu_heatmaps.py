import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot range-mu heatmaps from post-processing search CSV.")
    parser.add_argument("--csv", type=str, required=True, help="CSV produced by range_search.py")
    parser.add_argument("--out_dir", type=str, default="heatmaps", help="Output directory")
    parser.add_argument("--value_col", type=str, default="micro_auc", help="Value column to plot")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)

    required_cols = ["score_type", "range", "mu", "normalize_scores", args.value_col]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # make normalize readable
    df["normalize_label"] = df["normalize_scores"].map({False: "norm=False", True: "norm=True"})

    score_types = sorted(df["score_type"].unique())
    norm_labels = ["norm=False", "norm=True"]

    # ----- one heatmap per (score_type, normalize) -----
    for score_type in score_types:
        for norm_label in norm_labels:
            sub = df[(df["score_type"] == score_type) & (df["normalize_label"] == norm_label)].copy()
            if len(sub) == 0:
                continue

            pivot = sub.pivot_table(
                index="mu",
                columns="range",
                values=args.value_col,
                aggfunc="max"
            ).sort_index().sort_index(axis=1)

            plt.figure(figsize=(8, 6))
            im = plt.imshow(pivot.values, aspect="auto", origin="lower")
            plt.colorbar(im, label=args.value_col)

            plt.xticks(np.arange(len(pivot.columns)), pivot.columns)
            plt.yticks(np.arange(len(pivot.index)), pivot.index)

            plt.xlabel("range")
            plt.ylabel("mu")
            plt.title(f"{score_type} | {norm_label} | {args.value_col}")

            # write numbers inside cells
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        plt.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7)

            plt.tight_layout()
            out_path = os.path.join(args.out_dir, f"heatmap_{score_type}_{norm_label.replace('=', '_')}.png")
            plt.savefig(out_path, dpi=250)
            plt.close()

    # ----- combined figure: rows = score type, cols = normalize -----
    fig, axes = plt.subplots(
        nrows=len(score_types),
        ncols=2,
        figsize=(12, 4 * len(score_types)),
        squeeze=False
    )

    for r, score_type in enumerate(score_types):
        for c, norm_label in enumerate(norm_labels):
            ax = axes[r, c]
            sub = df[(df["score_type"] == score_type) & (df["normalize_label"] == norm_label)].copy()

            if len(sub) == 0:
                ax.axis("off")
                continue

            pivot = sub.pivot_table(
                index="mu",
                columns="range",
                values=args.value_col,
                aggfunc="max"
            ).sort_index().sort_index(axis=1)

            im = ax.imshow(pivot.values, aspect="auto", origin="lower")
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns)
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_xlabel("range")
            ax.set_ylabel("mu")
            ax.set_title(f"{score_type} | {norm_label}")

            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=6)

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95, label=args.value_col)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "all_heatmaps.png"), dpi=250)
    plt.close()

    print(f"Saved heatmaps to: {args.out_dir}")


if __name__ == "__main__":
    main()