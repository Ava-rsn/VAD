import pandas as pd

# ---- CONFIG ----
CSV_PATH = "ucf_shanghai_files/ucf_crime_debug_scores.csv"

TARGET_CATEGORIES = ["Assault","Abuse"]

MAX_SCORE_THRESHOLD = 0.3
MIN_SCORE_THRESHOLD = 0.3

OUTPUT_CSV = "arson_selected_frames_filtered.csv"


# ---- LOAD ----
df = pd.read_csv(CSV_PATH)

# Extract category from video name
df["category"] = df["video"].str.extract(r"([A-Za-z]+)")

# Keep only target categories
df = df[df["category"].isin(TARGET_CATEGORIES)]

results = []

# ---- PROCESS PER VIDEO ----
for video, group in df.groupby("video"):
    # GT anomaly frames only for max
    anomaly_group = group[group["label"] == 1]

    # GT normal frames only for min
    normal_group = group[group["label"] == 0]

    # Skip videos that don't contain both
    if len(anomaly_group) == 0 or len(normal_group) == 0:
        continue

    max_row = anomaly_group.loc[anomaly_group["score_combined_raw"].idxmax()]
    min_row = normal_group.loc[normal_group["score_combined_raw"].idxmin()]

    max_score = max_row["score_combined_raw"]
    min_score = min_row["score_combined_raw"]

    # Apply score thresholds
    if max_score <= MAX_SCORE_THRESHOLD or min_score >= MIN_SCORE_THRESHOLD:
        continue

    results.append({
        "video": video,

        "max_frame": max_row["frame"],
        "max_frame_num": str(max_row["frame"]).split("/")[-1],
        "max_score": max_score,
        "max_label": max_row["label"],

        "min_frame": min_row["frame"],
        "min_frame_num": str(min_row["frame"]).split("/")[-1],
        "min_score": min_score,
        "min_label": min_row["label"],
    })

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Sort by anomaly score descending
results_df = results_df.sort_values(by="max_score", ascending=False)

# ---- PRINT ----
for _, row in results_df.iterrows():
    print(f"{row['video']}:")
    print(f"  HIGH (GT anomaly) -> frame {row['max_frame_num']} | score {row['max_score']:.4f}")
    print(f"  LOW  (GT normal)  -> frame {row['min_frame_num']} | score {row['min_score']:.4f}")
    print()

# ---- SAVE ----
results_df.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved filtered summary to {OUTPUT_CSV}")
print(f"Number of selected videos: {len(results_df)}")