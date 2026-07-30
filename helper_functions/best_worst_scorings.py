import pandas as pd

# ---- CONFIG ----
CSV_PATH = "ucf_shanghai_files/ucf_crime_debug_scores.csv"

TARGET_CATEGORIES = [
    "Robbery",
    "Assault",
    "Fighting",
    "Abuse",
    "Burglary",
    "RoadAccidents",
    "Stealing"
]

OUTPUT_CSV = "ucf_selected_frames_summary.csv"


# ---- LOAD ----
df = pd.read_csv(CSV_PATH)

# Extract category from video name
# e.g. "Abuse002_x264" -> "Abuse"
df["category"] = df["video"].str.extract(r"([A-Za-z]+)")

# Keep only target categories
df = df[df["category"].isin(TARGET_CATEGORIES)]

results = []

# ---- PROCESS PER VIDEO ----
for video, group in df.groupby("video"):
    # Highest score
    max_row = group.loc[group["score_combined_raw"].idxmax()]
    
    # Lowest score
    min_row = group.loc[group["score_combined_raw"].idxmin()]
    
    results.append({
        "video": video,
        
        "max_frame": max_row["frame"],
        "max_score": max_row["score_combined_raw"],
        "max_label": max_row["label"],
        
        "min_frame": min_row["frame"],
        "min_score": min_row["score_combined_raw"],
        "min_label": min_row["label"],
    })

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Sort by max score descending (most interesting videos first)
results_df = results_df.sort_values(by="max_score", ascending=False)

# ---- PRINT ----
for _, row in results_df.iterrows():
    print(f"{row['video']}:")
    print(f"  HIGH -> frame {row['max_frame']} | score {row['max_score']:.4f} | label {row['max_label']}")
    print(f"  LOW  -> frame {row['min_frame']} | score {row['min_score']:.4f} | label {row['min_label']}")
    print()

# ---- SAVE ----
results_df.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved summary to {OUTPUT_CSV}")