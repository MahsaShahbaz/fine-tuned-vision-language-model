"""
Step 11 — Build both balanced and expanded fixed evaluation query sets

What this script does:
1) Loads the labeled metadata file
2) Loads the held-out test split file
3) Keeps only rows that belong to the held-out test split
4) Keeps only single-label rows
5) Builds an expanded test-only query set using all eligible single-label test rows
6) Checks how many examples are available for each category and projection
7) Builds a balanced test-only query set if possible
8) Saves both query sets to disk
9) Prints clear summaries for inspection

Why we do this:
- We want the final baseline vs fine-tuned comparison to use unseen test data only.
- We also want to solve a limitation from the earlier version:
  the balanced test query set became very small because it was limited by the rarest group.
- So in this updated version, we keep the balanced set for strict comparison,
  but we also create a larger expanded set for broader evaluation.

What is the difference between the two outputs:
1) Balanced query set:
   - same number of queries for every category and projection group
   - methodologically cleaner and easier to compare fairly
   - but can become very small if one rare group has very few examples

2) Expanded query set:
   - includes all eligible single-label test rows
   - gives us many more queries
   - but the category/projection distribution is no longer perfectly balanced

Important design decisions:
- We use only the held-out test split, not the full dataset.
- We use only single-label cases, so the relevance definition stays simpler.
- We save both query sets so later evaluation scripts can choose:
    - balanced set for stricter fairness
    - expanded set for more query coverage

Important note:
- This script does not yet change the evaluation scripts themselves.
- It only prepares the query-set files.
- Later, the baseline and fine-tuned evaluation scripts can be pointed to whichever
  query-set file we want to use.
"""

from pathlib import Path
import pandas as pd


# --------------------------------------------------
# 1) Define input and output paths
# --------------------------------------------------
INPUT_LABEL_CSV = Path("data/processed/indiana_metadata_eval_labels.csv")
TEST_SPLIT_CSV = Path("data/processed/finetuning/test.csv")

OUTPUT_BALANCED_CSV = Path("results/fixed_eval_query_set_balanced_test.csv")
OUTPUT_EXPANDED_CSV = Path("results/fixed_eval_query_set_all_single_label_test.csv")


# --------------------------------------------------
# 2) Check files and load inputs
# --------------------------------------------------
for p in [INPUT_LABEL_CSV, TEST_SPLIT_CSV]:
    if not p.exists():
        raise FileNotFoundError(f"Required file not found: {p.resolve()}")

label_df = pd.read_csv(INPUT_LABEL_CSV)
test_df = pd.read_csv(TEST_SPLIT_CSV)

print("Rows in full labeled metadata:", len(label_df))
print("Rows in held-out test split:", len(test_df))


# --------------------------------------------------
# 3) Define label columns and readable names
# --------------------------------------------------
label_cols = [
    "label_normal",
    "label_cardiomegaly",
    "label_pleural_effusion",
    "label_opacity",
    "label_pneumonia",
]

label_name_map = {
    "label_normal": "normal",
    "label_cardiomegaly": "cardiomegaly",
    "label_pleural_effusion": "pleural_effusion",
    "label_opacity": "opacity",
    "label_pneumonia": "pneumonia",
}

required_label_cols = ["filename", "projection"] + label_cols
missing_in_label_df = [c for c in required_label_cols if c not in label_df.columns]
if missing_in_label_df:
    raise ValueError(f"Missing required columns in labeled metadata: {missing_in_label_df}")

if "filename" not in test_df.columns:
    raise ValueError("Column 'filename' not found in held-out test split file.")


# --------------------------------------------------
# 4) Restrict labeled metadata to test split only
# --------------------------------------------------
test_filenames = set(test_df["filename"].astype(str))
df = label_df[label_df["filename"].astype(str).isin(test_filenames)].copy()

print("Rows after restricting labeled metadata to test split:", len(df))


# --------------------------------------------------
# 5) Keep only single-label rows
# --------------------------------------------------
df["num_eval_labels"] = df[label_cols].sum(axis=1)
single = df[df["num_eval_labels"] == 1].copy()

print("Rows in test split with exactly one eval label:", len(single))


# --------------------------------------------------
# 6) Add readable query_category
# --------------------------------------------------
single["query_category"] = ""

for label_col, category_name in label_name_map.items():
    single.loc[single[label_col] == 1, "query_category"] = category_name

if "image_path" in single.columns:
    single["image_path"] = single["image_path"].astype(str).str.replace("\\", "/", regex=False)

unassigned = (single["query_category"] == "").sum()
if unassigned > 0:
    raise ValueError(f"{unassigned} single-label rows did not receive a query_category.")


# --------------------------------------------------
# 7) Build and save the expanded query set
# --------------------------------------------------
expanded_query_df = single.copy()

preferred_cols = [
    "filename",
    "image_path",
    "projection",
    "query_category",
    "eval_labels",
    "Problems",
    "MeSH",
]

existing_preferred = [c for c in preferred_cols if c in expanded_query_df.columns]
remaining_cols = [c for c in expanded_query_df.columns if c not in existing_preferred]
expanded_query_df = expanded_query_df[existing_preferred + remaining_cols].copy()

expanded_query_df = expanded_query_df.sort_values(
    by=["query_category", "projection", "filename"]
).reset_index(drop=True)

OUTPUT_EXPANDED_CSV.parent.mkdir(parents=True, exist_ok=True)
expanded_query_df.to_csv(OUTPUT_EXPANDED_CSV, index=False)

print("\nSaved expanded query set to:", OUTPUT_EXPANDED_CSV)
print("Number of expanded queries:", len(expanded_query_df))

print("\nExpanded query counts by category and projection:")
print(
    expanded_query_df.groupby(["query_category", "projection"])
    .size()
    .to_string()
)


# --------------------------------------------------
# 8) Check available group sizes for balanced sampling
# --------------------------------------------------
group_sizes = []

for label_col in label_cols:
    for projection in ["Frontal", "Lateral"]:
        subset = single[
            (single[label_col] == 1) &
            (single["projection"] == projection)
        ].copy()

        group_name = f"{label_name_map[label_col]} | {projection}"
        group_count = len(subset)

        print(f"{group_name} | available single-label test rows: {group_count}")

        group_sizes.append({
            "label_col": label_col,
            "query_category": label_name_map[label_col],
            "projection": projection,
            "count": group_count
        })

group_sizes_df = pd.DataFrame(group_sizes)
min_count = group_sizes_df["count"].min()

print("\nSmallest available group size for balanced sampling:", min_count)


# --------------------------------------------------
# 9) Build balanced query set if possible
# --------------------------------------------------
if min_count == 0:
    print(
        "\nBalanced query set was NOT created because at least one "
        "category/projection group has zero available single-label test rows."
    )
else:
    samples = []

    for _, row in group_sizes_df.iterrows():
        label_col = row["label_col"]
        query_category = row["query_category"]
        projection = row["projection"]

        subset = single[
            (single[label_col] == 1) &
            (single["projection"] == projection)
        ].copy()

        sampled = subset.sample(n=min_count, random_state=42).copy()
        sampled["query_category"] = query_category
        samples.append(sampled)

    balanced_query_df = pd.concat(samples, axis=0).reset_index(drop=True)

    existing_preferred = [c for c in preferred_cols if c in balanced_query_df.columns]
    remaining_cols = [c for c in balanced_query_df.columns if c not in existing_preferred]
    balanced_query_df = balanced_query_df[existing_preferred + remaining_cols].copy()

    balanced_query_df = balanced_query_df.sort_values(
        by=["query_category", "projection", "filename"]
    ).reset_index(drop=True)

    OUTPUT_BALANCED_CSV.parent.mkdir(parents=True, exist_ok=True)
    balanced_query_df.to_csv(OUTPUT_BALANCED_CSV, index=False)

    print("\nSaved balanced query set to:", OUTPUT_BALANCED_CSV)
    print("Number of balanced queries:", len(balanced_query_df))

    print("\nBalanced query counts by category and projection:")
    print(
        balanced_query_df.groupby(["query_category", "projection"])
        .size()
        .to_string()
    )

    print("\nFirst 10 rows of balanced query set:")
    print(
        balanced_query_df[
            ["filename", "projection", "query_category", "eval_labels"]
        ].head(10).to_string(index=False)
    )


# --------------------------------------------------
# 10) Print a final preview of the expanded query set
# --------------------------------------------------
print("\nFirst 10 rows of expanded query set:")
print(
    expanded_query_df[
        ["filename", "projection", "query_category", "eval_labels"]
    ].head(10).to_string(index=False)
)