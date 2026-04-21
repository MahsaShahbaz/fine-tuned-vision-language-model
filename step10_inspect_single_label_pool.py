"""
Step 10 — Inspect the pool of single-label cases

What this script does:
1) Loads the labeled metadata file
2) Counts how many evaluation labels each image has
3) Keeps only rows with exactly one label
4) Prints how many single-label rows exist in total
5) For each category, prints:
   - how many single-label rows belong to that category
   - how they are split across Frontal and Lateral projections

Why we do this:
- In the previous step, we saw that some images have more than one evaluation label.
- For the fixed query set, we want a cleaner and easier-to-interpret evaluation.
- So we focus on single-label cases only.

Why this matters:
- If a query image has exactly one broad category, it is easier to define what counts
  as a relevant retrieval result.
- This step helps us check whether there are still enough examples left in each category
  after removing the multi-label rows.
- We also need to know whether each category still has enough Frontal and Lateral cases,
  because later retrieval is evaluated within the same projection type.
"""

from pathlib import Path
import pandas as pd


# --------------------------------------------------
# 1) Define the input path and load the labeled metadata
# --------------------------------------------------
INPUT_CSV = Path("data/processed/indiana_metadata_eval_labels.csv")

if not INPUT_CSV.exists():
    raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV.resolve()}")

df = pd.read_csv(INPUT_CSV)
print("Loaded rows:", len(df))


# --------------------------------------------------
# 2) Define which label columns we are working with
# --------------------------------------------------
label_cols = [
    "label_normal",
    "label_cardiomegaly",
    "label_pleural_effusion",
    "label_opacity",
    "label_pneumonia",
]

missing_label_cols = [col for col in label_cols if col not in df.columns]
if missing_label_cols:
    raise ValueError(f"Missing label columns: {missing_label_cols}")

if "projection" not in df.columns:
    raise ValueError("Column 'projection' not found in the input CSV.")


# --------------------------------------------------
# 3) Count the number of evaluation labels per row
# --------------------------------------------------
df["num_eval_labels"] = df[label_cols].sum(axis=1)


# --------------------------------------------------
# 4) Keep only single-label rows
# --------------------------------------------------
single = df[df["num_eval_labels"] == 1].copy()

print("Total single-label rows:", len(single))


# --------------------------------------------------
# 5) Inspect the single-label pool category by category
# --------------------------------------------------
for col in label_cols:
    print("\n" + "=" * 80)
    print(col)

    subset = single[single[col] == 1]

    print("\nTotal single-label rows in this category:")
    print(len(subset))

    print("\nBy projection:")
    if len(subset) == 0:
        print("No rows in this category.")
    else:
        print(subset["projection"].fillna("MISSING").value_counts(dropna=False).to_string())