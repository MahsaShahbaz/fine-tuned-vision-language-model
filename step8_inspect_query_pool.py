"""
Step 8 — Inspect the available query pool for evaluation

What this script does:
1) Loads the labeled metadata file
2) Prints the overall projection distribution
3) For each broad evaluation category, prints:
   - how many labeled images exist in total
   - how many are Frontal
   - how many are Lateral

Why we do this:
- Before building a fixed query set, we need to check whether each category
  has enough examples available.
- We also want to know whether the categories are reasonably represented
  across both projection types.
- This helps us decide whether a balanced evaluation query set is possible.

Why projection matters:
- Later in the project, we decided to evaluate retrieval within the same projection only.
- So it is not enough to know that a category exists in total.
- We also need to know whether there are enough Frontal and Lateral examples separately.
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
# 2) Define which label columns we want to inspect
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


# --------------------------------------------------
# 3) Print the overall projection distribution
# --------------------------------------------------
if "projection" not in df.columns:
    raise ValueError("Column 'projection' not found in the input CSV.")

print("Counts by projection:\n")
print(df["projection"].fillna("MISSING").value_counts(dropna=False).to_string())


# --------------------------------------------------
# 4) Inspect each label category one by one
# --------------------------------------------------
for col in label_cols:
    print("\n" + "=" * 80)
    print(col)

    subset = df[df[col] == 1]

    print("\nTotal:")
    print(len(subset))

    print("\nBy projection:")
    if len(subset) == 0:
        print("No rows in this category.")
    else:
        print(subset["projection"].fillna("MISSING").value_counts(dropna=False).to_string())