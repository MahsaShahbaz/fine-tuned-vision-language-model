"""
Step 9 — Check overlap between evaluation labels

What this script does:
1) Loads the labeled metadata file
2) Counts how many evaluation labels each image has
3) Prints how many rows have 0, 1, 2, ... labels
4) Extracts the rows that have more than one label
5) Prints example multi-label rows for inspection

Why we do this:
- Our broad evaluation categories are useful, but one image can sometimes match
  more than one category at the same time.
- For example, an image might be labeled as both cardiomegaly and opacity.
- This is not necessarily wrong clinically, but it can make retrieval evaluation
  harder to interpret.

Why this matters for the project:
- Later, when building a fixed query set, we want the evaluation to be as clean
  and understandable as possible.
- If a query image has multiple labels, it becomes less clear what should count
  as a relevant retrieval result.
- So in the next steps, we inspect this overlap and then decide whether to keep
  only single-label cases for the fixed evaluation query set.
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
# 2) Define which label columns we want to examine
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
# 3) Count how many labels each row has
# --------------------------------------------------
df["num_eval_labels"] = df[label_cols].sum(axis=1)


# --------------------------------------------------
# 4) Print how many rows have 0, 1, 2, ... labels
# --------------------------------------------------
print("How many rows have 0, 1, 2, ... labels:\n")
print(df["num_eval_labels"].value_counts().sort_index().to_string())


# --------------------------------------------------
# 5) Extract rows with more than one label
# --------------------------------------------------
multi = df[df["num_eval_labels"] > 1].copy()

print("\nNumber of multi-label rows:", len(multi))


# --------------------------------------------------
# 6) Choose columns to display for manual inspection
# --------------------------------------------------
show_cols = [
    "filename",
    "projection",
    "Problems",
    "MeSH",
    "eval_labels",
    "num_eval_labels",
]

existing_show_cols = [col for col in show_cols if col in df.columns]
if not existing_show_cols:
    raise ValueError("None of the display columns were found in the input CSV.")


# --------------------------------------------------
# 7) Print example multi-label rows
# --------------------------------------------------
print("\nFirst 20 multi-label rows:\n")
if len(multi) == 0:
    print("No multi-label rows found.")
else:
    print(multi[existing_show_cols].head(20).to_string(index=False))