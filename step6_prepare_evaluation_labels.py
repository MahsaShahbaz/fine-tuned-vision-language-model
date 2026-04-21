"""
Step 6 — Inspect metadata fields before creating evaluation labels

What this script does:
1) Loads the cleaned metadata file
2) Prints the available column names
3) Prints the projection distribution
4) Prints the most common values in Problems and MeSH
5) Prints a few example findings and impression texts

Why we do this:
- Before creating evaluation labels, we first want to understand what information
  already exists in the metadata.
- We do not want to design label rules blindly.
- This step helps us decide which fields are useful for broad category labeling.

Why this matters:
- Later, we evaluate retrieval using broad clinical categories such as:
    normal, cardiomegaly, pleural_effusion, opacity, and pneumonia
- To create those labels, we need to know whether the existing metadata fields
  already contain reliable category information.
- This script is an inspection step, not a labeling step yet.
"""

from pathlib import Path
import pandas as pd


# --------------------------------------------------
# 1) Define the input path and load the cleaned metadata
# --------------------------------------------------
INPUT_CSV = Path("data/processed/indiana_metadata_clean.csv")

if not INPUT_CSV.exists():
    raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV.resolve()}")

df = pd.read_csv(INPUT_CSV)


# --------------------------------------------------
# 2) Print the list of available columns
# --------------------------------------------------
print("Columns:")
print(df.columns.tolist())


# --------------------------------------------------
# 3) Print the projection distribution
# --------------------------------------------------
if "projection" in df.columns:
    print("\nProjection counts:")
    print(df["projection"].fillna("MISSING").value_counts(dropna=False).to_string())
else:
    print("\nColumn 'projection' not found.")


# --------------------------------------------------
# 4) Inspect the structured Problems column
# --------------------------------------------------
if "Problems" in df.columns:
    print("\nTop 30 Problems values:")
    print(df["Problems"].fillna("MISSING").value_counts().head(30).to_string())
else:
    print("\nColumn 'Problems' not found.")


# --------------------------------------------------
# 5) Inspect the structured MeSH column
# --------------------------------------------------
if "MeSH" in df.columns:
    print("\nTop 30 MeSH values:")
    print(df["MeSH"].fillna("MISSING").value_counts().head(30).to_string())
else:
    print("\nColumn 'MeSH' not found.")


# --------------------------------------------------
# 6) Print example findings text
# --------------------------------------------------
if "findings" in df.columns:
    print("\nSample findings:")
    print(df["findings"].fillna("MISSING").head(10).to_string(index=False))
else:
    print("\nColumn 'findings' not found.")


# --------------------------------------------------
# 7) Print example impression text
# --------------------------------------------------
if "impression" in df.columns:
    print("\nSample impression:")
    print(df["impression"].fillna("MISSING").head(10).to_string(index=False))
else:
    print("\nColumn 'impression' not found.")