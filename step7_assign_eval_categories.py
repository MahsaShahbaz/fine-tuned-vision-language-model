"""
Step 7 — Assign broad evaluation categories to each image

What this script does:
1) Loads the cleaned metadata file
2) Looks at the structured metadata fields Problems and MeSH
3) Assigns broad evaluation labels to each image
4) Saves the new labeled metadata file

Why we do this:
- For retrieval evaluation, we need a simple definition of what counts as a relevant match.
- Instead of using the full free-text report, we define a small set of broad categories:
    normal, cardiomegaly, pleural_effusion, opacity, and pneumonia
- These broad labels are easier to evaluate and compare than full report text.

Why we chose Problems and MeSH:
- Earlier inspection showed that these structured metadata fields already contain
  useful clinical terms.
- They are cleaner and more consistent than trying to parse the full free-text findings
  and impression from scratch.
- So for this first evaluation setup, we decided to build the broad labels mainly from
  Problems and MeSH.

Important note:
- A case can receive more than one label if more than one target pattern is found.
- Later, when building the fixed query set, we may restrict the queries to single-label cases
  to make evaluation easier to interpret.
"""

from pathlib import Path
import pandas as pd
import re


# --------------------------------------------------
# 1) Define input and output paths
# --------------------------------------------------
# INPUT_CSV is the cleaned metadata file produced earlier.
# OUTPUT_CSV will be a new metadata file that includes added evaluation labels.
INPUT_CSV = Path("data/processed/indiana_metadata_clean.csv")
OUTPUT_CSV = Path("data/processed/indiana_metadata_eval_labels.csv")


# --------------------------------------------------
# 2) Check input file and load the cleaned metadata
# --------------------------------------------------
# We stop early if the expected cleaned metadata file is missing.
# This makes debugging easier and avoids more confusing errors later.
if not INPUT_CSV.exists():
    raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV.resolve()}")

df = pd.read_csv(INPUT_CSV)
print("Loaded rows:", len(df))


# --------------------------------------------------
# 3) Decide which metadata fields to use for label assignment
# --------------------------------------------------
# We mainly use Problems and MeSH because they are structured metadata fields.
#
# Why this is useful:
# - they often already contain broad clinical terms
# - they are more consistent than raw free-text report sections
# - they are easier to match with simple rules
#
# We do not force both columns to exist.
# Instead, we use whichever of these preferred columns are available.
preferred_text_cols = ["Problems", "MeSH"]
text_cols = [col for col in preferred_text_cols if col in df.columns]

if not text_cols:
    raise ValueError("Neither 'Problems' nor 'MeSH' was found in the input CSV.")

print("Using text columns for label assignment:", text_cols)


# --------------------------------------------------
# 4) Helper function: combine selected text fields safely
# --------------------------------------------------
# For each row, we build one lowercase text string from the selected columns.
#
# Why we do this:
# - some rows may have missing values
# - some fields may be empty
# - we want one single searchable text block per image row
#
# Example:
# Problems = "Cardiomegaly"
# MeSH = "Pleural Effusion"
#
# Result:
# "cardiomegaly | pleural effusion"
def safe_text(row):
    parts = []
    for col in text_cols:
        value = row.get(col, None)
        if pd.notna(value):
            parts.append(str(value))
    return " | ".join(parts).lower()


# --------------------------------------------------
# 5) Helper function: check whether a label should be assigned
# --------------------------------------------------
# This function applies simple rule-based label matching.
#
# Logic:
# - first look for a positive pattern
# - if no positive pattern is found -> return 0
# - if a positive pattern is found but a negative pattern is also found -> return 0
# - otherwise -> return 1
#
# Why negative patterns matter:
# Medical text often contains phrases like:
# - "no pleural effusion"
# - "without pneumonia"
#
# If we only searched for the disease word itself,
# we would assign many incorrect labels.
def has_positive(text, positive_patterns, negative_patterns=None):
    positive_found = any(re.search(p, text) for p in positive_patterns)

    if not positive_found:
        return 0

    if negative_patterns and any(re.search(p, text) for p in negative_patterns):
        return 0

    return 1


# --------------------------------------------------
# 6) Prepare a list to store the label results for each row
# --------------------------------------------------
# We collect the label results row by row in a list of dictionaries.
# Later we turn that list into a DataFrame and attach it to the metadata.
all_labels = []


# --------------------------------------------------
# 7) Loop through each image row and assign labels
# --------------------------------------------------
# For each image row, we:
# - combine the selected metadata text into one lowercase string
# - apply simple regex rules for each target category
# - save both binary labels and a combined readable label string
for _, row in df.iterrows():
    text = safe_text(row)

    # ------------------------------------------
    # 7a) Cardiomegaly label
    # ------------------------------------------
    # We assign this label if the text contains "cardiomegaly".
    label_cardiomegaly = has_positive(
        text,
        [r"\bcardiomegaly\b"]
    )

    # ------------------------------------------
    # 7b) Pleural effusion label
    # ------------------------------------------
    # We allow both singular and plural forms.
    # We also exclude explicit negative phrases such as:
    # - "no pleural effusion"
    # - "without pleural effusion"
    label_pleural_effusion = has_positive(
        text,
        [r"\bpleural effusion\b", r"\bpleural effusions\b"],
        [r"\bno pleural effusion\b", r"\bwithout pleural effusion\b"]
    )

    # ------------------------------------------
    # 7c) Opacity label
    # ------------------------------------------
    # Opacity may appear in singular or plural form.
    # We also exclude common negative phrases such as:
    # - "no opacity"
    # - "no focal opacity"
    # - "without opacity"
    #
    # We keep this broad on purpose, because for this evaluation setup
    # we want a simple category rather than a very fine-grained radiology label.
    label_opacity = has_positive(
        text,
        [r"\bopacity\b", r"\bopacities\b"],
        [
            r"\bno opacity\b",
            r"\bno opacities\b",
            r"\bno focal opacity\b",
            r"\bno focal opacities\b",
            r"\bwithout opacity\b",
            r"\bwithout opacities\b"
        ]
    )

    # ------------------------------------------
    # 7d) Pneumonia label
    # ------------------------------------------
    # We allow both "pneumonia" and "pneumonic".
    # We exclude explicit negative mentions.
    label_pneumonia = has_positive(
        text,
        [r"\bpneumonia\b", r"\bpneumonic\b"],
        [r"\bno pneumonia\b", r"\bwithout pneumonia\b"]
    )

    # ------------------------------------------
    # 7e) Normal label
    # ------------------------------------------
    # We allow a few broad normal-style phrases.
    #
    # We keep this rule simple because we are building broad evaluation labels,
    # not a full clinical annotation system.
    label_normal = has_positive(
        text,
        [
            r"\bnormal\b",
            r"\bnormal chest\b",
            r"\bno acute cardiopulmonary abnormality\b",
            r"\bno acute pulmonary findings\b"
        ]
    )

    # ------------------------------------------
    # 7f) Resolve normal-vs-abnormal conflict
    # ------------------------------------------
    # If one of our abnormal target categories is found,
    # we remove the normal label.
    #
    # Why we do this:
    # For evaluation, it becomes confusing if one image is both:
    # - normal
    # - and also clearly abnormal, such as opacity or pneumonia
    #
    # So we use a simple rule:
    # abnormal target labels override normal.
    if label_cardiomegaly or label_pleural_effusion or label_opacity or label_pneumonia:
        label_normal = 0

    # ------------------------------------------
    # 7g) Build a readable label list
    # ------------------------------------------
    # We store both:
    # - separate binary label columns
    # - one combined text column called eval_labels
    #
    # Example eval_labels:
    # - "opacity"
    # - "cardiomegaly;opacity"
    #
    # We allow multi-label rows here because clinically an image can contain
    # more than one finding.
    # Later, for cleaner evaluation, we may restrict the query set
    # to single-label cases only.
    labels = []

    if label_normal:
        labels.append("normal")
    if label_cardiomegaly:
        labels.append("cardiomegaly")
    if label_pleural_effusion:
        labels.append("pleural_effusion")
    if label_opacity:
        labels.append("opacity")
    if label_pneumonia:
        labels.append("pneumonia")

    all_labels.append({
        "label_normal": label_normal,
        "label_cardiomegaly": label_cardiomegaly,
        "label_pleural_effusion": label_pleural_effusion,
        "label_opacity": label_opacity,
        "label_pneumonia": label_pneumonia,
        "eval_labels": ";".join(labels),
        "has_eval_label": int(len(labels) > 0)
    })


# --------------------------------------------------
# 8) Convert label results into a DataFrame and attach to main metadata
# --------------------------------------------------
# We turn the collected row-by-row label results into a DataFrame
# and attach it to the original metadata table.
label_df = pd.DataFrame(all_labels)
df = pd.concat([df, label_df], axis=1)


# --------------------------------------------------
# 9) Save the labeled metadata file
# --------------------------------------------------
# This output file will be used in the next evaluation-preparation steps.
df.to_csv(OUTPUT_CSV, index=False)


# --------------------------------------------------
# 10) Print a summary of the label counts
# --------------------------------------------------
# We print how many rows received each label.
# This is a simple sanity check to see whether the label assignment
# looks reasonable before moving to the next step.
print("Saved file:", OUTPUT_CSV)

print("\nLabel counts:")
for col in [
    "label_normal",
    "label_cardiomegaly",
    "label_pleural_effusion",
    "label_opacity",
    "label_pneumonia",
    "has_eval_label"
]:
    print(f"{col}: {df[col].sum()}")


# --------------------------------------------------
# 11) Print example labeled rows for inspection
# --------------------------------------------------
# We print the first few labeled rows so we can manually inspect:
# - the metadata text
# - the assigned labels
# - whether the output looks sensible
print("\nFirst 10 labeled rows:")
show_cols = [
    "filename", "projection", "Problems", "MeSH",
    "label_normal", "label_cardiomegaly",
    "label_pleural_effusion", "label_opacity",
    "label_pneumonia", "eval_labels"
]
existing_show_cols = [c for c in show_cols if c in df.columns]
print(df[existing_show_cols].head(10).to_string(index=False))