"""
Step 13 — Prepare train/validation/test splits for fine-tuning

What this script does:
1) Loads the cleaned metadata file
2) Keeps the columns needed for image-text fine-tuning
3) Removes rows with missing or empty required values
4) Builds the training text in one of two user-selected modes:
   - cpu_friendly
   - gpu_full
5) Splits the data into train, validation, and test sets
6) Saves the full fine-tuning table and the three split files
7) Checks that there is no UID overlap between the splits

Why we do this:
- Fine-tuning needs paired image-text examples.
- In our project, each training example is:
    image_path + training_text
- We also need separate train, validation, and test splits so that:
    - training is done on one part of the data
    - model selection is done on validation data
    - final evaluation uses held-out test data

Important design decision:
- We split by uid, not by image row.
- This is very important because one study/report can be linked to more than one image.
- If we split by rows only, images from the same study could leak into different splits.
- That would make the evaluation less trustworthy.

Important note:
- This script supports two text-building modes:
    1) cpu_friendly:
       - uses cleaned full report_text
       - simpler and closer to the earlier local CPU setup

    2) gpu_full:
       - builds a shorter cleaner text target from impression + findings
       - this is the stronger logic and should be the default final setup

- We save a unified column called training_text.
- Later fine-tuning scripts should use training_text directly.
"""

from pathlib import Path
import re
import pandas as pd
import numpy as np


# --------------------------------------------------
# 1) User choice: select the fine-tuning setup mode
# --------------------------------------------------
# Available options:
# - "cpu_friendly"
# - "gpu_full"
#
# Recommendation:
# - use "gpu_full" for the final stronger setup
# - use "cpu_friendly" only if you want the simpler earlier-style text target
SETUP_MODE = "gpu_full"


# --------------------------------------------------
# 2) Define input and output paths
# --------------------------------------------------
INPUT_CSV = Path("data/processed/indiana_metadata_clean.csv")

OUTPUT_DIR = Path("data/processed/finetuning")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = OUTPUT_DIR / "train.csv"
VAL_CSV = OUTPUT_DIR / "val.csv"
TEST_CSV = OUTPUT_DIR / "test.csv"
FULL_CSV = OUTPUT_DIR / "all_finetuning_pairs.csv"


# --------------------------------------------------
# 3) Define the split ratios
# --------------------------------------------------
TRAIN_RATIO = 0.65
VAL_RATIO = 0.15
TEST_RATIO = 0.20

ratio_sum = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
if abs(ratio_sum - 1.0) > 1e-9:
    raise ValueError(f"Split ratios must sum to 1.0, but got {ratio_sum}")


# --------------------------------------------------
# 4) Define text-building settings
# --------------------------------------------------
# These limits are used only in gpu_full mode.
#
# Why shorter text in gpu_full mode?
# - it removes some extra noise
# - it keeps the text target more focused
# - it gave a cleaner training signal in the later stronger setup
MAX_IMPRESSION_WORDS = 40
MAX_FINDINGS_WORDS = 80


# --------------------------------------------------
# 5) Helper functions for text cleaning and text building
# --------------------------------------------------
def clean_text(text):
    """
    Basic text cleaning:
    - handle missing values safely
    - convert to string
    - remove repeated whitespace
    - remove standalone XXXX placeholder tokens
    - strip leading/trailing spaces
    """
    if pd.isna(text):
        return ""

    text = str(text)

    # Remove anonymization-like placeholder tokens such as XXXX
    text = re.sub(r"\bX{2,}\b", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def shorten_to_n_words(text, max_words):
    """
    Keep only the first max_words words.
    This creates a shorter and more controlled text target.
    """
    if not text:
        return ""

    words = text.split()
    return " ".join(words[:max_words])


def build_gpu_style_text(row):
    """
    Build the stronger gpu_full training text from:
    - impression
    - findings

    Logic:
    - clean each text
    - shorten each text separately
    - combine them into one final text
    - avoid repeating the same text twice if impression and findings are identical
    """
    impression = shorten_to_n_words(
        clean_text(row.get("impression", "")),
        MAX_IMPRESSION_WORDS
    )

    findings = shorten_to_n_words(
        clean_text(row.get("findings", "")),
        MAX_FINDINGS_WORDS
    )

    if impression and findings:
        if impression.lower() == findings.lower():
            return impression, "impression_only"
        return f"Impression: {impression} Findings: {findings}", "impression_plus_findings"

    if impression:
        return f"Impression: {impression}", "impression_only"

    if findings:
        return f"Findings: {findings}", "findings_only"

    return "", "missing"


def build_cpu_style_text(row):
    """
    Build the simpler cpu_friendly training text from:
    - cleaned full report_text

    This keeps the earlier simpler logic:
    use the full combined report text instead of the shorter gpu-style target.
    """
    report_text = clean_text(row.get("report_text", ""))

    if report_text:
        return report_text, "report_text"

    return "", "missing"


# --------------------------------------------------
# 6) Check input file and load metadata
# --------------------------------------------------
if not INPUT_CSV.exists():
    raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV.resolve()}")

df = pd.read_csv(INPUT_CSV)
print("Loaded rows from cleaned metadata:", len(df))


# --------------------------------------------------
# 7) Keep only the columns needed for fine-tuning
# --------------------------------------------------
# We keep:
# - uid, filename, image_path, projection
# - findings and impression for gpu_full mode
# - report_text for cpu_friendly mode and traceability
keep_cols = [
    "uid",
    "filename",
    "image_path",
    "projection",
    "findings",
    "impression",
    "report_text",
]

missing_keep_cols = [c for c in keep_cols if c not in df.columns]
if missing_keep_cols:
    raise ValueError(f"Missing required columns in cleaned metadata: {missing_keep_cols}")

df = df[keep_cols].copy()


# --------------------------------------------------
# 8) Final basic cleaning before text building
# --------------------------------------------------
df = df.dropna(subset=["uid", "filename", "image_path"]).copy()

df["image_path"] = df["image_path"].astype(str).str.replace("\\", "/", regex=False)
df["filename"] = df["filename"].astype(str).str.strip()
df["uid"] = df["uid"].astype(str).str.strip()

df = df[df["image_path"] != ""].copy()
df = df[df["filename"] != ""].copy()
df = df[df["uid"] != ""].copy()

print("Rows after basic cleaning:", len(df))

# Quick path check
sample_paths = df["image_path"].head(20).tolist()
missing_sample = sum(not Path(p).exists() for p in sample_paths)
print("Missing among first 20 image paths:", missing_sample)

if missing_sample > 0:
    raise FileNotFoundError("Some sample image paths do not exist after path normalization.")


# --------------------------------------------------
# 9) Build the training text based on the selected mode
# --------------------------------------------------
# Important design:
# We always create the same final output columns:
# - training_text
# - text_source
# - setup_mode
#
# This makes later fine-tuning scripts simpler,
# because they can always use training_text no matter which mode was chosen.
if SETUP_MODE == "gpu_full":
    built_text = df.apply(build_gpu_style_text, axis=1, result_type="expand")
    built_text.columns = ["training_text", "text_source"]

elif SETUP_MODE == "cpu_friendly":
    built_text = df.apply(build_cpu_style_text, axis=1, result_type="expand")
    built_text.columns = ["training_text", "text_source"]

else:
    raise ValueError(
        f"Unsupported SETUP_MODE: {SETUP_MODE}. "
        f"Use 'cpu_friendly' or 'gpu_full'."
    )

df = pd.concat([df, built_text], axis=1)
df["setup_mode"] = SETUP_MODE

# Remove rows where the final training text is empty
df["training_text"] = df["training_text"].astype(str).str.strip()
df = df[df["training_text"] != ""].copy()

print("Rows after building training_text:", len(df))

print("\ntraining_text source counts:")
print(df["text_source"].value_counts(dropna=False).to_string())

print("\nFirst 5 training_text examples:")
print(df[["filename", "text_source", "training_text"]].head(5).to_string(index=False))


# --------------------------------------------------
# 10) Create a fixed UID-based split
# --------------------------------------------------
unique_uids = sorted(df["uid"].unique())

rng = np.random.default_rng(42)
shuffled_uids = np.array(unique_uids, dtype=object)
rng.shuffle(shuffled_uids)


# --------------------------------------------------
# 11) Decide split sizes
# --------------------------------------------------
n = len(shuffled_uids)

n_train = int(TRAIN_RATIO * n)
n_val = int(VAL_RATIO * n)

train_uids = set(shuffled_uids[:n_train])
val_uids = set(shuffled_uids[n_train:n_train + n_val])
test_uids = set(shuffled_uids[n_train + n_val:])


# --------------------------------------------------
# 12) Build the split DataFrames
# --------------------------------------------------
train_df = df[df["uid"].isin(train_uids)].copy()
val_df = df[df["uid"].isin(val_uids)].copy()
test_df = df[df["uid"].isin(test_uids)].copy()

if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
    raise ValueError("At least one split is empty. Check the split ratios and input data.")


# --------------------------------------------------
# 13) Save outputs
# --------------------------------------------------
df.to_csv(FULL_CSV, index=False)
train_df.to_csv(TRAIN_CSV, index=False)
val_df.to_csv(VAL_CSV, index=False)
test_df.to_csv(TEST_CSV, index=False)


# --------------------------------------------------
# 14) Print saved file paths
# --------------------------------------------------
print("Saved full fine-tuning pairs to:", FULL_CSV)
print("Saved train split to:", TRAIN_CSV)
print("Saved val split to:", VAL_CSV)
print("Saved test split to:", TEST_CSV)


# --------------------------------------------------
# 15) Print setup mode and split ratios used
# --------------------------------------------------
print("\nSetup mode used:")
print(SETUP_MODE)

print("\nSplit ratios used:")
print("Train ratio:", TRAIN_RATIO)
print("Val ratio:", VAL_RATIO)
print("Test ratio:", TEST_RATIO)


# --------------------------------------------------
# 16) Print row counts
# --------------------------------------------------
print("\nRow counts:")
print("All:", len(df))
print("Train:", len(train_df))
print("Val:", len(val_df))
print("Test:", len(test_df))


# --------------------------------------------------
# 17) Print unique UID counts
# --------------------------------------------------
print("\nUnique uid counts:")
print("All:", df["uid"].nunique())
print("Train:", train_df["uid"].nunique())
print("Val:", val_df["uid"].nunique())
print("Test:", test_df["uid"].nunique())


# --------------------------------------------------
# 18) Check UID overlap between splits
# --------------------------------------------------
overlap_train_val = set(train_df["uid"]) & set(val_df["uid"])
overlap_train_test = set(train_df["uid"]) & set(test_df["uid"])
overlap_val_test = set(val_df["uid"]) & set(test_df["uid"])

print("\nUID overlap check:")
print("Train-Val overlap:", len(overlap_train_val))
print("Train-Test overlap:", len(overlap_train_test))
print("Val-Test overlap:", len(overlap_val_test))

if overlap_train_val or overlap_train_test or overlap_val_test:
    raise ValueError("UID overlap detected between splits.")


# --------------------------------------------------
# 19) Print final reminder
# --------------------------------------------------
print("\nReminder:")
print("Later fine-tuning scripts should use training_text.")
print("In cpu_friendly mode, training_text comes from cleaned report_text.")
print("In gpu_full mode, training_text comes from the shorter impression/findings target.")
print("If the held-out test split changed, rerun Step 11 and Step 12.")