"""
Step 14 — Create a tiny subset for fine-tuning smoke testing

What this script does:
1) Loads the train and validation split files
2) Samples a very small fixed subset from each split
3) Saves those tiny subsets as separate CSV files
4) Prints a short summary of the result

Why we do this:
- Before running a larger fine-tuning experiment, we want to make sure
  the training pipeline works end to end.
- Running a full training job immediately can be slower and harder to debug.
- So we first create a very small subset that can be used for a quick smoke test.

What "smoke test" means here:
- A smoke test is a small test run to check that the code works at all.
- It is not meant to give strong performance results.
- It is only meant to answer:
    "Can the training script load the data, run forward/backward passes,
     and finish without crashing?"

Important note:
- This step does not define the real training data.
- It only creates a tiny temporary subset for debugging and pipeline verification.
"""

from pathlib import Path
import pandas as pd


# --------------------------------------------------
# 1) Define input and output paths
# --------------------------------------------------
TRAIN_CSV = Path("data/processed/finetuning/train.csv")
VAL_CSV = Path("data/processed/finetuning/val.csv")

OUTPUT_DIR = Path("data/processed/finetuning/tiny_subset")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TINY_TRAIN_CSV = OUTPUT_DIR / "train_tiny.csv"
TINY_VAL_CSV = OUTPUT_DIR / "val_tiny.csv"


# --------------------------------------------------
# 2) Define the tiny subset sizes
# --------------------------------------------------
TINY_TRAIN_SIZE = 200
TINY_VAL_SIZE = 50


# --------------------------------------------------
# 3) Check input files and load splits
# --------------------------------------------------
for p in [TRAIN_CSV, VAL_CSV]:
    if not p.exists():
        raise FileNotFoundError(f"Required file not found: {p.resolve()}")

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

# In the updated unified pipeline, later fine-tuning should use
# training_text instead of older setup-specific text columns.
#
# This keeps the later training scripts simple because:
# - cpu_friendly mode from Step 13 already writes training_text
# - gpu_full mode from Step 13 also writes training_text
#
# So Step 14 does not need separate logic anymore.
required_cols = ["filename", "image_path", "projection", "training_text"]
missing_train = [c for c in required_cols if c not in train_df.columns]
missing_val = [c for c in required_cols if c not in val_df.columns]

if missing_train:
    raise ValueError(f"Missing required columns in train split: {missing_train}")
if missing_val:
    raise ValueError(f"Missing required columns in val split: {missing_val}")


# --------------------------------------------------
# 4) Normalize paths and basic text cleanup
# --------------------------------------------------
# We do a small cleanup here so the tiny subset is safe to use
# in the smoke-test training script.
#
# Why we do this:
# - normalize stored paths
# - remove extra spaces
# - avoid rows with empty image_path or empty training_text
for df in [train_df, val_df]:
    df["image_path"] = df["image_path"].astype(str).str.replace("\\", "/", regex=False)
    df["training_text"] = df["training_text"].astype(str).str.strip()
    df["filename"] = df["filename"].astype(str).str.strip()

train_df = train_df[(train_df["image_path"] != "") & (train_df["training_text"] != "")].copy()
val_df = val_df[(val_df["image_path"] != "") & (val_df["training_text"] != "")].copy()

if len(train_df) == 0 or len(val_df) == 0:
    raise ValueError("Train or validation split is empty after cleanup.")


# --------------------------------------------------
# 5) Decide the actual sample sizes safely
# --------------------------------------------------
n_train = min(TINY_TRAIN_SIZE, len(train_df))
n_val = min(TINY_VAL_SIZE, len(val_df))


# --------------------------------------------------
# 6) Sample tiny subsets
# --------------------------------------------------
# We use random_state=42 so the tiny subsets are reproducible.
train_tiny = train_df.sample(n=n_train, random_state=42).copy()
val_tiny = val_df.sample(n=n_val, random_state=42).copy()


# --------------------------------------------------
# 7) Quick path checks
# --------------------------------------------------
# We do a small path sanity check on the sampled rows before saving.
# This is not a full path audit, but it helps catch obvious issues early.
missing_train_paths = sum(not Path(p).exists() for p in train_tiny["image_path"].head(20))
missing_val_paths = sum(not Path(p).exists() for p in val_tiny["image_path"].head(20))

print("Missing among first 20 tiny-train image paths:", missing_train_paths)
print("Missing among first 20 tiny-val image paths:", missing_val_paths)

if missing_train_paths > 0 or missing_val_paths > 0:
    raise FileNotFoundError("Some sampled tiny-subset image paths do not exist.")


# --------------------------------------------------
# 8) Save the tiny subsets
# --------------------------------------------------
train_tiny.to_csv(TINY_TRAIN_CSV, index=False)
val_tiny.to_csv(TINY_VAL_CSV, index=False)


# --------------------------------------------------
# 9) Print a short summary
# --------------------------------------------------
print("Saved tiny train set to:", TINY_TRAIN_CSV)
print("Saved tiny val set to:", TINY_VAL_CSV)

print("\nTarget subset sizes:")
print("Tiny train target:", TINY_TRAIN_SIZE)
print("Tiny val target:", TINY_VAL_SIZE)

print("\nActual row counts:")
print("Tiny train:", len(train_tiny))
print("Tiny val:", len(val_tiny))

# Print a few sample rows using the updated unified training text field.
print("\nFirst 5 tiny train rows:")
print(train_tiny[["filename", "projection", "training_text"]].head(5).to_string(index=False))