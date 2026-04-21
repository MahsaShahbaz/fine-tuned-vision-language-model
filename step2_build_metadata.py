"""
Step 2 — Build one clean metadata table

What this script does:
1) Loads the Indiana report CSV
2) Loads the Indiana projection CSV
3) Merges them using the shared key: uid
4) Creates a full image path for each image
5) Saves one combined metadata CSV for later steps

Why we do this:
- The reports file contains the text information
- The projections file contains the image filenames and projection types
- We need one image-level table that connects each image to its report data
- Later, this table will be the base for cleaning, adding report_text,
  generating embeddings, and running retrieval
"""

from pathlib import Path
import pandas as pd


# --------------------------------------------------
# 1) Define input and output paths
# --------------------------------------------------
# We keep all important file paths at the top of the script
# so they are easy to see and change later if needed.
#
# BASE_DIR points to the main Indiana dataset folder.
# From there, we define:
# - the report CSV
# - the projection CSV
# - the folder containing the actual chest X-ray image files
BASE_DIR = Path("data/raw/indiana")
REPORTS_CSV = BASE_DIR / "indiana_reports.csv"
PROJECTIONS_CSV = BASE_DIR / "indiana_projections.csv"
IMAGE_DIR = BASE_DIR / "images" / "images_normalized"

# OUTPUT_DIR is where we save processed metadata files.
# We create it automatically if it does not already exist.
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "indiana_metadata.csv"


# --------------------------------------------------
# 2) Basic file existence checks
# --------------------------------------------------
# Before loading anything, we check that the expected input files
# and image folder really exist.
#
# Why this is useful:
# - it catches missing-file problems early
# - it gives a clear error message
# - it avoids more confusing errors later in the script
for p in [REPORTS_CSV, PROJECTIONS_CSV, IMAGE_DIR]:
    if not p.exists():
        raise FileNotFoundError(f"Required path not found: {p.resolve()}")


# --------------------------------------------------
# 3) Load the two CSV files
# --------------------------------------------------
# We read both metadata files into pandas DataFrames.
#
# Why two files?
# - indiana_reports.csv contains report-level text information
#   such as findings, impression, Problems, and MeSH
# - indiana_projections.csv contains image-level information
#   such as filename, uid, and projection type
#
# We need both, because retrieval will later happen at the image level,
# but we also want each image to stay linked to its report information.
reports_df = pd.read_csv(REPORTS_CSV)
projections_df = pd.read_csv(PROJECTIONS_CSV)


# --------------------------------------------------
# 4) Merge the two tables on uid
# --------------------------------------------------
# We merge the projection table with the report table using the shared key "uid".
#
# Why uid?
# - uid identifies the study/report
# - one uid can be linked to one or more images
#
# Important logic:
# We use the projection table as the left table because we want
# the result to be image-level, meaning one row per image.
#
# So after merging:
# - each row corresponds to one image
# - that row also includes the related report fields
metadata_df = projections_df.merge(reports_df, on="uid", how="left")


# --------------------------------------------------
# 5) Create a full image path column
# --------------------------------------------------
# The projection CSV gives us the filename,
# but later scripts need the full path to actually open the image file.
#
# So here we create a new column called image_path by joining:
# - the main image folder
# - the filename from each row
#
# We also:
# - convert filename to string
# - strip extra spaces
# - force POSIX-style separators (/)
#
# Why POSIX-style paths?
# - they are safer to store inside CSV files
# - they are more portable across environments
# - they help avoid backslash issues between Windows and Linux
metadata_df["image_path"] = metadata_df["filename"].astype(str).str.strip().apply(
    lambda x: (IMAGE_DIR / x).as_posix()
)


# --------------------------------------------------
# 6) Reorder a few useful columns first
# --------------------------------------------------
# The merge keeps many columns, but we want the most important ones
# to appear first when we inspect the file.
#
# We place these columns at the front:
# - uid
# - filename
# - image_path
# - projection
# - Problems
# - findings
# - impression
#
# We do this mainly to make the file easier to read and debug.
# Any remaining columns are still kept after these preferred columns.
preferred_cols = [
    "uid",
    "filename",
    "image_path",
    "projection",
    "Problems",
    "findings",
    "impression",
]

existing_preferred = [col for col in preferred_cols if col in metadata_df.columns]
remaining_cols = [col for col in metadata_df.columns if col not in existing_preferred]
metadata_df = metadata_df[existing_preferred + remaining_cols]


# --------------------------------------------------
# 7) Save the merged metadata file
# --------------------------------------------------
# We save the combined metadata table as a CSV file.
#
# This file is an important foundation for the rest of the project.
# In later steps, we will:
# - inspect missing values
# - clean rows with no usable report text
# - create one combined report_text column
# - generate image embeddings
# - prepare evaluation labels
metadata_df.to_csv(OUTPUT_CSV, index=False)


# --------------------------------------------------
# 8) Print quick checks
# --------------------------------------------------
# We print a few simple checks so we can confirm that the script worked:
# - where the file was saved
# - how many rows it has
# - which columns it contains
# - the first few rows
#
# This helps us verify early that the image-report linkage looks correct
# before moving to cleaning and embedding generation.
print("Metadata file saved to:", OUTPUT_CSV)
print("Number of rows:", len(metadata_df))
print("Columns:")
print(list(metadata_df.columns))

print("\nFirst 3 rows:")
print(metadata_df.head(3).to_string())

# Check whether a few sample image paths actually exist.
# This is not a full dataset check, but it is a quick sanity check
# that helps catch path-building mistakes early.
sample_paths = metadata_df["image_path"].head(10).tolist()
missing_count = sum(not Path(p).exists() for p in sample_paths)
print(f"\nMissing among first 10 image paths: {missing_count}")