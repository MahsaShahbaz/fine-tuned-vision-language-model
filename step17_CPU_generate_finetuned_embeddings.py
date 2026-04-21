"""
Step 17 — Generate fine-tuned image embeddings for the full dataset
# step17_generate_finetuned_embeddings.py - CPU-Friendly version

What this script does:
1) Loads the full-dataset metadata file
2) Loads the best fine-tuned CLIP checkpoint
3) Loads each image from the metadata
4) Generates one fine-tuned image embedding per image
5) Normalizes each embedding
6) Saves the full fine-tuned embedding matrix
7) Saves a copy of the aligned metadata

Why we do this:
- After fine-tuning the model, we need a new set of image embeddings.
- The old baseline embeddings were created by the pretrained model.
- To evaluate whether fine-tuning helped retrieval, we must regenerate
  the entire dataset embedding database using the fine-tuned model.

Why this matters:
- Retrieval works by comparing the query embedding to all database embeddings.
- If the model has changed after fine-tuning, the database embeddings must also change.
- Otherwise, we would be comparing a fine-tuned query representation
  against an old baseline database, which would not be a fair test.

What is updated in this version:
- The checkpoint path now matches the updated Step 16 output folder.
- The script uses CUDA automatically if available.
- The script supports optional image-path remapping, which is useful if the metadata
  file was created on a different machine than the one currently running the code.
- The script generates embeddings in batches instead of one image at a time,
  which is faster and more practical on GPU.

Important note:
- In our final pipeline, both the baseline and fine-tuned systems use the same
  explicit embedding definition:
      vision_model -> pooler_output -> visual_projection
- This keeps the comparison method consistent and cleaner.
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel


# --------------------------------------------------
# 1) Define input and output paths
# --------------------------------------------------
# INPUT_CSV:
# We use the metadata file aligned with the full baseline embedding database.
# It contains one row per image and, importantly, the image_path.
#
# CHECKPOINT_PATH:
# This is the best fine-tuned model selected from the updated Step 16 run.
INPUT_CSV = Path("embeddings/full_dataset/metadata_full_dataset.csv")
CHECKPOINT_PATH = Path("results/ft16_real/best.pt")

# OUTPUT_DIR:
# We save the fine-tuned embedding database in a separate folder
# so it stays separate from the baseline embeddings.
#
# We keep the folder name short to avoid Windows long-path issues.
OUTPUT_DIR = Path("embeddings/ft_full")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_EMB = OUTPUT_DIR / "img_emb_ft.npy"
OUTPUT_META = OUTPUT_DIR / "meta_ft.csv"


# --------------------------------------------------
# 2) Define the runtime settings
# --------------------------------------------------
# MODEL_NAME:
# We use the same CLIP backbone architecture as before.
# The difference now is that we will load the fine-tuned checkpoint weights.
MODEL_NAME = "openai/clip-vit-base-patch32"

# DEVICE:
# Use GPU if available, otherwise fall back to CPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# BATCH_SIZE:
# Larger than training because this is inference only.
# If you later see out-of-memory errors on GPU, reduce this.
BATCH_SIZE = 32

# NUM_WORKERS:
# Keep this at 0 for maximum compatibility, especially on Windows.
NUM_WORKERS = 0


# --------------------------------------------------
# 3) Optional path remapping settings
# --------------------------------------------------
# These are useful if the metadata CSV was created on a different machine.
#
# Example use case:
# - CSV contains Windows paths from your laptop
# - you want to run the script on the Kudos server
#
# If PATH_PREFIX_OLD and PATH_PREFIX_NEW are both not None,
# then the script will replace the old path prefix with the new one.
#
# If you rerun the metadata-building steps on the server, you may not need this.
PATH_PREFIX_OLD = None
PATH_PREFIX_NEW = None


# --------------------------------------------------
# 4) Helper function: resolve image path
# --------------------------------------------------
# This function optionally remaps an image path from one machine layout
# to another machine layout.
def resolve_image_path(path_str: str) -> str:
    path_str = str(path_str)

    if PATH_PREFIX_OLD is not None and PATH_PREFIX_NEW is not None:
        if path_str.startswith(PATH_PREFIX_OLD):
            path_str = path_str.replace(PATH_PREFIX_OLD, PATH_PREFIX_NEW, 1)

    return path_str


# --------------------------------------------------
# 5) Load the full metadata
# --------------------------------------------------
# This file gives us:
# - the row order for all images
# - the image_path for each image
#
# As before, row order matters.
# Later, row i in this metadata file must match row i in the fine-tuned embedding matrix.
df = pd.read_csv(INPUT_CSV).copy()

# Apply optional image-path remapping
df["image_path_runtime"] = df["image_path"].astype(str).apply(resolve_image_path)

print("Device:", DEVICE)
print("Metadata rows:", len(df))
print("Checkpoint path:", CHECKPOINT_PATH)
print("Output embedding file:", OUTPUT_EMB)
print("Output metadata file:", OUTPUT_META)


# --------------------------------------------------
# 6) Define a Dataset class for image loading
# --------------------------------------------------
# We use a Dataset + DataLoader here so that we can process images in batches.
class ImageOnlyDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.reset_index(drop=True).copy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path_runtime"]

        # Load the image and convert it to RGB,
        # which is what CLIP expects.
        image = Image.open(img_path).convert("RGB")

        # We also return the row index so we can preserve exact alignment.
        return image, idx


dataset = ImageOnlyDataset(df)


# --------------------------------------------------
# 7) Load the CLIP processor and model
# --------------------------------------------------
# First we load the standard CLIP processor and base model structure.
# Then we load our saved fine-tuned weights into that model.
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)

# Load the fine-tuned checkpoint from disk
state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

# strict=False allows the checkpoint to load even if some non-critical keys differ.
# In our case, this was a practical choice that worked safely for our setup.
model.load_state_dict(state_dict, strict=False)

# model.eval() switches the model into inference mode.
# This is important because we are generating embeddings, not training.
model.eval()

print("Loaded checkpoint successfully.")


# --------------------------------------------------
# 8) Define a collate function for batched preprocessing
# --------------------------------------------------
# The processor converts a list of PIL images into model-ready tensors.
def collate_fn(batch):
    images, indices = zip(*batch)

    enc = processor(
        images=list(images),
        return_tensors="pt"
    )

    return enc, list(indices)


# --------------------------------------------------
# 9) Create the DataLoader
# --------------------------------------------------
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn
)


# --------------------------------------------------
# 10) Prepare a list to store all fine-tuned embeddings
# --------------------------------------------------
# We store embeddings in row order so that:
# - row i in emb_array
# - matches row i in OUTPUT_META
all_embeddings = [None] * len(df)


# --------------------------------------------------
# 11) Loop through all images and generate fine-tuned embeddings
# --------------------------------------------------
# For each batch, we:
# - preprocess the images
# - pass them through the fine-tuned model
# - normalize the resulting embeddings
# - store each embedding in the correct row position
processed_count = 0

with torch.no_grad():
    for batch_num, (enc, indices) in enumerate(loader, start=1):
        pixel_values = enc["pixel_values"].to(DEVICE)

        # Use the same explicit embedding path as in the baseline pipeline.
        vision_outputs = model.vision_model(pixel_values=pixel_values)
        pooled_output = vision_outputs.pooler_output
        vec = model.visual_projection(pooled_output)

        # Normalize each embedding to length 1.
        vec = vec / vec.norm(dim=-1, keepdim=True)

        # Move to CPU and store each row vector in the correct order.
        vec_np = vec.cpu().numpy()

        for local_i, row_idx in enumerate(indices):
            all_embeddings[row_idx] = vec_np[local_i]

        processed_count += len(indices)

        if processed_count % 200 == 0 or processed_count == len(df):
            print(f"[{processed_count}/{len(df)}] embedded")


# --------------------------------------------------
# 12) Final safety check
# --------------------------------------------------
# Make sure every row received an embedding.
if any(v is None for v in all_embeddings):
    raise ValueError("Some embeddings were not generated correctly.")


# --------------------------------------------------
# 13) Save the fine-tuned embedding matrix and aligned metadata
# --------------------------------------------------
# np.vstack combines all row vectors into one 2D array:
# - one row per image
# - one column per embedding dimension
#
# We convert to float32 because:
# - it is the standard format for FAISS
# - it uses less memory
emb_array = np.vstack(all_embeddings).astype("float32")

# Save the embedding matrix
np.save(OUTPUT_EMB, emb_array)

# Save aligned metadata in the same row order.
# We keep both the original path and the runtime-resolved path for transparency.
df.to_csv(OUTPUT_META, index=False)


# --------------------------------------------------
# 14) Print the final summary
# --------------------------------------------------
print("\nSaved fine-tuned embeddings to:", OUTPUT_EMB)
print("Saved aligned metadata to:", OUTPUT_META)
print("Embedding array shape:", emb_array.shape)