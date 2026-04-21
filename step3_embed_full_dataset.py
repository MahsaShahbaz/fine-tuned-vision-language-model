"""
Step 3 — Generate embeddings for the full cleaned Indiana dataset

What this script does:
1) Loads the cleaned metadata file
2) Loads each image from the image_path column
3) Uses a pretrained CLIP model to create one image embedding per image
4) Normalizes each embedding
5) Saves all embeddings as a NumPy file
6) Saves a copy of the aligned metadata

Why we do this:
- Retrieval needs a searchable numeric representation for every image in the dataset.
- This script creates the full "image database" in embedding form.
- Later, we will build a FAISS index on top of these embeddings for fast similarity search.

Important note:
- In the final version of our project, we use CLIP's projected image embeddings.
- We follow the explicit path:
    vision_model -> pooler_output -> visual_projection
- We chose this on purpose so that the baseline and fine-tuned pipelines use the same
  embedding definition, which makes the comparison cleaner and fairer.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


# --------------------------------------------------
# 1) Define input and output paths
# --------------------------------------------------
INPUT_CSV = Path("data/processed/indiana_metadata_clean.csv")

OUTPUT_DIR = Path("embeddings/full_dataset")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_EMB = OUTPUT_DIR / "image_embeddings.npy"
OUTPUT_META = OUTPUT_DIR / "metadata_full_dataset.csv"


# --------------------------------------------------
# 2) Basic file existence checks
# --------------------------------------------------
if not INPUT_CSV.exists():
    raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV.resolve()}")


# --------------------------------------------------
# 3) Load the cleaned metadata
# --------------------------------------------------
df = pd.read_csv(INPUT_CSV)

if "image_path" not in df.columns:
    raise ValueError("Column 'image_path' not found in input CSV.")

if "filename" not in df.columns:
    raise ValueError("Column 'filename' not found in input CSV.")

# Convert stored paths to Linux-friendly format if needed
df["image_path"] = df["image_path"].astype(str).str.replace("\\", "/", regex=False)


# --------------------------------------------------
# 4) Load the pretrained CLIP model and processor
# --------------------------------------------------
MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
processor = CLIPProcessor.from_pretrained(MODEL_NAME, use_fast=False)
model.eval()

print(f"Using device: {DEVICE}")
print(f"Number of images to embed: {len(df)}")


# --------------------------------------------------
# 5) Quick path check before the long loop
# --------------------------------------------------
sample_paths = df["image_path"].head(20).tolist()
missing_sample = sum(not Path(p).exists() for p in sample_paths)
print(f"Missing among first 20 image paths: {missing_sample}")

if missing_sample > 0:
    raise FileNotFoundError("Some sample image paths do not exist. Check image_path values in the CSV.")


# --------------------------------------------------
# 6) Prepare a list to store all embeddings
# --------------------------------------------------
all_embeddings = []


# --------------------------------------------------
# 7) Loop through all images in the cleaned dataset
# --------------------------------------------------
for i, row in df.iterrows():
    img_path = row["image_path"]

    try:
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(DEVICE)

        with torch.no_grad():
            vision_outputs = model.vision_model(pixel_values=pixel_values)
            pooled_output = vision_outputs.pooler_output
            vec = model.visual_projection(pooled_output)
            vec = vec / vec.norm(dim=-1, keepdim=True)

        all_embeddings.append(vec[0].detach().cpu().numpy())

        if (i + 1) % 100 == 0 or (i + 1) == len(df):
            print(f"[{i+1}/{len(df)}] embedded")

    except Exception as e:
        print(f"ERROR at row {i} | file: {row['filename']} | path: {img_path} | {e}")
        raise


# --------------------------------------------------
# 8) Save the final embedding matrix and aligned metadata
# --------------------------------------------------
emb_array = np.vstack(all_embeddings).astype("float32")

np.save(OUTPUT_EMB, emb_array)
df.to_csv(OUTPUT_META, index=False)


# --------------------------------------------------
# 9) Print final summary
# --------------------------------------------------
print("\nSaved embedding file:", OUTPUT_EMB)
print("Saved metadata file:", OUTPUT_META)
print("Embedding array shape:", emb_array.shape)