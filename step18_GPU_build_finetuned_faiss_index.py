"""
Step 18 — Build the FAISS index for the fine-tuned embeddings - GPU Setup 

What this script does:
1) Loads the fine-tuned full-dataset image embedding file
2) Builds a FAISS index from those embeddings
3) Saves the fine-tuned FAISS index to disk

Why we do this:
- After generating fine-tuned image embeddings for the full dataset,
  we need to make them searchable.
- FAISS provides a fast similarity-search structure for vector retrieval.
- This fine-tuned index will later be used to compare retrieval performance
  against the baseline index under the same evaluation protocol.

Why this is a separate index:
- The baseline embeddings and the fine-tuned embeddings are different databases.
- So each one needs its own FAISS index.
- This lets us run the same retrieval procedure on both systems
  and compare the results fairly.

What is updated in this version:
- The input path now matches the updated Step 17 output folder.
- The output folder name is shorter to avoid long Windows path issues.
- We keep the same exact-search inner-product setup as before,
  because the embeddings are normalized.

Important note:
- Our fine-tuned embeddings are already normalized before this step.
- Because of that, inner product search is an appropriate similarity measure here.
"""

from pathlib import Path
import numpy as np
import faiss


# --------------------------------------------------
# 1) Define input and output paths
# --------------------------------------------------
# EMB_PATH points to the fine-tuned embedding matrix created in the previous step.
# Each row in this file corresponds to one image from the full Indiana dataset.
EMB_PATH = Path("embeddings/ft_full/img_emb_ft.npy")

# OUT_DIR is where we save the FAISS index for the fine-tuned database.
# We keep this separate from the baseline index folder.
#
# We also keep the folder name short to avoid long Windows path issues.
OUT_DIR = Path("indexes/ft_full")
OUT_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = OUT_DIR / "faiss_ft.index"


# --------------------------------------------------
# 2) Check input file exists
# --------------------------------------------------
if not EMB_PATH.exists():
    raise FileNotFoundError(f"Embedding file not found: {EMB_PATH.resolve()}")


# --------------------------------------------------
# 3) Load the fine-tuned embeddings
# --------------------------------------------------
# We load the full fine-tuned embedding matrix and convert it to float32.
# FAISS expects float32 vectors.
emb = np.load(EMB_PATH).astype("float32")

# Basic safety checks so we fail early if something is wrong.
if emb.ndim != 2:
    raise ValueError(f"Expected a 2D embedding array, but got shape: {emb.shape}")

if emb.shape[0] == 0:
    raise ValueError("Fine-tuned embedding array is empty.")


# --------------------------------------------------
# 4) Build the FAISS index
# --------------------------------------------------
# dim is the embedding dimension.
# In our setup, this should be 512.
dim = emb.shape[1]

# We use IndexFlatIP, which performs exact search using inner product.
#
# Why this choice?
# - "Flat" means exact search, which is simple and easy to interpret.
# - "IP" means inner product.
# - Since our embeddings are normalized, inner product works well
#   as a similarity measure for retrieval.
index = faiss.IndexFlatIP(dim)

# Add all fine-tuned embeddings into the index.
index.add(emb)


# --------------------------------------------------
# 5) Save the fine-tuned FAISS index
# --------------------------------------------------
# We save the index so it can be reused later in:
# - the fine-tuned evaluation script
# - the retrieval app
faiss.write_index(index, str(INDEX_PATH))


# --------------------------------------------------
# 6) Print a final summary
# --------------------------------------------------
# index.ntotal tells us how many vectors were added to the index.
# This should match the number of rows in the fine-tuned embedding matrix.
print("Index saved to:", INDEX_PATH)
print("Number of vectors in index:", index.ntotal)
print("Embedding dimension:", dim)
print("Embedding matrix shape:", emb.shape)