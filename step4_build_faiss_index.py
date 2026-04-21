"""
Step 4 — Build the FAISS index for the full dataset embeddings

What this script does:
1) Loads the full-dataset image embedding file
2) Builds a FAISS index from those embeddings
3) Saves the index to disk for later retrieval

Why we do this:
- After converting all images into embeddings, we need a fast way to search them.
- FAISS is a library designed for efficient similarity search on vectors.
- Instead of comparing the query image against every dataset image manually each time,
  we build an index once and then reuse it for retrieval.

Important note:
- Our embeddings are already normalized before this step.
- Because of that, inner product search is a good choice here.
- With normalized vectors, inner product behaves similarly to cosine similarity,
  which is commonly used in retrieval tasks.
"""

from pathlib import Path
import numpy as np
import faiss


# --------------------------------------------------
# 1) Define input and output paths
# --------------------------------------------------
EMB_PATH = Path("embeddings/full_dataset/image_embeddings.npy")

OUT_DIR = Path("indexes/full_dataset")
OUT_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = OUT_DIR / "faiss_index.index"


# --------------------------------------------------
# 2) Check input file exists
# --------------------------------------------------
if not EMB_PATH.exists():
    raise FileNotFoundError(f"Embedding file not found: {EMB_PATH.resolve()}")


# --------------------------------------------------
# 3) Load the embeddings
# --------------------------------------------------
emb = np.load(EMB_PATH).astype("float32")

if emb.ndim != 2:
    raise ValueError(f"Expected a 2D embedding array, but got shape: {emb.shape}")

if emb.shape[0] == 0:
    raise ValueError("Embedding array is empty.")


# --------------------------------------------------
# 4) Build the FAISS index
# --------------------------------------------------
dim = emb.shape[1]

index = faiss.IndexFlatIP(dim)
index.add(emb)


# --------------------------------------------------
# 5) Save the FAISS index
# --------------------------------------------------
faiss.write_index(index, str(INDEX_PATH))


# --------------------------------------------------
# 6) Print a final summary
# --------------------------------------------------
print("Index saved to:", INDEX_PATH)
print("Number of vectors in index:", index.ntotal)
print("Embedding dimension:", dim)
print("Embedding array shape:", emb.shape)