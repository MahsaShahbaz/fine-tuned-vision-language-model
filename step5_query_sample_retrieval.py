"""
Step 5 — Run manual retrieval on a small fixed query set

What this script does:
1) Loads the saved FAISS index
2) Loads the full image embedding matrix
3) Loads the aligned metadata table
4) Loads a small fixed query set of selected images
5) Runs retrieval for each query image
6) Prints the top retrieved results for manual inspection

Why we do this:
- Before formal evaluation, we wanted to manually inspect whether retrieval
  looked reasonable on real examples.
- This step helps us check if the pipeline is working end to end:
    metadata -> embeddings -> FAISS search -> retrieved images/reports
- It also helps us understand whether the baseline is mainly capturing
  visual similarity, projection similarity, or something that also looks
  clinically meaningful.

Important note:
- This script is for qualitative inspection, not final evaluation.
- Later, we built a more formal evaluation pipeline with fixed relevance labels
  and ranking metrics such as Precision@5 and Hit Rate@5.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import faiss


# --------------------------------------------------
# 1) Define input paths
# --------------------------------------------------
INDEX_PATH = Path("indexes/full_dataset/faiss_index.index")
EMB_PATH = Path("embeddings/full_dataset/image_embeddings.npy")
META_PATH = Path("embeddings/full_dataset/metadata_full_dataset.csv")
QUERY_SET_PATH = Path("results/manual_query_set_10.csv")


# --------------------------------------------------
# 2) Check required files exist
# --------------------------------------------------
for p in [INDEX_PATH, EMB_PATH, META_PATH, QUERY_SET_PATH]:
    if not p.exists():
        raise FileNotFoundError(f"Required file not found: {p.resolve()}")


# --------------------------------------------------
# 3) Load the index, embeddings, metadata, and query set
# --------------------------------------------------
index = faiss.read_index(str(INDEX_PATH))
emb = np.load(EMB_PATH).astype("float32")
meta = pd.read_csv(META_PATH)
query_set = pd.read_csv(QUERY_SET_PATH)

if "filename" not in meta.columns:
    raise ValueError("Column 'filename' not found in metadata file.")

if "filename" not in query_set.columns:
    raise ValueError("Column 'filename' not found in query set file.")

if emb.ndim != 2:
    raise ValueError(f"Expected embedding array to be 2D, but got shape: {emb.shape}")

if len(meta) != emb.shape[0]:
    raise ValueError(
        f"Metadata row count ({len(meta)}) does not match embedding count ({emb.shape[0]})."
    )


# --------------------------------------------------
# 4) Set the number of neighbors to search
# --------------------------------------------------
k = min(6, len(meta))


# --------------------------------------------------
# 5) Choose which metadata columns to print
# --------------------------------------------------
display_cols = [col for col in ["filename", "projection", "report_text"] if col in meta.columns]
if not display_cols:
    raise ValueError("None of the expected display columns were found in metadata.")


# --------------------------------------------------
# 6) Loop through each query image in the fixed query set
# --------------------------------------------------
for _, query_row in query_set.iterrows():
    query_filename = query_row["filename"]

    matches = meta.index[meta["filename"] == query_filename].tolist()

    if len(matches) == 0:
        print(f"Query image not found in metadata: {query_filename}")
        print("=" * 100)
        continue

    query_idx = matches[0]
    query_vec = emb[query_idx:query_idx + 1]

    scores, indices = index.search(query_vec, k)

    print("=" * 100)
    print("Query image:")
    print(meta.loc[query_idx, display_cols].to_string())
    print("\nTop retrieved results:\n")

    rank_num = 1

    for score, idx in zip(scores[0], indices[0]):
        if idx == query_idx:
            continue

        print(f"Rank {rank_num}")
        print("Row index:", int(idx))
        print("Score:", float(score))
        print(meta.loc[idx, display_cols].to_string())
        print("-" * 80)

        rank_num += 1
        if rank_num > 5:
            break

    print()