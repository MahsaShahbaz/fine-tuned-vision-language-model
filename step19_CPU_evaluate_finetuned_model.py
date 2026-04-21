"""
Step 19 — Evaluate the fine-tuned retrieval system - CPU-Friendly version

What this script does:
1) Loads the fine-tuned FAISS index and fine-tuned embedding matrix
2) Loads the aligned fine-tuned metadata and the broad evaluation labels
3) Loads either:
   - the balanced test-only query set, or
   - the expanded test-only query set
4) Runs retrieval for each query image
5) Checks which retrieved results are relevant
6) Computes ranking metrics such as Precision@5, Hit Rate@5,
   Precision@10, and Hit Rate@10
7) Saves both detailed per-query results and summary tables

Why we do this:
- After fine-tuning, we need to measure whether retrieval actually improved.
- It is not enough to say the model was trained successfully.
- We need the same kind of formal retrieval evaluation that we used for the baseline.

What is updated in this version:
- The script now uses the updated fine-tuned embedding and FAISS paths.
- It can evaluate either the balanced or the expanded query set,
  just like the updated baseline evaluation script.
- It uses shorter output folder names to avoid Windows long-path issues.

How relevance is defined here:
- A retrieved image is considered relevant if it has the same broad
  evaluation category as the query image.
- We use the same broad categories as before:
    normal, cardiomegaly, pleural_effusion, opacity, pneumonia

Important evaluation rules:
- We must use the same query-set definition as the baseline run we want to compare against.
- We keep the same-projection rule:
    Frontal queries are compared only with Frontal images,
    Lateral queries are compared only with Lateral images.
- The query image itself is excluded from the retrieved results.

Why this is important:
- We want the baseline and fine-tuned comparison to be as fair as possible.
- That means the only important difference should be the model/embedding database,
  not the evaluation logic.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import faiss


# --------------------------------------------------
# 1) Choose which query set to evaluate on
# --------------------------------------------------
# QUERY_SET_MODE should match the baseline evaluation mode
# if we want a fair side-by-side comparison.
#
# Options:
# - "expanded"
# - "balanced"
#
# Since the updated baseline was run on the expanded query set,
# we keep "expanded" here by default.
QUERY_SET_MODE = "expanded"


# --------------------------------------------------
# 2) Define input paths
# --------------------------------------------------
# INDEX_PATH: saved FAISS index built from the fine-tuned embeddings
# EMB_PATH: fine-tuned image embedding matrix
# META_BASE_PATH: metadata aligned row-by-row with the fine-tuned embeddings
# LABEL_META_PATH: metadata file containing the broad evaluation labels
INDEX_PATH = Path("indexes/ft_full/faiss_ft.index")
EMB_PATH = Path("embeddings/ft_full/img_emb_ft.npy")
META_BASE_PATH = Path("embeddings/ft_full/meta_ft.csv")
LABEL_META_PATH = Path("data/processed/indiana_metadata_eval_labels.csv")


# --------------------------------------------------
# 3) Decide which query-set file and output folder to use
# --------------------------------------------------
if QUERY_SET_MODE == "balanced":
    QUERY_SET_PATH = Path("results/fixed_eval_query_set_balanced_test.csv")
    OUTPUT_DIR = Path("results/ft_eval_bal")
    OUTPUT_TAG = "bal"
elif QUERY_SET_MODE == "expanded":
    QUERY_SET_PATH = Path("results/fixed_eval_query_set_all_single_label_test.csv")
    OUTPUT_DIR = Path("results/ft_eval_exp")
    OUTPUT_TAG = "exp"
else:
    raise ValueError(
        f"Unsupported QUERY_SET_MODE: {QUERY_SET_MODE}. "
        f"Use 'balanced' or 'expanded'."
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PER_QUERY_CSV = OUTPUT_DIR / f"per_query_{OUTPUT_TAG}.csv"
OVERALL_CSV = OUTPUT_DIR / f"overall_{OUTPUT_TAG}.csv"
BY_CATEGORY_CSV = OUTPUT_DIR / f"by_cat_{OUTPUT_TAG}.csv"
BY_PROJECTION_CSV = OUTPUT_DIR / f"by_proj_{OUTPUT_TAG}.csv"


# --------------------------------------------------
# 4) Load the index, embeddings, metadata, and query set
# --------------------------------------------------
# We load:
# - the FAISS index for fine-tuned retrieval
# - the fine-tuned embedding matrix
# - the metadata aligned with that embedding matrix
# - the label metadata containing the broad evaluation labels
# - the final query set
index = faiss.read_index(str(INDEX_PATH))
emb = np.load(EMB_PATH).astype("float32")

meta_base = pd.read_csv(META_BASE_PATH)
label_meta = pd.read_csv(LABEL_META_PATH)
query_df = pd.read_csv(QUERY_SET_PATH)


# --------------------------------------------------
# 5) Keep only the label columns we need and merge them into the metadata
# --------------------------------------------------
# The fine-tuned metadata already has the row order and image-level fields we need,
# but the broad evaluation labels come from the labeled metadata file.
#
# We merge those label columns into the fine-tuned metadata using filename.
# After this, each row in "meta" contains:
# - image information
# - projection
# - broad evaluation labels
#
# This is the same logic used in the baseline evaluation script.
label_cols = [
    "label_normal",
    "label_cardiomegaly",
    "label_pleural_effusion",
    "label_opacity",
    "label_pneumonia",
    "eval_labels",
]

label_subset = label_meta[["filename"] + label_cols].copy()
meta = meta_base.merge(label_subset, on="filename", how="left")


# --------------------------------------------------
# 6) Create a mapping from readable category names to binary label columns
# --------------------------------------------------
# query_category in the query set is a string like "opacity".
# But relevance in the metadata is stored in columns like "label_opacity".
# This dictionary connects the two.
label_map = {
    "normal": "label_normal",
    "cardiomegaly": "label_cardiomegaly",
    "pleural_effusion": "label_pleural_effusion",
    "opacity": "label_opacity",
    "pneumonia": "label_pneumonia",
}


# --------------------------------------------------
# 7) Define retrieval and evaluation settings
# --------------------------------------------------
# We evaluate at top 5 and top 10.
k_values = [5, 10]

# We first ask FAISS for more than 10 results because we later filter out:
# - the query image itself
# - results with the wrong projection
#
# So we search deeper first and keep valid results afterward.
k_search = 200
max_k = max(k_values)


# --------------------------------------------------
# 8) Prepare a list to store per-query evaluation results
# --------------------------------------------------
rows = []


# --------------------------------------------------
# 9) Print a short setup summary
# --------------------------------------------------
print("Fine-tuned evaluation starting...")
print("Query-set mode:", QUERY_SET_MODE)
print("Query set file:", QUERY_SET_PATH)
print("Output folder:", OUTPUT_DIR)
print("Number of queries loaded:", len(query_df))
print("Number of metadata rows:", len(meta))
print("Embedding matrix shape:", emb.shape)
print("FAISS index size:", index.ntotal)


# --------------------------------------------------
# 10) Evaluate each query image one by one
# --------------------------------------------------
# For each query, we:
# - find its row index in the fine-tuned metadata
# - get the matching fine-tuned query embedding
# - search the fine-tuned FAISS index
# - filter the results
# - decide which retrieved images are relevant
# - compute the ranking metrics
for _, query_row in query_df.iterrows():
    query_filename = query_row["filename"]
    query_projection = query_row["projection"]
    query_category = query_row["query_category"]

    if query_category not in label_map:
        print(f"Skipping query with unknown category: {query_filename} | {query_category}")
        continue

    target_label_col = label_map[query_category]

    # ------------------------------------------
    # 10a) Find the query image in the fine-tuned metadata
    # ------------------------------------------
    matches = meta.index[meta["filename"] == query_filename].tolist()

    if len(matches) == 0:
        print(f"Query image not found in metadata: {query_filename}")
        continue

    query_idx = matches[0]

    # Keep the query vector as shape (1, embedding_dim)
    # because FAISS expects a batch dimension.
    query_vec = emb[query_idx:query_idx + 1]

    # ------------------------------------------
    # 10b) Search the fine-tuned FAISS index
    # ------------------------------------------
    scores, indices = index.search(query_vec, k_search)

    # ------------------------------------------
    # 10c) Filter the retrieved results
    # ------------------------------------------
    filtered = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        # Exclude the query image itself
        if idx == query_idx:
            continue

        # Keep only same-projection matches
        if meta.loc[idx, "projection"] != query_projection:
            continue

        filtered.append((idx, float(score)))

        if len(filtered) >= max_k:
            break

    retrieved_indices = [idx for idx, _ in filtered]
    retrieved_scores = [score for _, score in filtered]

    # ------------------------------------------
    # 10d) Decide which retrieved results are relevant
    # ------------------------------------------
    relevant_flags = []
    retrieved_filenames = []

    for idx in retrieved_indices:
        retrieved_filenames.append(meta.loc[idx, "filename"])

        # Check whether this retrieved row has the target label
        is_relevant = int(meta.loc[idx, target_label_col] == 1)
        relevant_flags.append(is_relevant)

    # ------------------------------------------
    # 10e) Save detailed per-query information
    # ------------------------------------------
    row_result = {
        "query_filename": query_filename,
        "projection": query_projection,
        "query_category": query_category,
        "query_eval_labels": query_row.get("eval_labels", ""),
        "num_retrieved_after_filter": len(retrieved_indices),
        "top10_filenames": ";".join(retrieved_filenames[:10]),
        "top10_relevant_flags": ";".join(map(str, relevant_flags[:10])),
        "top10_scores": ";".join([f"{s:.6f}" for s in retrieved_scores[:10]]),
    }

    # ------------------------------------------
    # 10f) Compute ranking metrics at each cutoff
    # ------------------------------------------
    for k in k_values:
        topk_flags = relevant_flags[:k]
        num_relevant = sum(topk_flags)
        precision_k = num_relevant / k
        hit_rate_k = int(num_relevant > 0)

        row_result[f"num_relevant_at_{k}"] = num_relevant
        row_result[f"precision_at_{k}"] = precision_k
        row_result[f"hit_rate_at_{k}"] = hit_rate_k

    rows.append(row_result)


# --------------------------------------------------
# 11) Convert results into a DataFrame
# --------------------------------------------------
results_df = pd.DataFrame(rows)

if len(results_df) == 0:
    raise ValueError(
        "No query results were produced. Please check the query-set file, "
        "metadata alignment, and filename matching."
    )

PER_QUERY_CSV.parent.mkdir(parents=True, exist_ok=True)
OVERALL_CSV.parent.mkdir(parents=True, exist_ok=True)
BY_CATEGORY_CSV.parent.mkdir(parents=True, exist_ok=True)
BY_PROJECTION_CSV.parent.mkdir(parents=True, exist_ok=True)

results_df.to_csv(PER_QUERY_CSV, index=False)


# --------------------------------------------------
# 12) Compute and save the overall summary
# --------------------------------------------------
overall_df = pd.DataFrame([{
    "query_set_mode": QUERY_SET_MODE,
    "num_queries": len(results_df),
    "precision_at_5": results_df["precision_at_5"].mean(),
    "hit_rate_at_5": results_df["hit_rate_at_5"].mean(),
    "precision_at_10": results_df["precision_at_10"].mean(),
    "hit_rate_at_10": results_df["hit_rate_at_10"].mean(),
}])
overall_df.to_csv(OVERALL_CSV, index=False)


# --------------------------------------------------
# 13) Compute and save summary by category
# --------------------------------------------------
by_category_df = (
    results_df.groupby("query_category")[[
        "precision_at_5",
        "hit_rate_at_5",
        "precision_at_10",
        "hit_rate_at_10"
    ]]
    .mean()
    .reset_index()
)
by_category_df.to_csv(BY_CATEGORY_CSV, index=False)


# --------------------------------------------------
# 14) Compute and save summary by projection
# --------------------------------------------------
by_projection_df = (
    results_df.groupby("projection")[[
        "precision_at_5",
        "hit_rate_at_5",
        "precision_at_10",
        "hit_rate_at_10"
    ]]
    .mean()
    .reset_index()
)
by_projection_df.to_csv(BY_PROJECTION_CSV, index=False)


# --------------------------------------------------
# 15) Print a final summary
# --------------------------------------------------
print("\nFine-tuned evaluation finished.")
print("Query set file:", QUERY_SET_PATH)
print("Saved per-query results to:", PER_QUERY_CSV)
print("Saved overall summary to:", OVERALL_CSV)
print("Saved category summary to:", BY_CATEGORY_CSV)
print("Saved projection summary to:", BY_PROJECTION_CSV)

print("\nOverall fine-tuned results:")
print(overall_df.to_string(index=False))

print("\nFine-tuned results by category:")
print(by_category_df.to_string(index=False))

print("\nFine-tuned results by projection:")
print(by_projection_df.to_string(index=False))