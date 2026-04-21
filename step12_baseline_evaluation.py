"""
Step 12 — Evaluate the baseline retrieval system

What this script does:
1) Loads the baseline FAISS index and baseline embedding matrix
2) Loads the aligned metadata and the broad evaluation labels
3) Loads either:
   - the balanced test-only query set, or
   - the expanded test-only query set
4) Runs retrieval for each query image
5) Checks which retrieved results are relevant
6) Computes ranking metrics such as Precision@5, Hit Rate@5,
   Precision@10, and Hit Rate@10
7) Saves both detailed per-query results and summary tables

Why we do this:
- It is not enough to say that retrieval "looks reasonable" by manual inspection.
- We need formal metrics so that we can compare the baseline system
  with the fine-tuned system in a fair and reproducible way.

What is new in this updated version:
- The old version always used only the balanced query set.
- This updated version can use either:
    1) balanced query set
    2) expanded query set
- The expanded query set contains more queries, which is useful when the balanced set
  becomes too small.

How relevance is defined here:
- A retrieved image is considered relevant if it has the same broad
  evaluation category as the query image.
- We use the broad categories defined earlier:
    normal, cardiomegaly, pleural_effusion, opacity, pneumonia

Important evaluation rules:
- Queries must come only from the held-out test split.
- We retrieve only within the same projection type:
    Frontal queries are compared with Frontal images,
    Lateral queries are compared with Lateral images.
- The query image itself is excluded from the retrieved results.

Why same-projection filtering matters:
- Otherwise the model might look better simply because it retrieves the same view,
  not because it retrieves clinically relevant matches.
- So this rule makes the comparison stricter and more meaningful.

Important note:
- This script evaluates the baseline retrieval system only.
- The baseline system means:
    - pretrained CLIP embeddings
    - baseline FAISS index
- Later, we will use almost the same evaluation logic for the fine-tuned system.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import faiss


# --------------------------------------------------
# 1) Choose which query set to evaluate on
# --------------------------------------------------
QUERY_SET_MODE = "expanded"


# --------------------------------------------------
# 2) Define input paths
# --------------------------------------------------
INDEX_PATH = Path("indexes/full_dataset/faiss_index.index")
EMB_PATH = Path("embeddings/full_dataset/image_embeddings.npy")
META_BASE_PATH = Path("embeddings/full_dataset/metadata_full_dataset.csv")
LABEL_META_PATH = Path("data/processed/indiana_metadata_eval_labels.csv")


# --------------------------------------------------
# 3) Decide which query-set file and output folder to use
# --------------------------------------------------
if QUERY_SET_MODE == "balanced":
    QUERY_SET_PATH = Path("results/fixed_eval_query_set_balanced_test.csv")
    OUTPUT_DIR = Path("results/base_eval_bal")
    OUTPUT_TAG = "bal"
elif QUERY_SET_MODE == "expanded":
    QUERY_SET_PATH = Path("results/fixed_eval_query_set_all_single_label_test.csv")
    OUTPUT_DIR = Path("results/base_eval_exp")
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
# 4) Check files and load inputs
# --------------------------------------------------
for p in [INDEX_PATH, EMB_PATH, META_BASE_PATH, LABEL_META_PATH, QUERY_SET_PATH]:
    if not p.exists():
        raise FileNotFoundError(f"Required file not found: {p.resolve()}")

index = faiss.read_index(str(INDEX_PATH))
emb = np.load(EMB_PATH).astype("float32")

meta_base = pd.read_csv(META_BASE_PATH)
label_meta = pd.read_csv(LABEL_META_PATH)
query_df = pd.read_csv(QUERY_SET_PATH)


# --------------------------------------------------
# 5) Validate basic alignment and required columns
# --------------------------------------------------
if emb.ndim != 2:
    raise ValueError(f"Expected embedding array to be 2D, but got shape: {emb.shape}")

if len(meta_base) != emb.shape[0]:
    raise ValueError(
        f"Metadata row count ({len(meta_base)}) does not match embedding count ({emb.shape[0]})."
    )

required_query_cols = ["filename", "projection", "query_category"]
missing_query_cols = [c for c in required_query_cols if c not in query_df.columns]
if missing_query_cols:
    raise ValueError(f"Missing required columns in query file: {missing_query_cols}")

required_label_cols = [
    "filename",
    "label_normal",
    "label_cardiomegaly",
    "label_pleural_effusion",
    "label_opacity",
    "label_pneumonia",
    "eval_labels",
]
missing_label_cols = [c for c in required_label_cols if c not in label_meta.columns]
if missing_label_cols:
    raise ValueError(f"Missing required columns in label metadata: {missing_label_cols}")


# --------------------------------------------------
# 6) Merge label columns into aligned metadata
# --------------------------------------------------
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

if len(meta) != len(meta_base):
    raise ValueError("Merging labels into metadata changed the number of rows.")


# --------------------------------------------------
# 7) Create mapping from readable category names to binary label columns
# --------------------------------------------------
label_map = {
    "normal": "label_normal",
    "cardiomegaly": "label_cardiomegaly",
    "pleural_effusion": "label_pleural_effusion",
    "opacity": "label_opacity",
    "pneumonia": "label_pneumonia",
}


# --------------------------------------------------
# 8) Define retrieval and evaluation settings
# --------------------------------------------------
k_values = [5, 10]
k_search = 200
max_k = max(k_values)


# --------------------------------------------------
# 9) Prepare a list to store per-query evaluation results
# --------------------------------------------------
rows = []
skipped_queries = 0


# --------------------------------------------------
# 10) Print a short setup summary
# --------------------------------------------------
print("Baseline evaluation starting...")
print("Query-set mode:", QUERY_SET_MODE)
print("Query set file:", QUERY_SET_PATH)
print("Output folder:", OUTPUT_DIR)
print("Output folder exists now:", OUTPUT_DIR.exists())
print("Output folder absolute path:", OUTPUT_DIR.resolve())
print("Number of queries loaded:", len(query_df))
print("Number of metadata rows:", len(meta))
print("Embedding matrix shape:", emb.shape)
print("FAISS index size:", index.ntotal)


# --------------------------------------------------
# 11) Evaluate each query image one by one
# --------------------------------------------------
for _, query_row in query_df.iterrows():
    query_filename = query_row["filename"]
    query_projection = query_row["projection"]
    query_category = query_row["query_category"]

    if query_category not in label_map:
        print(f"Skipping query with unknown category: {query_filename} | {query_category}")
        skipped_queries += 1
        continue

    target_label_col = label_map[query_category]

    matches = meta.index[meta["filename"] == query_filename].tolist()

    if len(matches) == 0:
        print(f"Query image not found in metadata: {query_filename}")
        skipped_queries += 1
        continue

    query_idx = matches[0]
    query_vec = emb[query_idx:query_idx + 1]

    scores, indices = index.search(query_vec, k_search)

    filtered = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        if idx == query_idx:
            continue

        if meta.loc[idx, "projection"] != query_projection:
            continue

        filtered.append((int(idx), float(score)))

        if len(filtered) >= max_k:
            break

    retrieved_indices = [idx for idx, _ in filtered]
    retrieved_scores = [score for _, score in filtered]

    relevant_flags = []
    retrieved_filenames = []

    for idx in retrieved_indices:
        retrieved_filenames.append(meta.loc[idx, "filename"])
        is_relevant = int(meta.loc[idx, target_label_col] == 1)
        relevant_flags.append(is_relevant)

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
# 12) Convert results into a DataFrame
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

print("\nSaving files to:")
print(PER_QUERY_CSV.resolve())
print(OVERALL_CSV.resolve())
print(BY_CATEGORY_CSV.resolve())
print(BY_PROJECTION_CSV.resolve())

results_df.to_csv(PER_QUERY_CSV, index=False)


# --------------------------------------------------
# 13) Compute and save the overall summary
# --------------------------------------------------
overall_df = pd.DataFrame([{
    "query_set_mode": QUERY_SET_MODE,
    "num_queries": len(results_df),
    "num_skipped_queries": skipped_queries,
    "precision_at_5": results_df["precision_at_5"].mean(),
    "hit_rate_at_5": results_df["hit_rate_at_5"].mean(),
    "precision_at_10": results_df["precision_at_10"].mean(),
    "hit_rate_at_10": results_df["hit_rate_at_10"].mean(),
}])
overall_df.to_csv(OVERALL_CSV, index=False)


# --------------------------------------------------
# 14) Compute and save summary by category
# --------------------------------------------------
by_category_df = (
    results_df.groupby("query_category")[
        [
            "precision_at_5",
            "hit_rate_at_5",
            "precision_at_10",
            "hit_rate_at_10",
        ]
    ]
    .mean()
    .reset_index()
)
by_category_df.to_csv(BY_CATEGORY_CSV, index=False)


# --------------------------------------------------
# 15) Compute and save summary by projection
# --------------------------------------------------
by_projection_df = (
    results_df.groupby("projection")[
        [
            "precision_at_5",
            "hit_rate_at_5",
            "precision_at_10",
            "hit_rate_at_10",
        ]
    ]
    .mean()
    .reset_index()
)
by_projection_df.to_csv(BY_PROJECTION_CSV, index=False)


# --------------------------------------------------
# 16) Print a final summary
# --------------------------------------------------
print("\nBaseline evaluation finished.")
print("Query set file:", QUERY_SET_PATH)
print("Saved per-query results to:", PER_QUERY_CSV)
print("Saved overall summary to:", OVERALL_CSV)
print("Saved category summary to:", BY_CATEGORY_CSV)
print("Saved projection summary to:", BY_PROJECTION_CSV)

print("\nOverall baseline results:")
print(overall_df.to_string(index=False))

print("\nBaseline results by category:")
print(by_category_df.to_string(index=False))

print("\nBaseline results by projection:")
print(by_projection_df.to_string(index=False))