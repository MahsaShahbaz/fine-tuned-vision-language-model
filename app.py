"""
Streamlit App — MedVision Retrieval Interface

What this app does:
1) Loads the baseline and fine-tuned retrieval resources
2) Lets the user upload a chest X-ray image
3) Lets the user choose:
   - Baseline only
   - Fine-tuned only
   - Side-by-side comparison
4) Generates an embedding for the uploaded image
5) Searches the Indiana chest X-ray database for similar cases
6) Displays the retrieved images, similarity scores, projection type, and report text
7) If the uploaded image belongs to the Indiana dataset, also shows the report text
   of the uploaded query image itself using its filename

Why we built this app:
- The earlier scripts let us build and evaluate the retrieval pipeline,
  but they are all command-line based.
- This app gives us an interactive way to demonstrate the project.
- It is useful for screenshots, qualitative comparison, and the final demo.

Important note:
- The app always searches inside the Indiana retrieval database.
- However, the uploaded query image can come either from Indiana
  or from another dataset or local file.
- We also keep the same embedding logic here as in the final pipeline:
    vision_model -> pooler_output -> visual_projection
- This makes the app consistent with both the baseline and fine-tuned retrieval setup.

What is updated in this version:
- Fine-tuned resource paths now match the updated Steps 16–18 outputs
- Shorter folder/file names are used to avoid long Windows path issues
- Optional path remapping is supported for displaying retrieved dataset images
- GPU is used automatically if available, otherwise CPU is used
- The app now also tries to show the report text of the uploaded query image
  by matching its filename against the dataset metadata
"""

from pathlib import Path
from PIL import Image, ImageOps
import pandas as pd
import faiss
import streamlit as st
import torch
from transformers import CLIPProcessor, CLIPModel


# --------------------------------------------------
# 1) Basic Streamlit page setup
# --------------------------------------------------
st.set_page_config(page_title="MedVision Retrieval", layout="wide")


# --------------------------------------------------
# 2) Define file paths
# --------------------------------------------------
BASELINE_INDEX_PATH = Path("indexes/full_dataset/faiss_index.index")
BASELINE_META_PATH = Path("embeddings/full_dataset/metadata_full_dataset.csv")

FINETUNED_INDEX_PATH = Path("indexes/ft_full/faiss_ft.index")
FINETUNED_META_PATH = Path("embeddings/ft_full/meta_ft.csv")
FINETUNED_CKPT_PATH = Path("results/ft16_real/best.pt")

MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------
# 3) Optional path remapping for displayed dataset images
# --------------------------------------------------
PATH_PREFIX_OLD = None
PATH_PREFIX_NEW = None


# --------------------------------------------------
# 4) Helper function: resolve dataset image path
# --------------------------------------------------
def resolve_image_path(path_str: str) -> str:
    path_str = str(path_str).replace("\\", "/")

    if PATH_PREFIX_OLD is not None and PATH_PREFIX_NEW is not None:
        if path_str.startswith(PATH_PREFIX_OLD):
            path_str = path_str.replace(PATH_PREFIX_OLD, PATH_PREFIX_NEW, 1)

    return path_str


# --------------------------------------------------
# 5) Helper function: check required app files
# --------------------------------------------------
def check_required_files():
    missing = []

    for p in [BASELINE_INDEX_PATH, BASELINE_META_PATH]:
        if not p.exists():
            missing.append(str(p))

    for p in [FINETUNED_INDEX_PATH, FINETUNED_META_PATH, FINETUNED_CKPT_PATH]:
        if not p.exists():
            missing.append(str(p))

    return missing


# --------------------------------------------------
# 6) Load baseline resources once and cache them
# --------------------------------------------------
@st.cache_resource
def load_baseline_resources():
    index = faiss.read_index(str(BASELINE_INDEX_PATH))
    meta = pd.read_csv(BASELINE_META_PATH).copy()

    if "image_path" in meta.columns:
        meta["image_path_runtime"] = meta["image_path"].astype(str).apply(resolve_image_path)

    processor = CLIPProcessor.from_pretrained(MODEL_NAME, use_fast=False)
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    return index, meta, processor, model


# --------------------------------------------------
# 7) Load fine-tuned resources once and cache them
# --------------------------------------------------
@st.cache_resource
def load_finetuned_resources():
    index = faiss.read_index(str(FINETUNED_INDEX_PATH))
    meta = pd.read_csv(FINETUNED_META_PATH).copy()

    if "image_path_runtime" not in meta.columns and "image_path" in meta.columns:
        meta["image_path_runtime"] = meta["image_path"].astype(str).apply(resolve_image_path)

    processor = CLIPProcessor.from_pretrained(MODEL_NAME, use_fast=False)
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)

    state_dict = torch.load(FINETUNED_CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return index, meta, processor, model


# --------------------------------------------------
# 8) Load metadata for query-image report lookup
# --------------------------------------------------
# We only need metadata here, not the full model/index.
# Baseline metadata is enough because it is aligned to the Indiana dataset
# and contains the report text for those images.
@st.cache_data
def load_query_lookup_metadata():
    meta = pd.read_csv(BASELINE_META_PATH).copy()
    return meta


# --------------------------------------------------
# 9) Helper function: find report text for uploaded query image
# --------------------------------------------------
# This tries to match the uploaded filename exactly to the metadata filename.
# If the uploaded image comes from Indiana and keeps the same filename,
# we can display its own report text.
def find_query_report_by_filename(uploaded_filename):
    meta = load_query_lookup_metadata()

    if "filename" not in meta.columns:
        return None

    matches = meta[meta["filename"].astype(str) == str(uploaded_filename)].copy()

    if len(matches) == 0:
        return None

    row = matches.iloc[0]

    return {
        "filename": row.get("filename", ""),
        "projection": row.get("projection", ""),
        "report_text": row.get("report_text", ""),
    }


# --------------------------------------------------
# 10) Helper function: embed the uploaded query image
# --------------------------------------------------
def embed_query_image(image, processor, model):
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE)

    with torch.no_grad():
        vision_outputs = model.vision_model(pixel_values=pixel_values)
        pooled_output = vision_outputs.pooler_output
        vec = model.visual_projection(pooled_output)

    vec = vec / vec.norm(dim=-1, keepdim=True)

    return vec.cpu().numpy().astype("float32")


# --------------------------------------------------
# 11) Helper function: retrieve similar results
# --------------------------------------------------
def retrieve_results(model_choice, image, query_projection, top_k, uploaded_filename):
    if model_choice == "Baseline CLIP":
        index, meta, processor, model = load_baseline_resources()
    else:
        index, meta, processor, model = load_finetuned_resources()

    query_vec = embed_query_image(image, processor, model)

    k_search = max(300, top_k * 30)
    scores, indices = index.search(query_vec, k_search)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        row = meta.loc[idx]

        row_filename = row["filename"] if "filename" in row else ""
        row_projection = row["projection"] if "projection" in row else None

        if row_filename == uploaded_filename:
            continue

        if row_projection != query_projection:
            continue

        display_path = row["image_path_runtime"] if "image_path_runtime" in row else row.get("image_path", "")

        results.append({
            "rank": len(results) + 1,
            "filename": row_filename,
            "projection": row_projection,
            "score": float(score),
            "report_text": row.get("report_text", ""),
            "image_path": display_path,
        })

        if len(results) >= top_k:
            break

    return results


# --------------------------------------------------
# 12) Helper function: prepare a dataset image for display
# --------------------------------------------------
def prepare_display_image_from_path(image_path, canvas_size=(512, 512), bg_color=0):
    img_path = Path(image_path)

    if not img_path.exists():
        return None

    try:
        img = Image.open(img_path).convert("RGB")
        fitted = ImageOps.contain(img, canvas_size)

        canvas = Image.new("RGB", canvas_size, color=(bg_color, bg_color, bg_color))
        x = (canvas_size[0] - fitted.width) // 2
        y = (canvas_size[1] - fitted.height) // 2
        canvas.paste(fitted, (x, y))

        return canvas

    except Exception:
        return None


# --------------------------------------------------
# 13) Helper function: prepare the uploaded query image for display
# --------------------------------------------------
def prepare_display_image_from_pil(image, canvas_size=(320, 320), bg_color=0):
    try:
        fitted = ImageOps.contain(image, canvas_size)

        canvas = Image.new("RGB", canvas_size, color=(bg_color, bg_color, bg_color))
        x = (canvas_size[0] - fitted.width) // 2
        y = (canvas_size[1] - fitted.height) // 2
        canvas.paste(fitted, (x, y))

        return canvas

    except Exception:
        return image


# --------------------------------------------------
# 14) Helper function: render one result card
# --------------------------------------------------
def render_result_card(item, title_prefix):
    prefix = f"{title_prefix} " if title_prefix else ""
    st.markdown(f"### {prefix}Rank {item['rank']}")

    display_img = prepare_display_image_from_path(item["image_path"], canvas_size=(512, 512))

    if display_img is not None:
        st.image(display_img, caption=item["filename"], use_container_width=True)
    else:
        st.write("Could not load image.")

    st.write(f"**Filename:** {item['filename']}")
    st.write(f"**Projection:** {item['projection']}")
    st.write(f"**Similarity score:** {item['score']:.4f}")

    with st.expander("Show report text"):
        st.write(item["report_text"])


# --------------------------------------------------
# 15) Helper function: render a list of results
# --------------------------------------------------
def render_results_block(title, results):
    st.subheader(title)

    if len(results) == 0:
        st.warning("No results found.")
        return

    for item in results:
        render_result_card(item, "")
        st.markdown("---")


# --------------------------------------------------
# 16) Build the main app UI
# --------------------------------------------------
st.title("MedVision Retrieval")
st.write("Upload a chest X-ray image and compare retrieved similar cases.")
st.caption(
    "You can upload a chest X-ray from Indiana or from another dataset. "
    "The search database remains the Indiana chest X-ray collection."
)

st.caption(f"Running on device: {DEVICE}")


# --------------------------------------------------
# 17) Check that retrieval resources exist
# --------------------------------------------------
missing_files = check_required_files()
if missing_files:
    st.error("Some required retrieval files are missing.")
    for p in missing_files:
        st.write(p)
    st.stop()


# --------------------------------------------------
# 18) Sidebar controls
# --------------------------------------------------
st.sidebar.header("Settings")

app_mode = st.sidebar.selectbox(
    "App mode",
    ["Compare Baseline vs Fine-tuned", "Baseline only", "Fine-tuned only"]
)

top_k = st.sidebar.selectbox("Number of results", [3, 5, 10], index=1)

query_projection = st.sidebar.selectbox(
    "Projection of uploaded image",
    ["Frontal", "Lateral"]
)


# --------------------------------------------------
# 19) File uploader
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["png", "jpg", "jpeg"]
)


# --------------------------------------------------
# 20) Main app behavior after file upload
# --------------------------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    query_display = prepare_display_image_from_pil(image, canvas_size=(320, 320))

    st.subheader("Query image")
    left_spacer, center_col, right_spacer = st.columns([1, 2, 1])
    with center_col:
        st.image(query_display, caption=uploaded_file.name, use_container_width=True)

    # Try to find report text for the uploaded query image itself
    query_report_info = find_query_report_by_filename(uploaded_file.name)

    st.subheader("Query image report text")

    if query_report_info is not None:
        st.write(f"**Filename:** {query_report_info['filename']}")

        projection_value = query_report_info["projection"]
        if pd.notna(projection_value) and str(projection_value).strip() != "":
            st.write(f"**Projection in metadata:** {projection_value}")

        report_text_value = query_report_info["report_text"]

        with st.expander("Show query image report text", expanded=True):
            if pd.isna(report_text_value) or str(report_text_value).strip() == "":
                st.write("Report text is empty for this image.")
            else:
                st.write(str(report_text_value))
    else:
        st.info(
            "No matching report text was found for this uploaded image. "
            "This usually means the uploaded file is not from the Indiana dataset "
            "or its filename does not exactly match the metadata."
        )

    if st.button("Run retrieval"):
        if app_mode == "Compare Baseline vs Fine-tuned":
            baseline_results = retrieve_results(
                "Baseline CLIP", image, query_projection, top_k, uploaded_file.name
            )
            finetuned_results = retrieve_results(
                "Fine-tuned CLIP", image, query_projection, top_k, uploaded_file.name
            )

            st.subheader("Baseline vs Fine-tuned Comparison")

            max_rows = max(len(baseline_results), len(finetuned_results))

            for i in range(max_rows):
                left_col, right_col = st.columns(2)

                with left_col:
                    if i < len(baseline_results):
                        render_result_card(baseline_results[i], "Baseline")
                    else:
                        st.write("No result")

                with right_col:
                    if i < len(finetuned_results):
                        render_result_card(finetuned_results[i], "Fine-tuned")
                    else:
                        st.write("No result")

                st.markdown("---")

        elif app_mode == "Baseline only":
            baseline_results = retrieve_results(
                "Baseline CLIP", image, query_projection, top_k, uploaded_file.name
            )
            render_results_block("Baseline CLIP Results", baseline_results)

        else:
            finetuned_results = retrieve_results(
                "Fine-tuned CLIP", image, query_projection, top_k, uploaded_file.name
            )
            render_results_block("Fine-tuned CLIP Results", finetuned_results)

else:
    st.info("Please upload an image to continue.")