# MedVision Retrieval AI (Case 3)

Medical image retrieval for chest X-rays using CLIP, FAISS, and the Indiana University chest X-ray dataset.

## Project Overview

This project builds a chest X-ray image retrieval system that returns visually and clinically similar historical cases from the Indiana dataset. The system starts with a pretrained CLIP baseline, builds a searchable FAISS index from image embeddings, evaluates retrieval quality using broad clinical relevance labels, and then compares the baseline against fine-tuned versions of the model.

The final system includes:
- a baseline retrieval pipeline
- a fine-tuned retrieval pipeline
- formal retrieval evaluation
- a local Streamlit interface for image upload and comparison

The goal is not diagnosis prediction. The goal is retrieval: given a chest X-ray query image, return similar chest X-ray cases together with related report text.

## Main Features

- Chest X-ray image retrieval using CLIP image embeddings
- Fast similarity search using FAISS
- Cleaned Indiana metadata pipeline linking images and report text
- Broad-category retrieval evaluation using:
  - normal
  - cardiomegaly
  - pleural_effusion
  - opacity
  - pneumonia
- Baseline vs fine-tuned comparison under the same evaluation protocol
- Local Streamlit GUI for:
  - image upload
  - baseline-only retrieval
  - fine-tuned-only retrieval
  - side-by-side comparison

## Technologies Used

### Models and ML libraries
- CLIP (`openai/clip-vit-base-patch32`)
- PyTorch
- Hugging Face Transformers

### Retrieval and data handling
- FAISS
- NumPy
- pandas
- Pillow

### Interface
- Streamlit

### Dataset
This project uses the Indiana University chest X-ray dataset.

The dataset contains:
- chest X-ray image files
- report-level metadata
- image-level metadata
- report text fields such as findings and impression

In our implementation:
- the raw metadata files were merged using `uid`
- a cleaned image-level metadata table was created
- rows without usable report text were removed
- a combined `report_text` field was created from `findings` and `impression`

Final cleaned dataset used in the project:
- 7426 image rows
- valid image paths for all rows
- no missing or empty `report_text` values

## Final Pipeline

The final end-to-end workflow is:

1. Prepare and clean Indiana dataset metadata  
2. Generate baseline image embeddings  
3. Build a baseline FAISS index  
4. Define broad clinical evaluation labels  
5. Build held-out test query sets  
6. Evaluate baseline retrieval  
7. Prepare image-text train/validation/test splits  
8. Run a tiny smoke test for fine-tuning  
9. Choose one fine-tuning route:
   - CPU-friendly route
   - GPU full fine-tuning route
10. Fine-tune CLIP on Indiana image-text pairs  
11. Generate fine-tuned image embeddings  
12. Build a fine-tuned FAISS index  
13. Evaluate fine-tuned retrieval  
14. Run the Streamlit interface for interactive comparison  

## Evaluation Setup

Retrieval was evaluated using:
- query sets built from the held-out test split
- same-projection retrieval only
- exclusion of the query image itself
- relevance defined by matching the same broad evaluation category

Metrics used:
- Precision@5
- Hit Rate@5
- Precision@10
- Hit Rate@10

Two query-set styles were used:
- **Balanced query set**: same number of queries per category and projection group
- **Expanded query set**: all eligible single-label test rows

The balanced query set is methodologically cleaner, but can become very small if one category-projection group is rare. The expanded query set gives broader coverage.

## Final Results

### CPU-Friendly Setup

#### Baseline
- Precision@5 = 0.14
- Hit Rate@5 = 0.30
- Precision@10 = 0.12
- Hit Rate@10 = 0.40

#### Fine-tuned
- Precision@5 = 0.14
- Hit Rate@5 = 0.40
- Precision@10 = 0.16
- Hit Rate@10 = 0.50

#### Interpretation
The CPU-friendly setup gave the first complete local proof of concept for fine-tuning and retrieval comparison. On the held-out balanced test query set, fine-tuning improved Hit Rate@5, Precision@10, and Hit Rate@10, while Precision@5 stayed the same.

These results were encouraging, but they should still be interpreted carefully. The CPU-friendly experiment used a more restricted setup, and the balanced test query set was very small. So this stage was important mainly as an initial proof that the retrieval pipeline, evaluation logic, and fine-tuning workflow could work together end to end.

### GPU Setup

#### Baseline
- Precision@5 = 0.345711
- Hit Rate@5 = 0.765685
- Precision@10 = 0.340077
- Hit Rate@10 = 0.870679

#### Fine-tuned
- Precision@5 = 0.410499
- Hit Rate@5 = 0.774648
- Precision@10 = 0.408451
- Hit Rate@10 = 0.887324

#### Improvement over baseline
- Precision@5: +0.064788
- Hit Rate@5: +0.008963
- Precision@10: +0.068374
- Hit Rate@10: +0.016645

#### Interpretation
The GPU setup produced the strongest final model in the project. On the expanded held-out test query set, the improved fine-tuned model outperformed the baseline on all four overall metrics.

The strongest improvements appeared in:
- cardiomegaly
- opacity
- pleural_effusion

The normal category showed mixed behavior, with higher precision but slightly lower hit rate. Pneumonia remained difficult, likely because that category is still too small for stable evaluation.

This final result suggests that the improved GPU fine-tuning design was more effective than both the earlier CPU-based frozen-encoder experiment and the first end-to-end server fine-tuning attempt. The better performance likely came from the combination of:
- shorter cleaned training text
- gentler optimization
- larger effective batch size
- early stopping

### Overall Interpretation
Taken together, the two setups show two different stages of progress in the project.

The CPU-friendly setup demonstrated that the full retrieval pipeline, formal evaluation logic, and fine-tuning workflow could be implemented successfully under limited hardware conditions.

The GPU setup then extended that work into a stronger end-to-end fine-tuning experiment and produced the best overall retrieval performance in the project.

## Repository Structure

```text
LLM-RESEARCHER-CASE-3-MEDVISION-RETRIVAL-AI-DOMAIN-ADAPTIVE-MEDICAL-IMAGE-SEARCH/
│
├── category_examples_test_only/
├── data/
│   ├── raw/
│   │   └── indiana/
│   └── processed/
│       ├── indiana_metadata.csv
│       ├── indiana_metadata_clean.csv
│       ├── indiana_metadata_eval_labels.csv
│       └── finetuning/
│
├── embeddings/
│   ├── finetuned_full_dataset/
│   │   ├── image_embeddings_finetuned.npy
│   │   └── metadata_finetuned_full_dataset.csv
│   ├── ft_full/
│   │   ├── img_emb_ft.npy
│   │   └── meta_ft.csv
│   ├── full_dataset/
│   │   ├── image_embeddings.npy
│   │   └── metadata_full_dataset.csv
│   └── sample20/
│
├── indexes/
│   ├── finetuned_full_dataset/
│   │   └── faiss_index_finetuned.index
│   ├── ft_full/
│   │   └── faiss_ft.index
│   └── full_dataset/
│       └── faiss_index.index
│
├── results/
│   ├── baseline_evaluation/
│   │   ├── baseline_eval_by_category.csv
│   │   ├── baseline_eval_by_projection.csv
│   │   ├── baseline_eval_overall.csv
│   │   └── baseline_eval_per_query.csv
│   ├── finetuned_evaluation/
│   │   ├── finetuned_eval_by_category.csv
│   │   ├── finetuned_eval_by_projection.csv
│   │   ├── finetuned_eval_overall.csv
│   │   └── finetuned_eval_per_query.csv
│   ├── real_finetune_full/
│   │   ├── best_model.pt
│   │   ├── last_model.pt
│   │   └── training_metrics.csv
│   └── fixed_eval_query_set_all_single_label_test.csv
│
├── .gitignore
├── app.py
├── README.md
├── requirements.txt
├── step1_clip_embedding.py
├── step2_build_metadata.py
├── step3_embed_full_dataset.py
├── step4_build_faiss_index.py
├── step5_query_sample_retrieval.py
├── step6_prepare_evaluation_labels.py
├── step7_assign_eval_categories.py
├── step8_inspect_query_pool.py
├── step9_check_label_overlap.py
├── step10_inspect_single_label_pool.py
├── step11_build_fixed_query_set.py
├── step12_baseline_evaluation.py
├── step13_prepare_finetuning_data.py
├── step14_make_tiny_finetune_subset.py
├── step15_smoke_test_finetune.py
├── step16_CPU_real_finetune_full.py
├── step16_GPU_real_finetune_full.py
├── step17_CPU_generate_finetuned_embeddings.py
├── step17_GPU_generate_finetuned_embeddings.py
├── step18_CPU_build_finetuned_faiss_index.py
├── step18_GPU_build_finetuned_faiss_index.py
├── step19_CPU_evaluate_finetuned_model.py
├── step19_GPU_evaluate_finetuned_model.py
└── test.jpg
```

## Project File Logic

The project is organized as a step-by-step pipeline. Each script has a clear role in building, evaluating, and improving the medical image retrieval system.

The workflow is divided into two main parts:

1. **Baseline retrieval pipeline**
2. **Fine-tuned retrieval pipeline**

The baseline pipeline builds the first working retrieval system using pretrained CLIP embeddings. The fine-tuned pipeline then improves that system by training CLIP on the Indiana chest X-ray image-text pairs and comparing the new retrieval results against the baseline.

### Part A — Baseline Retrieval Pipeline

These files build the original retrieval system before any domain-specific fine-tuning:

- `step1_clip_embedding.py`
- `step2_build_metadata.py`
- `step3_embed_full_dataset.py`
- `step4_build_faiss_index.py`
- `step5_query_sample_retrieval.py`
- `step6_prepare_evaluation_labels.py`
- `step7_assign_eval_categories.py`
- `step8_inspect_query_pool.py`
- `step9_check_label_overlap.py`
- `step10_inspect_single_label_pool.py`
- `step11_build_fixed_query_set.py`
- `step12_baseline_evaluation.py`

### Part B — Fine-Tuning Preparation

These files prepare and test the fine-tuning setup:

- `step13_prepare_finetuning_data.py`
- `step14_make_tiny_finetune_subset.py`
- `step15_smoke_test_finetune.py`

### Part C — Two Separate Fine-Tuning Routes

After the smoke test, the project continues with **two separate routes**:

#### CPU-Friendly Route
- `step16_CPU_real_finetune_full.py`
- `step17_CPU_generate_finetuned_embeddings.py`
- `step18_CPU_build_finetuned_faiss_index.py`
- `step19_CPU_evaluate_finetuned_model.py`

#### GPU Full Fine-Tuning Route
- `step16_GPU_real_finetune_full.py`
- `step17_GPU_generate_finetuned_embeddings.py`
- `step18_GPU_build_finetuned_faiss_index.py`
- `step19_GPU_evaluate_finetuned_model.py`

These routes are kept as separate files because the CPU-friendly and GPU setups are meaningfully different in:
- training design
- trainable vs frozen model parts
- computational cost

Keeping them separate makes the project easier to:
- understand
- debug
- reproduce
- document

### File-by-File Summary

#### `step1_clip_embedding.py`
Loads one image and converts it into a CLIP embedding vector.

Purpose:
- verify that CLIP can process an image
- demonstrate the core idea of retrieval:
  **image -> embedding -> similarity search**

#### `step2_build_metadata.py`
Merges the Indiana report CSV and projection CSV into one image-level metadata table.

Purpose:
- connect each image to its report information
- build a clean metadata foundation for later steps

#### `step3_embed_full_dataset.py`
Generates one CLIP embedding for every image in the cleaned dataset.

Purpose:
- create the full numeric image database for retrieval

#### `step4_build_faiss_index.py`
Builds a FAISS index from the baseline embeddings.

Purpose:
- make similarity search fast and reusable

#### `step5_query_sample_retrieval.py`
Runs manual retrieval on a small fixed query set.

Purpose:
- perform qualitative inspection before formal evaluation

#### `step6_prepare_evaluation_labels.py`
Inspects metadata fields such as `Problems`, `MeSH`, `findings`, and `impression`.

Purpose:
- understand the available metadata before designing evaluation labels

#### `step7_assign_eval_categories.py`
Assigns broad evaluation labels to each image:
- normal
- cardiomegaly
- pleural_effusion
- opacity
- pneumonia

Purpose:
- create a simple relevance definition for retrieval evaluation

#### `step8_inspect_query_pool.py`
Checks how many labeled examples exist in each category and projection type.

Purpose:
- assess whether a balanced evaluation query set is possible

#### `step9_check_label_overlap.py`
Checks how often images receive more than one evaluation label.

Purpose:
- understand label overlap before final query selection

#### `step10_inspect_single_label_pool.py`
Keeps only single-label rows and checks category/projection counts.

Purpose:
- prepare a cleaner evaluation pool

#### `step11_build_fixed_query_set.py`
Builds final test-only query sets:
- balanced query set
- expanded query set

Purpose:
- create fair and reusable evaluation inputs

#### `step12_baseline_evaluation.py`
Evaluates the baseline retrieval system using formal ranking metrics.

Purpose:
- produce the baseline performance numbers used for later comparison

#### `step13_prepare_finetuning_data.py`
Prepares image-text fine-tuning data and creates train/validation/test splits.

Purpose:
- build leakage-safe fine-tuning splits using `uid`

#### `step14_make_tiny_finetune_subset.py`
Creates very small train/validation subsets.

Purpose:
- support a quick smoke test before real training

#### `step15_smoke_test_finetune.py`
Runs a tiny end-to-end fine-tuning smoke test.

Purpose:
- verify that the training code works without committing to a large run

#### `step16_CPU_real_finetune_full.py`
Runs the CPU-friendly fine-tuning experiment.

Purpose:
- provide a lighter fine-tuning route for limited hardware

#### `step17_CPU_generate_finetuned_embeddings.py`
Generates full-dataset embeddings using the CPU-friendly fine-tuned checkpoint.

Purpose:
- rebuild the retrieval database after CPU-friendly fine-tuning

#### `step18_CPU_build_finetuned_faiss_index.py`
Builds a FAISS index from the CPU-friendly fine-tuned embeddings.

Purpose:
- make the CPU-friendly fine-tuned database searchable

#### `step19_CPU_evaluate_finetuned_model.py`
Evaluates the CPU-friendly fine-tuned retrieval system.

Purpose:
- compare CPU-friendly fine-tuned retrieval against the baseline

#### `step16_GPU_real_finetune_full.py`
Runs the main full-model GPU fine-tuning experiment.

Purpose:
- perform the stronger end-to-end fine-tuning setup with full CLIP training

#### `step17_GPU_generate_finetuned_embeddings.py`
Generates full-dataset embeddings using the best GPU fine-tuned checkpoint.

Purpose:
- rebuild the retrieval database after GPU fine-tuning

#### `step18_GPU_build_finetuned_faiss_index.py`
Builds a FAISS index from the GPU fine-tuned embeddings.

Purpose:
- make the GPU fine-tuned database searchable

#### `step19_GPU_evaluate_finetuned_model.py`
Evaluates the GPU fine-tuned retrieval system.

Purpose:
- produce the final fine-tuned retrieval results for fair comparison with baseline

## Installation

### 1. Clone the repository
```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### 2. Create the conda environment
```bash
conda create -n medvision python=3.10
conda activate medvision
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Data Setup

Download the Indiana University chest X-ray dataset manually and place it inside:

```text
data/raw/indiana/
```

Expected important files/folders include:
- `indiana_reports.csv`
- `indiana_projections.csv`
- `images/images_normalized/`

## How to Run the Project

### A. Build the metadata
```bash
python step2_build_metadata.py
```

### B. Generate baseline embeddings and index
```bash
python step3_embed_full_dataset.py
python step4_build_faiss_index.py
```

### C. Inspect manual baseline retrieval
```bash
python step5_query_sample_retrieval.py
```

### D. Prepare evaluation labels and query sets
```bash
python step6_prepare_evaluation_labels.py
python step7_assign_eval_categories.py
python step8_inspect_query_pool.py
python step9_check_label_overlap.py
python step10_inspect_single_label_pool.py
python step11_build_fixed_query_set.py
```

### E. Run baseline evaluation
```bash
python step12_baseline_evaluation.py
```

### F. Prepare fine-tuning data
```bash
python step13_prepare_finetuning_data.py
python step14_make_tiny_finetune_subset.py
python step15_smoke_test_finetune.py
```

### G. Choose one fine-tuning route

#### CPU-friendly route
```bash
python step16_CPU_real_finetune_full.py
python step17_CPU_generate_finetuned_embeddings.py
python step18_CPU_build_finetuned_faiss_index.py
python step19_CPU_evaluate_finetuned_model.py
```

#### GPU route
```bash
python step16_GPU_real_finetune_full.py
python step17_GPU_generate_finetuned_embeddings.py
python step18_GPU_build_finetuned_faiss_index.py
python step19_GPU_evaluate_finetuned_model.py
```

## Run the Application

To launch the Streamlit interface:

```bash
streamlit run app.py
```

## How to Use the Application

1. Launch the Streamlit app  
2. Upload a chest X-ray image (`png`, `jpg`, or `jpeg`)  
3. Choose the projection type of the uploaded image:
   - Frontal
   - Lateral
4. Choose the app mode:
   - Compare Baseline vs Fine-tuned
   - Baseline only
   - Fine-tuned only
5. Choose how many results to display:
   - 3
   - 5
   - 10
6. Click **Run retrieval**
7. Review the returned similar cases, similarity scores, and report text

Notes:
- The search database remains the Indiana chest X-ray dataset
- The uploaded image can come from Indiana or from another source
- If the uploaded image already exists in the Indiana dataset, the app avoids returning the exact same file as the first result


## Limitations

- Broad-category relevance labels were used instead of expert clinical annotation
- This project is a retrieval prototype, not a diagnostic system


## Future Improvements

- Improve rare-category retrieval, especially pneumonia
- Explore stronger medical-domain vision-language models
- Further analyze the difference between CPU-friendly and GPU full fine-tuning setups

## Team Contributions

Team members:
- Mahsa Shahbazi
- Hien Tran


## Acknowledgements

This project was completed as Case 3 of the GPT Lab capstone project from Tampere University (Fine-tuning Large Language Models)

## References

- OpenAI CLIP
- Hugging Face Transformers
- FAISS
- Indiana University Chest X-ray dataset
