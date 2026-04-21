"""
Step 15 — Run a tiny end-to-end fine-tuning smoke test

What this script does:
1) Loads a tiny train subset and a tiny validation subset
2) Builds a simple image-text dataset for CLIP
3) Loads a pretrained CLIP model and processor
4) Keeps the full model trainable (no freezing)
5) Runs a very small training loop
6) Runs a short validation loop
7) Saves the training metrics and a checkpoint

Why we do this:
- Before running a larger real fine-tuning experiment, we want to make sure
  the training pipeline works end to end.
- This step is mainly for debugging and verification, not for strong performance.
- It helps us answer:
    "Can the model load, read the data, run forward and backward passes,
     compute loss, update parameters, and finish without crashing?"

What is different from the older version:
- The older version froze the heavy vision and text encoders to make the run lighter.
- That meant it was not a true end-to-end fine-tuning test.
- In this updated version, we do NOT freeze any part of CLIP.
- This makes the smoke test match the real fine-tuning logic more closely.

Important note:
- This is still only a smoke test.
- We intentionally keep it tiny:
    - 1 epoch
    - only 2 training batches
    - only 1 validation batch
    - very small batch size
- The goal is functional verification, not strong retrieval performance.

Device note:
- If CUDA is available, the script will use GPU automatically.
- Otherwise, it will fall back to CPU.
- On CPU, this step may still be slower, but because it is extremely small,
  it is still reasonable as a smoke test.
"""

from pathlib import Path
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from torch.optim import AdamW


# --------------------------------------------------
# 1) Define input and output paths
# --------------------------------------------------
TRAIN_CSV = Path("data/processed/finetuning/tiny_subset/train_tiny.csv")
VAL_CSV = Path("data/processed/finetuning/tiny_subset/val_tiny.csv")

OUTPUT_DIR = Path("results/finetune_smoke_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS_CSV = OUTPUT_DIR / "smoke_test_metrics.csv"
CHECKPOINT_PATH = OUTPUT_DIR / "smoke_test_checkpoint.pt"


# --------------------------------------------------
# 2) Define the basic settings for the smoke test
# --------------------------------------------------
MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
EPOCHS = 1
MAX_TRAIN_BATCHES = 2
MAX_VAL_BATCHES = 1
LR = 1e-5
WEIGHT_DECAY = 1e-4


# --------------------------------------------------
# 3) Check input files
# --------------------------------------------------
for p in [TRAIN_CSV, VAL_CSV]:
    if not p.exists():
        raise FileNotFoundError(f"Required file not found: {p.resolve()}")


# --------------------------------------------------
# 4) Define a simple Dataset class for image-text pairs
# --------------------------------------------------
class ImageTextDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path).copy()

        # In the updated unified pipeline, later fine-tuning uses
        # training_text instead of older setup-specific columns such as
        # report_text or finetune_text.
        required_cols = ["image_path", "training_text", "filename"]
        missing_cols = [c for c in required_cols if c not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in {csv_path}: {missing_cols}")

        self.df["image_path"] = self.df["image_path"].astype(str).str.replace("\\", "/", regex=False)
        self.df["training_text"] = self.df["training_text"].astype(str).str.strip()
        self.df["filename"] = self.df["filename"].astype(str).str.strip()

        self.df = self.df[
            (self.df["image_path"] != "") &
            (self.df["training_text"] != "") &
            (self.df["filename"] != "")
        ].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"No usable rows found in dataset: {csv_path}")

        # Quick sample existence check
        missing_sample = sum(not Path(p).exists() for p in self.df["image_path"].head(20))
        print(f"{csv_path} | missing among first 20 image paths:", missing_sample)
        if missing_sample > 0:
            raise FileNotFoundError(f"Some sample image paths do not exist in {csv_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        text = str(row["training_text"])
        return image, text


# --------------------------------------------------
# 5) Create train and validation dataset objects
# --------------------------------------------------
train_ds = ImageTextDataset(TRAIN_CSV)
val_ds = ImageTextDataset(VAL_CSV)


# --------------------------------------------------
# 6) Load the CLIP processor and model
# --------------------------------------------------
processor = CLIPProcessor.from_pretrained(MODEL_NAME, use_fast=False)
model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)


# --------------------------------------------------
# 7) Keep the full model trainable
# --------------------------------------------------
trainable_names = []
trainable_params = []

for name, param in model.named_parameters():
    if param.requires_grad:
        trainable_names.append(name)
        trainable_params.append(param)


# --------------------------------------------------
# 8) Create the optimizer
# --------------------------------------------------
optimizer = AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)


# --------------------------------------------------
# 9) Define a collate function for batching
# --------------------------------------------------
def collate_fn(batch):
    images, texts = zip(*batch)

    enc = processor(
        text=list(texts),
        images=list(images),
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    return enc


# --------------------------------------------------
# 10) Create the train and validation DataLoaders
# --------------------------------------------------
train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)


# --------------------------------------------------
# 11) Print a short setup summary
# --------------------------------------------------
num_trainable_params = sum(p.numel() for p in trainable_params)

print("Device:", DEVICE)
print("Train rows:", len(train_ds))
print("Val rows:", len(val_ds))
print("Batch size:", BATCH_SIZE)
print("Trainable parameter tensors:", len(trainable_params))
print("Total trainable parameter count:", num_trainable_params)


# --------------------------------------------------
# 12) Prepare a list to store metrics
# --------------------------------------------------
metrics_rows = []


# --------------------------------------------------
# 13) Run the tiny training + validation loop
# --------------------------------------------------
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_losses = []

    for step, batch in enumerate(train_loader, start=1):
        if step > MAX_TRAIN_BATCHES:
            break

        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        optimizer.zero_grad()

        outputs = model(**batch, return_loss=True)
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

        train_losses.append(loss.item())
        print(f"Epoch {epoch} | Train step {step} | loss = {loss.item():.4f}")

    model.eval()
    val_losses = []

    with torch.no_grad():
        for step, batch in enumerate(val_loader, start=1):
            if step > MAX_VAL_BATCHES:
                break

            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            outputs = model(**batch, return_loss=True)
            loss = outputs.loss

            val_losses.append(loss.item())
            print(f"Epoch {epoch} | Val step {step} | loss = {loss.item():.4f}")

    avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else None
    avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else None

    metrics_rows.append({
        "epoch": epoch,
        "avg_train_loss": avg_train_loss,
        "avg_val_loss": avg_val_loss,
        "num_train_steps": len(train_losses),
        "num_val_steps": len(val_losses),
    })


# --------------------------------------------------
# 14) Save the metrics table
# --------------------------------------------------
metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(METRICS_CSV, index=False)


# --------------------------------------------------
# 15) Save the smoke-test checkpoint
# --------------------------------------------------
torch.save(model.state_dict(), CHECKPOINT_PATH)


# --------------------------------------------------
# 16) Print a final summary
# --------------------------------------------------
print("\nSaved metrics to:", METRICS_CSV)
print("Saved checkpoint to:", CHECKPOINT_PATH)
print("\nSmoke test summary:")
print(metrics_df.to_string(index=False))