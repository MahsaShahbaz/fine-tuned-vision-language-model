"""
Step 16 — Run the first real end-to-end fine-tuning experiment
on the full train/validation splits - CPU-Friendly version

What this script does:
1) Loads the full train and validation split files
2) Builds image-text datasets and dataloaders
3) Loads a pretrained CLIP model and processor
4) Keeps the full CLIP model trainable (no freezing)
5) Runs a real end-to-end fine-tuning experiment
6) Evaluates the model on the validation split after each epoch
7) Saves:
   - training history
   - best checkpoint based on validation loss
   - last checkpoint from the final epoch

Why we do this:
- After the smoke test, we want to run a larger controlled training experiment
  on the real train and validation splits.
- The goal is to adapt the CLIP image-text representation to the Indiana
  chest X-ray domain.
- We then use the best checkpoint later to generate fine-tuned image embeddings
  and compare retrieval performance against the baseline.

What is different from the older version:
- The older version froze the heavy image and text encoders.
- That meant it was not true end-to-end fine-tuning.
- In this updated version, we do NOT freeze any part of CLIP.
- The whole model remains trainable.

Important note:
- Full end-to-end CLIP fine-tuning on the full dataset is much heavier
  than the earlier smoke test.
- Because of that, this script is designed to use CUDA/GPU.
- If GPU is not available, the script stops with a clear error message,
  because running this full experiment on CPU would be impractically slow.

Training design notes:
- We use a small batch size because the full CLIP model is large.
- We use gradient accumulation so that the effective batch size becomes larger
  without requiring too much GPU memory at once.
- We use a smaller learning rate than before because the whole model is trainable.
"""

from pathlib import Path
import time
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from torch.optim import AdamW


# --------------------------------------------------
# 1) Define input and output paths
# --------------------------------------------------
# We use the full train and validation split files created earlier.
TRAIN_CSV = Path("data/processed/finetuning/train.csv")
VAL_CSV = Path("data/processed/finetuning/val.csv")

# We save all outputs from this real training run in a separate folder.
#
# We use a shorter folder name here because long Windows paths caused
# saving issues in earlier steps.
OUTPUT_DIR = Path("results/ft16_real")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS_CSV = OUTPUT_DIR / "metrics.csv"
BEST_CKPT = OUTPUT_DIR / "best.pt"
LAST_CKPT = OUTPUT_DIR / "last.pt"


# --------------------------------------------------
# 2) Define the main training settings
# --------------------------------------------------
# MODEL_NAME:
# We use the same CLIP backbone as in the baseline pipeline
# so the comparison remains meaningful.
MODEL_NAME = "openai/clip-vit-base-patch32"

# DEVICE:
# This real full-model fine-tuning experiment is designed for GPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE != "cuda":
    raise RuntimeError(
        "Step 16 is configured as a real full end-to-end CLIP fine-tuning run. "
        "CUDA/GPU was not found. Running this full experiment on CPU would be too slow. "
        "Please run it on a GPU machine."
    )

# BATCH_SIZE:
# Keep this small because the whole CLIP model is trainable.
BATCH_SIZE = 2

# GRAD_ACCUM_STEPS:
# We accumulate gradients across several mini-batches before stepping the optimizer.
#
# Why this helps:
# - GPU memory may be too small for a large true batch size
# - gradient accumulation lets us simulate a larger effective batch size
#
# Effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS
GRAD_ACCUM_STEPS = 4

# EPOCHS:
# A first real controlled experiment.
EPOCHS = 3

# LR:
# Smaller than before because now the entire model is trainable.
LR = 1e-5

# WEIGHT_DECAY:
# Small regularization term to help reduce overfitting.
WEIGHT_DECAY = 1e-4

# PRINT_EVERY:
# How often to print progress inside train/validation loops.
PRINT_EVERY = 100

# MAX_GRAD_NORM:
# Used for gradient clipping to improve stability.
MAX_GRAD_NORM = 1.0


# --------------------------------------------------
# 3) Define a Dataset class for image-text pairs
# --------------------------------------------------
# Each training example in this project is:
# - one image
# - one report_text string
#
# This custom Dataset reads rows from a CSV and returns one (image, text) pair at a time.
class ImageTextDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path).copy()

    def __len__(self):
        # Total number of rows/examples in the dataset
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load the image from disk and convert to RGB for CLIP
        image = Image.open(row["image_path"]).convert("RGB")

        # Read the paired report text
        text = str(row["report_text"])

        return image, text


# --------------------------------------------------
# 4) Create dataset objects for train and validation
# --------------------------------------------------
train_ds = ImageTextDataset(TRAIN_CSV)
val_ds = ImageTextDataset(VAL_CSV)


# --------------------------------------------------
# 5) Load the CLIP processor and model
# --------------------------------------------------
# The processor handles both:
# - image preprocessing
# - text tokenization / padding / truncation
#
# The model is the main CLIP network.
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)


# --------------------------------------------------
# 6) Keep the full model trainable
# --------------------------------------------------
# This is the main difference from the old Step 16.
#
# We do NOT freeze:
# - the vision encoder
# - the text encoder
#
# So the whole CLIP model remains trainable.
trainable_names = []
trainable_params = []

for name, param in model.named_parameters():
    if param.requires_grad:
        trainable_names.append(name)
        trainable_params.append(param)


# --------------------------------------------------
# 7) Create the optimizer
# --------------------------------------------------
# AdamW is a standard optimizer for transformer-based models.
# It will update all trainable CLIP parameters.
optimizer = AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)


# --------------------------------------------------
# 8) Define a collate function for batching
# --------------------------------------------------
# A DataLoader groups multiple dataset samples into a batch.
# Since each sample here is a PIL image + text string,
# we use a custom collate function so the CLIP processor can convert
# the whole batch into model-ready tensors.
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
# 9) Create train and validation DataLoaders
# --------------------------------------------------
# train_loader:
# - shuffle=True so training order changes each epoch
#
# val_loader:
# - shuffle=False because validation does not need random order
#
# num_workers=0 keeps loading simpler and avoids Windows multiprocessing issues.
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
# 10) Print a short setup summary
# --------------------------------------------------
num_trainable_params = sum(p.numel() for p in trainable_params)

print("Device:", DEVICE)
print("Train rows:", len(train_ds))
print("Val rows:", len(val_ds))
print("Batch size:", BATCH_SIZE)
print("Gradient accumulation steps:", GRAD_ACCUM_STEPS)
print("Effective batch size:", BATCH_SIZE * GRAD_ACCUM_STEPS)
print("Epochs:", EPOCHS)
print("Learning rate:", LR)
print("Trainable parameter tensors:", len(trainable_params))
print("Total trainable parameter count:", num_trainable_params)


# --------------------------------------------------
# 11) Prepare training history tracking
# --------------------------------------------------
# history will store one row per epoch.
# best_val_loss keeps track of the best validation result seen so far.
history = []
best_val_loss = float("inf")


# --------------------------------------------------
# 12) Run the full training loop
# --------------------------------------------------
for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()

    # ------------------------------------------
    # 12a) Training phase
    # ------------------------------------------
    model.train()
    train_losses = []

    # Clear gradients before starting the epoch.
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader, start=1):
        # Move batch tensors to the selected device.
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        # CLIP computes its own contrastive loss when return_loss=True.
        outputs = model(**batch, return_loss=True)
        raw_loss = outputs.loss

        # With gradient accumulation, we divide the loss by the number
        # of accumulation steps before backpropagation.
        loss = raw_loss / GRAD_ACCUM_STEPS

        loss.backward()

        # We store the original step loss for logging.
        train_losses.append(raw_loss.item())

        # Only update parameters after enough accumulation steps,
        # or at the final step of the epoch.
        if (step % GRAD_ACCUM_STEPS == 0) or (step == len(train_loader)):
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=MAX_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad()

        # Print progress every PRINT_EVERY steps
        if step % PRINT_EVERY == 0:
            avg_so_far = sum(train_losses) / len(train_losses)
            print(
                f"Epoch {epoch} | Train step {step}/{len(train_loader)} "
                f"| step_loss={raw_loss.item():.4f} | avg_train_so_far={avg_so_far:.4f}"
            )

    # Average training loss for this epoch
    avg_train_loss = sum(train_losses) / len(train_losses)

    # ------------------------------------------
    # 12b) Validation phase
    # ------------------------------------------
    # In validation, we do not update model parameters.
    model.eval()
    val_losses = []

    with torch.no_grad():
        for step, batch in enumerate(val_loader, start=1):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            outputs = model(**batch, return_loss=True)
            loss = outputs.loss
            val_losses.append(loss.item())

            if step % PRINT_EVERY == 0:
                avg_val_so_far = sum(val_losses) / len(val_losses)
                print(
                    f"Epoch {epoch} | Val step {step}/{len(val_loader)} "
                    f"| step_loss={loss.item():.4f} | avg_val_so_far={avg_val_so_far:.4f}"
                )

    # Average validation loss for this epoch
    avg_val_loss = sum(val_losses) / len(val_losses)

    # Measure how long the epoch took
    epoch_minutes = (time.time() - epoch_start) / 60.0

    # Save this epoch's summary into history
    row = {
        "epoch": epoch,
        "avg_train_loss": avg_train_loss,
        "avg_val_loss": avg_val_loss,
        "train_steps": len(train_losses),
        "val_steps": len(val_losses),
        "epoch_minutes": epoch_minutes,
    }
    history.append(row)

    # Print a clear end-of-epoch summary
    print("\n" + "=" * 80)
    print(
        f"Epoch {epoch} done | "
        f"avg_train_loss={avg_train_loss:.4f} | "
        f"avg_val_loss={avg_val_loss:.4f} | "
        f"time={epoch_minutes:.2f} min"
    )
    print("=" * 80 + "\n")

    # ------------------------------------------
    # 12c) Save the best checkpoint
    # ------------------------------------------
    # We use validation loss to decide which checkpoint is best.
    # Lower validation loss is better.
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), BEST_CKPT)
        print(f"New best model saved to: {BEST_CKPT}")

    # ------------------------------------------
    # 12d) Save the last checkpoint and metrics
    # ------------------------------------------
    # We save the last checkpoint every epoch, even if it is not the best one.
    torch.save(model.state_dict(), LAST_CKPT)

    # We also save the training history after each epoch
    # so progress is not lost if a later issue happens.
    pd.DataFrame(history).to_csv(METRICS_CSV, index=False)


# --------------------------------------------------
# 13) Print the final summary
# --------------------------------------------------
print("\nTraining finished.")
print("Saved metrics to:", METRICS_CSV)
print("Saved best model to:", BEST_CKPT)
print("Saved last model to:", LAST_CKPT)

print("\nFinal training history:")
print(pd.DataFrame(history).to_string(index=False))