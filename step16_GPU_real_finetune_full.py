"""
Step 16 — Run the first real end-to-end fine-tuning experiment
on the full train/validation splits - GPU Setup 

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
import random
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from torch.optim import AdamW


# --------------------------------------------------
# 1) Define input and output paths
# --------------------------------------------------
TRAIN_CSV = Path("data/processed/finetuning/train.csv")
VAL_CSV = Path("data/processed/finetuning/val.csv")

OUTPUT_DIR = Path("results/ft16_real")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS_CSV = OUTPUT_DIR / "metrics.csv"
BEST_CKPT = OUTPUT_DIR / "best.pt"
LAST_CKPT = OUTPUT_DIR / "last.pt"


# --------------------------------------------------
# 2) Define the main training settings
# --------------------------------------------------
MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE != "cuda":
    raise RuntimeError(
        "Step 16 is configured as a real full end-to-end CLIP fine-tuning run. "
        "CUDA/GPU was not found. Running this full experiment on CPU would be too slow. "
        "Please run it on a GPU machine."
    )

BATCH_SIZE = 2

# We increase gradient accumulation so the effective batch size becomes larger.
# Effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS = 16
GRAD_ACCUM_STEPS = 8

# We allow more epochs, but early stopping below can stop training earlier
# if validation loss stops improving.
EPOCHS = 10

# We reduce the learning rate compared with the earlier run,
# because the whole CLIP model is trainable and we want a gentler update.
LR = 5e-6

WEIGHT_DECAY = 1e-4
PRINT_EVERY = 100
MAX_GRAD_NORM = 1.0
SEED = 42

# EARLY STOPPING SETTINGS:
# - PATIENCE: how many epochs without meaningful validation improvement we allow
# - MIN_DELTA: the minimum drop in validation loss required to count as improvement
EARLY_STOPPING_PATIENCE = 2
EARLY_STOPPING_MIN_DELTA = 1e-4


# --------------------------------------------------
# 3) Reproducibility and CUDA settings
# --------------------------------------------------
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Helpful on A100 / modern NVIDIA GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# --------------------------------------------------
# 4) Check input files exist
# --------------------------------------------------
for p in [TRAIN_CSV, VAL_CSV]:
    if not p.exists():
        raise FileNotFoundError(f"Required file not found: {p.resolve()}")


# --------------------------------------------------
# 5) Define a Dataset class for image-text pairs
# --------------------------------------------------
class ImageTextDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path).copy()

        # In the updated pipeline, training uses finetune_text instead of the older
        # full report_text field.
        required_cols = ["filename", "image_path", "finetune_text"]
        missing_cols = [c for c in required_cols if c not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in {csv_path}: {missing_cols}")

        self.df["filename"] = self.df["filename"].astype(str).str.strip()
        self.df["image_path"] = self.df["image_path"].astype(str).str.replace("\\", "/", regex=False)
        self.df["finetune_text"] = self.df["finetune_text"].astype(str).str.strip()

        self.df = self.df[
            (self.df["filename"] != "") &
            (self.df["image_path"] != "") &
            (self.df["finetune_text"] != "")
        ].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"No usable rows found in dataset: {csv_path}")

        missing_sample = sum(not Path(p).exists() for p in self.df["image_path"].head(20))
        print(f"{csv_path} | missing among first 20 image paths:", missing_sample)
        if missing_sample > 0:
            raise FileNotFoundError(f"Some sample image paths do not exist in {csv_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")

        # Use the updated shorter cleaned fine-tuning text target.
        text = str(row["finetune_text"])

        return image, text


# --------------------------------------------------
# 6) Create dataset objects for train and validation
# --------------------------------------------------
train_ds = ImageTextDataset(TRAIN_CSV)
val_ds = ImageTextDataset(VAL_CSV)


# --------------------------------------------------
# 7) Load the CLIP processor and model
# --------------------------------------------------
processor = CLIPProcessor.from_pretrained(MODEL_NAME, use_fast=False)
model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)


# --------------------------------------------------
# 8) Keep the full model trainable
# --------------------------------------------------
trainable_names = []
trainable_params = []

for name, param in model.named_parameters():
    if param.requires_grad:
        trainable_names.append(name)
        trainable_params.append(param)


# --------------------------------------------------
# 9) Create the optimizer
# --------------------------------------------------
optimizer = AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)


# --------------------------------------------------
# 10) Define a collate function for batching
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
# 11) Create train and validation DataLoaders
# --------------------------------------------------
train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    collate_fn=collate_fn
)

if len(train_loader) == 0 or len(val_loader) == 0:
    raise ValueError("Train loader or validation loader is empty.")


# --------------------------------------------------
# 12) Print a short setup summary
# --------------------------------------------------
num_trainable_params = sum(p.numel() for p in trainable_params)

print("Device:", DEVICE)
print("Train rows:", len(train_ds))
print("Val rows:", len(val_ds))
print("Train batches:", len(train_loader))
print("Val batches:", len(val_loader))
print("Batch size:", BATCH_SIZE)
print("Gradient accumulation steps:", GRAD_ACCUM_STEPS)
print("Effective batch size:", BATCH_SIZE * GRAD_ACCUM_STEPS)
print("Epochs:", EPOCHS)
print("Learning rate:", LR)
print("Weight decay:", WEIGHT_DECAY)
print("Early stopping patience:", EARLY_STOPPING_PATIENCE)
print("Early stopping min delta:", EARLY_STOPPING_MIN_DELTA)
print("Trainable parameter tensors:", len(trainable_params))
print("Total trainable parameter count:", num_trainable_params)
print("Text field used:", "finetune_text")
print("Metrics file:", METRICS_CSV)
print("Best checkpoint path:", BEST_CKPT)
print("Last checkpoint path:", LAST_CKPT)


# --------------------------------------------------
# 13) Prepare training history tracking
# --------------------------------------------------
history = []
best_val_loss = float("inf")
epochs_without_improvement = 0


# --------------------------------------------------
# 14) Run the full training loop
# --------------------------------------------------
for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()

    # ------------------------------------------
    # 14a) Training phase
    # ------------------------------------------
    model.train()
    train_losses = []

    optimizer.zero_grad()

    for step, batch in enumerate(train_loader, start=1):
        batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}

        outputs = model(**batch, return_loss=True)
        raw_loss = outputs.loss

        # With gradient accumulation, divide the loss before backpropagation.
        loss = raw_loss / GRAD_ACCUM_STEPS

        loss.backward()
        train_losses.append(raw_loss.item())

        if (step % GRAD_ACCUM_STEPS == 0) or (step == len(train_loader)):
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=MAX_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad()

        if step % PRINT_EVERY == 0:
            avg_so_far = sum(train_losses) / len(train_losses)
            print(
                f"Epoch {epoch} | Train step {step}/{len(train_loader)} "
                f"| step_loss={raw_loss.item():.4f} | avg_train_so_far={avg_so_far:.4f}"
            )

    avg_train_loss = sum(train_losses) / len(train_losses)

    # ------------------------------------------
    # 14b) Validation phase
    # ------------------------------------------
    model.eval()
    val_losses = []

    with torch.no_grad():
        for step, batch in enumerate(val_loader, start=1):
            batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}

            outputs = model(**batch, return_loss=True)
            loss = outputs.loss
            val_losses.append(loss.item())

            if step % PRINT_EVERY == 0:
                avg_val_so_far = sum(val_losses) / len(val_losses)
                print(
                    f"Epoch {epoch} | Val step {step}/{len(val_loader)} "
                    f"| step_loss={loss.item():.4f} | avg_val_so_far={avg_val_so_far:.4f}"
                )

    avg_val_loss = sum(val_losses) / len(val_losses)
    epoch_minutes = (time.time() - epoch_start) / 60.0

    row = {
        "epoch": epoch,
        "avg_train_loss": avg_train_loss,
        "avg_val_loss": avg_val_loss,
        "train_steps": len(train_losses),
        "val_steps": len(val_losses),
        "epoch_minutes": epoch_minutes,
        "best_val_loss_so_far": None,
        "epochs_without_improvement": None,
    }
    history.append(row)

    print("\n" + "=" * 80)
    print(
        f"Epoch {epoch} done | "
        f"avg_train_loss={avg_train_loss:.4f} | "
        f"avg_val_loss={avg_val_loss:.4f} | "
        f"time={epoch_minutes:.2f} min"
    )
    print("=" * 80 + "\n")

    # ------------------------------------------
    # 14c) Save the best checkpoint
    # ------------------------------------------
    improvement = best_val_loss - avg_val_loss

    if improvement > EARLY_STOPPING_MIN_DELTA:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), BEST_CKPT)
        print(f"New best model saved to: {BEST_CKPT}")
    else:
        epochs_without_improvement += 1
        print(
            f"No meaningful validation improvement this epoch. "
            f"Patience counter: {epochs_without_improvement}/{EARLY_STOPPING_PATIENCE}"
        )

    # Update the stored history row with the latest tracking values.
    history[-1]["best_val_loss_so_far"] = best_val_loss
    history[-1]["epochs_without_improvement"] = epochs_without_improvement

    # ------------------------------------------
    # 14d) Save the last checkpoint and metrics
    # ------------------------------------------
    torch.save(model.state_dict(), LAST_CKPT)
    pd.DataFrame(history).to_csv(METRICS_CSV, index=False)

    # ------------------------------------------
    # 14e) Early stopping check
    # ------------------------------------------
    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        print("\nEarly stopping triggered.")
        break


# --------------------------------------------------
# 15) Print the final summary
# --------------------------------------------------
print("\nTraining finished.")
print("Saved metrics to:", METRICS_CSV)
print("Saved best model to:", BEST_CKPT)
print("Saved last model to:", LAST_CKPT)

print("\nFinal training history:")
print(pd.DataFrame(history).to_string(index=False))