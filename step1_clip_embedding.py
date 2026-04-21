"""
Step 1 — Image -> Embedding (the core of retrieval)

What this script does:
1) Loads one image file: test.jpg
2) Uses a pretrained CLIP model to convert that image into an embedding vector
3) Prints the embedding shape and a few example values

Why we do this:
- In an image retrieval system, we need a way to compare images numerically.
- Comparing raw pixel values is not very useful, because two medically similar images
  may still have different brightness, size, or small visual differences.
- Instead, we use an embedding, which is a compact numeric representation of the image.
- Later in the project, we will:
    A) Generate embeddings for all dataset images
    B) Generate an embedding for a query image
    C) Retrieve the closest dataset images in embedding space

Important words:
- Pretrained model:
  A model already trained by others, so we can use it directly without training from scratch.
- CLIP:
  A vision-language model that can map images and text into a shared embedding space.
- Embedding / vector:
  A list of numbers representing the image in a compact way.
- Normalize:
  Scale the vector so its length becomes 1. This makes similarity comparison more stable.
"""

from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


# --------------------------------------------------
# 1) Choose the image file
# --------------------------------------------------
img_path = Path("test.jpg")

if not img_path.exists():
    raise FileNotFoundError(f"Image file not found: {img_path.resolve()}")


# --------------------------------------------------
# 2) Choose device
# --------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)


# --------------------------------------------------
# 3) Load a pretrained CLIP model
# --------------------------------------------------
model_name = "openai/clip-vit-base-patch32"

model = CLIPModel.from_pretrained(model_name).to(DEVICE)
model.eval()

processor = CLIPProcessor.from_pretrained(model_name, use_fast=False)


# --------------------------------------------------
# 4) Load the image from disk
# --------------------------------------------------
image = Image.open(img_path).convert("RGB")


# --------------------------------------------------
# 5) Convert the image into model input format
# --------------------------------------------------
inputs = processor(images=image, return_tensors="pt")
inputs = {k: v.to(DEVICE) for k, v in inputs.items()}


# --------------------------------------------------
# 6) Generate the image embedding
# --------------------------------------------------
with torch.no_grad():
    # Step 6a: pass the image through CLIP's vision encoder
    vision_outputs = model.vision_model(**inputs)

    # Step 6b: take the pooled image representation
    pooled_output = vision_outputs.pooler_output

    # Step 6c: apply CLIP's visual projection layer
    vec = model.visual_projection(pooled_output)

    # Step 6d: normalize the embedding
    vec = vec / vec.norm(dim=-1, keepdim=True)


# --------------------------------------------------
# 7) Print the results
# --------------------------------------------------
print("Embedding shape:", tuple(vec.shape))
print("First 8 values:", vec[0, :8].detach().cpu().tolist())