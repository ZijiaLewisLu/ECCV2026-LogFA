#!/usr/bin/env python3
"""Extract SigLIP features from generated image-editing outputs."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model", default="google/siglip2-large-patch16-256"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model)
    paths = sorted(args.images_dir.glob("*.png"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for start in tqdm(range(0, len(paths), args.batch_size)):
            batch = paths[start:start + args.batch_size]
            images = [Image.open(path).convert("RGB") for path in batch]
            inputs = processor(images=images, return_tensors="pt").to(device)
            features = model.get_image_features(**inputs).float().cpu().numpy()
            for path, feature in zip(batch, features):
                np.save(args.output_dir / f"{path.stem}.npy", feature)


if __name__ == "__main__":
    main()
