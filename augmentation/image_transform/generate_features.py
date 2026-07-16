#!/usr/bin/env python3
"""Generate SigLIP features from randomly transformed EgoPER frames."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor


def transform():
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ]
    )


@torch.no_grad()
def process_video(
    video_dir,
    output_dir,
    model,
    processor,
    device,
    variants,
    batch_size,
):
    frames = sorted(video_dir.glob("*.jpg"))
    output_dir.mkdir(parents=True, exist_ok=True)
    augmentation = transform()

    for variant in range(variants):
        output = output_dir / f"{video_dir.name}_aug{variant:02d}.npy"
        features = []
        for start in tqdm(
            range(0, len(frames), batch_size),
            desc=f"{video_dir.name}:{variant:02d}",
        ):
            images = [
                augmentation(Image.open(path).convert("RGB"))
                for path in frames[start:start + batch_size]
            ]
            inputs = processor(images=images, return_tensors="pt").to(device)
            encoded = model.get_image_features(**inputs)
            features.append(encoded.float().cpu().numpy())
        np.save(output, np.concatenate(features, axis=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--model", default="google/siglip2-large-patch16-256"
    )
    parser.add_argument("--variants", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model)

    for recipe_dir in sorted(path for path in args.frames_root.iterdir() if path.is_dir()):
        for video_dir in sorted(path for path in recipe_dir.iterdir() if path.is_dir()):
            process_video(
                video_dir,
                args.output_root / recipe_dir.name,
                model,
                processor,
                device,
                args.variants,
                args.batch_size,
            )


if __name__ == "__main__":
    main()
