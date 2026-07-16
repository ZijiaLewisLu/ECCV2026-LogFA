#!/usr/bin/env python3
"""Generate edited frames with Qwen-Image-Edit from a JSONL manifest."""

import argparse
import json
from pathlib import Path

import torch
from diffusers import QwenImageEditPipeline
from PIL import Image
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen-Image-Edit")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    items = [
        json.loads(line)
        for line in args.manifest.read_text().splitlines()
        if line.strip()
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = QwenImageEditPipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    )
    pipeline.enable_model_cpu_offload()

    for item in tqdm(items):
        image = Image.open(item["image_path"]).convert("RGB")
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        result = pipeline(
            image=image,
            prompt=item["prompt"],
            negative_prompt=" ",
            true_cfg_scale=4.0,
            num_inference_steps=args.steps,
            generator=generator,
        ).images[0]
        output = args.output_dir / f"{item['id']}.png"
        result.save(output)


if __name__ == "__main__":
    main()
