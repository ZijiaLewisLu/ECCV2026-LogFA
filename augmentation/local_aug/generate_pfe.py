#!/usr/bin/env python3
"""Generate local PFE arrays from rewritten captions and base SigLIP features."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor


class PromptTextEncoder(torch.nn.Module):
    def __init__(self, model_name, prompt_length):
        super().__init__()
        reference = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.encoder = reference.text_model
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        hidden = self.encoder.embeddings.token_embedding.weight.shape[1]
        self.prompt = torch.nn.Parameter(
            torch.zeros(1, prompt_length, hidden, dtype=torch.float32)
        )

    def tokenize(self, captions, device):
        return self.processor(
            text=captions,
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        ).to(device)

    def forward(self, captions, device, use_prompt):
        input_ids = self.tokenize(captions, device).input_ids
        hidden = self.encoder.embeddings(input_ids=input_ids, position_ids=None)
        if use_prompt:
            prompt = self.prompt.to(hidden.dtype).expand(len(captions), -1, -1)
            hidden = torch.cat((prompt, hidden), dim=1)
        encoded = self.encoder.encoder(inputs_embeds=hidden)
        states = self.encoder.final_layer_norm(encoded.last_hidden_state)
        return self.encoder.head(states[:, -1, :])


def captions_for_frame(item, naug, rewrite_cap=4):
    captions = [item["caption"], *item["rewrite"]["Method1"]][:naug]
    primary_count = len(captions)
    for method in ("Method2", "Method3", "Method4"):
        # rewrite_cap replicates the original generator's rewrite_cap=4: keep only
        # the first N rewrites of each secondary method (K = naug + 3*rewrite_cap).
        captions.extend(item["rewrite"][method][:rewrite_cap])
    return captions, primary_count


def generate_frame_pfe(
    model,
    captions,
    primary_count,
    image_feature,
    loss_weight,
    learning_rate,
    max_iterations,
    device,
):
    model.prompt.data.normal_()
    optimizer = torch.optim.AdamW([model.prompt], lr=learning_rate)

    with torch.no_grad():
        reference = model(captions, device, use_prompt=False).float()
        reference = torch.nn.functional.normalize(reference, dim=-1)
    image = torch.nn.functional.normalize(
        image_feature.to(device).float(), dim=-1
    )

    for _ in range(max_iterations):
        predicted = model(captions, device, use_prompt=True).float()
        predicted = torch.nn.functional.normalize(predicted, dim=-1)

        original_loss = 1 - (
            image @ predicted[:primary_count].T
        ).mean()
        rewritten_loss = 1 - (
            image @ predicted[primary_count:].T
        ).mean()
        contrastive = torch.nn.functional.cross_entropy(
            predicted @ reference.T,
            torch.arange(len(predicted), device=device),
        )

        image_loss = (
            loss_weight[0] * original_loss
            + (1 - loss_weight[0]) * rewritten_loss
        )
        loss = loss_weight[1] * image_loss + (
            1 - loss_weight[1]
        ) * contrastive
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if original_loss.item() < 0.2:
            break

    return predicted.detach().float().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--captions-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--model", default="google/siglip2-large-patch16-256"
    )
    parser.add_argument("--prompt-length", type=int, default=3)
    parser.add_argument("--naug", type=int, default=4)
    parser.add_argument("--rewrite-cap", type=int, default=4,
                        help="Keep first N rewrites of each Method2/3/4 (original used 4).")
    parser.add_argument("--tprompt-idx", type=int, default=0,
                        help="Prompt-config index; output file is {video}_{tprompt_idx}.npy.")
    parser.add_argument("--seed", type=int, default=-1,
                        help="If >=0, seed torch/numpy for deterministic prompt init.")
    parser.add_argument("--loss-weight-image", type=float, default=0.7)
    parser.add_argument("--loss-weight-total", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--max-iterations", type=int, default=800)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    if args.seed >= 0:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    output_dir = args.output_dir or (
        args.data_root / "features" / "local_pfe_sweep" / args.recipe
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    base_dir = args.data_root / "features" / "siglip2"
    split_dir = args.data_root / "annotations" / "splits" / args.recipe
    videos = set()
    for split in split_dir.glob("*.train"):
        videos.update(line for line in split.read_text().splitlines() if line)

    device = torch.device(f"cuda:{args.gpu}")
    model = PromptTextEncoder(args.model, args.prompt_length).to(device)
    model.train()

    for video in sorted(videos):
        items = [
            json.loads(line)
            for line in (
                args.captions_dir / f"{video}_rewritten.jsonl"
            ).read_text().splitlines()
            if line
        ]
        image_features = torch.from_numpy(
            np.load(base_dir / f"{video}.npy")[::10]
        )
        length = min(len(items), len(image_features))
        frame_features = []
        for index in tqdm(range(length), desc=video):
            captions, primary_count = captions_for_frame(
                items[index], args.naug, args.rewrite_cap
            )
            frame_features.append(
                generate_frame_pfe(
                    model,
                    captions,
                    primary_count,
                    image_features[index],
                    (
                        args.loss_weight_image,
                        args.loss_weight_total,
                    ),
                    args.learning_rate,
                    args.max_iterations,
                    device,
                )
            )
        np.save(
            output_dir / f"{video}_{args.tprompt_idx}.npy",
            np.stack(frame_features).astype(np.float32),
        )


if __name__ == "__main__":
    main()
