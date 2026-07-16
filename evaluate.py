#!/usr/bin/env python3
"""Evaluate a trained checkpoint (EgoPER or EgoProceL)."""

import argparse
import json
from pathlib import Path

import torch

from .configs.utils import hiedict2cfg, setup_cfg
from .models.blocks import FACT
from .utils.dataset import DataLoader, create_dataset
from .utils.evaluate import Checkpoint
from .utils.train_tools import compute_null_weight, save_results


def resolve_device(requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def find_config(checkpoint, configured):
    if configured:
        return Path(configured)
    for name in ("config.yaml", "config.json", "args.json"):
        candidate = checkpoint.parent / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Pass --config or place config.yaml beside model.pth")


def load_config(path):
    if path.suffix in {".yaml", ".yml"}:
        return setup_cfg([str(path)], None)
    return hiedict2cfg(json.loads(path.read_text()))


@torch.no_grad()
def evaluate_model(model, loader, device):
    result = Checkpoint(
        -1, bg_class=[] if model.cfg.eval_bg else loader.dataset.bg_class
    )
    model.eval()
    for videos, sequences, labels, eval_labels in loader:
        sequences = [sequence.to(device) for sequence in sequences]
        labels = [label.to(device) for label in labels]
        saved = model(sequences, labels)
        save_results(result, videos, eval_labels, saved)
    result.compute_metrics()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config")
    parser.add_argument("--data-root")
    parser.add_argument("--gdag-root")
    parser.add_argument("--test-split")
    parser.add_argument("--error-video", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    cfg = load_config(find_config(checkpoint, args.config))
    cfg.defrost()
    if args.data_root:
        cfg.data_root = args.data_root
    if args.gdag_root:
        cfg.gdag_root = args.gdag_root
    if args.test_split:
        cfg.test_split = args.test_split
    cfg.error_vid = args.error_video
    cfg.aux.device = args.device
    cfg.freeze()

    device = resolve_device(cfg.aux.device)
    train_dataset, test_dataset = create_dataset(cfg)
    if cfg.Loss.nullw == -1:
        compute_null_weight(cfg, train_dataset, verbose=False)
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = FACT(cfg, train_dataset.input_dimension, train_dataset.nclasses)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device)

    result = evaluate_model(model, loader, device)
    text = json.dumps(result.metrics, indent=2, default=float)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n")


if __name__ == "__main__":
    main()
