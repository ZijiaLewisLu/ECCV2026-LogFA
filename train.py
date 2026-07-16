#!/usr/bin/env python3
"""Train Local Aug, Global Aug, or LogFA."""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
import wandb
from torch import optim

from .configs.utils import cfg2flatdict, setup_cfg
from .home import get_project_base
from .models.blocks import FACT
from .models.loss import MatchCriterion
from .utils.dataset import DataLoader, create_dataset
from .utils.evaluate import Checkpoint
from .utils.utils import file_sha256
from .utils.train_tools import (
    compute_null_weight,
    resume_ckpt,
    resume_wandb_runid,
    save_results,
)
from .utils.utils import count_parameters


# Per-dataset registry: task list, config directory/files, and the canonical
# (train_split, test_split) for each task. `mode` configs (local_aug/global_aug/
# logfa.yaml) live in each dataset's config dir. See src/data/ for the data
# packages consumed by each dataset.
DATASETS = {
    "egoper": {
        "config_dir": "egoper",
        "fact_config": "FACT_EgoPER.yaml",
        "task_config": "EgoPER_{task}.yaml",
        "task_splits": {
            "coffee": ("a2_3v", "a2_28v"),
            "oatmeal": ("a1_3v", "a1_29v"),
            "pinwheels": ("a1_3v", "a1_31v"),
            "quesadilla": ("a2_3v", "a2_39v"),
            "tea": ("a1_3v", "a1_40v"),
        },
    },
    "egoprocel": {
        "config_dir": "egoprocel",
        "fact_config": "FACT_EgoProceL.yaml",
        "task_config": "EgoProceL_{task}.yaml",
        "task_splits": {
            "ETENT": ("split1", "split1"),
            "PC_assemble": ("split1", "split1"),
            "PC_disassemble": ("split1", "split1"),
            "MECANNO": ("split1", "split1"),
        },
    },
}

# All selectable tasks across datasets (argparse choices; validated per dataset).
ALL_TASKS = tuple(
    task for spec in DATASETS.values() for task in spec["task_splits"]
)


def resolve_device(requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


@torch.no_grad()
def evaluate(global_step, model, loader, device, run, save_dir):
    result = Checkpoint(
        global_step + 1,
        bg_class=[] if model.cfg.eval_bg else loader.dataset.bg_class,
    )
    model.eval()
    for videos, sequences, labels, eval_labels in loader:
        sequences = [sequence.to(device) for sequence in sequences]
        labels = [label.to(device) for label in labels]
        saved = model(sequences, labels)
        save_results(result, videos, eval_labels, saved)
    result.compute_metrics()
    model.train()

    if run is not None:
        run.log(
            {f"test-metric/{key}": value for key, value in result.metrics.items()},
            step=global_step + 1,
        )
    result.save(save_dir / f"{global_step + 1}.gz")
    (save_dir / f"{global_step + 1}.json").write_text(
        json.dumps(result.metrics, indent=2) + "\n"
    )
    return result


def build_config(args):
    config_files = list(args.cfg)
    overrides = list(args.set or [])
    if args.mode:
        if not args.recipe:
            raise ValueError("--recipe is required when --mode is used")
        spec = DATASETS[args.dataset]
        if args.recipe not in spec["task_splits"]:
            raise ValueError(
                f"task {args.recipe!r} is not a {args.dataset} task; "
                f"choices: {sorted(spec['task_splits'])}"
            )
        config_root = Path("src") / "configs" / spec["config_dir"]
        config_files = [
            str(config_root / spec["task_config"].format(task=args.recipe)),
            str(config_root / spec["fact_config"]),
            str(config_root / f"{args.mode}.yaml"),
        ]
        default_train, default_test = spec["task_splits"][args.recipe]
        overrides = [
            "recipe", args.recipe,
            "split", args.train_split or default_train,
            "test_split", args.test_split or default_test,
            "max_iter", str(args.max_iter),
            "epoch", "None",
            "aux.eval_every", str(args.eval_every),
            "aux.print_every", str(args.print_every),
            "aux.runid", str(args.runid),
            *overrides,
        ]
    cfg = setup_cfg(config_files, overrides, logdir="log/")
    cfg.defrost()
    if args.data_root:
        cfg.data_root = args.data_root
    if args.gdag_root:
        cfg.gdag_root = args.gdag_root
    cfg.aux.device = args.device
    cfg.freeze()
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("local_aug", "global_aug", "logfa")
    )
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="egoper")
    parser.add_argument("--recipe", choices=ALL_TASKS)
    parser.add_argument("--train-split")
    parser.add_argument("--test-split")
    parser.add_argument("--data-root")
    parser.add_argument("--gdag-root")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--runid", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--print-every", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--export-checkpoint",
        help="Write the final model state dict to this path.",
    )
    parser.add_argument("--cfg", nargs="*", default=[])
    parser.add_argument("--set", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = build_config(args)
    if args.dry_run:
        print(cfg.dump())
        return

    device = resolve_device(cfg.aux.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)
    print(f"Device: {device}")
    print(cfg)

    if cfg.aux.debug:
        np.random.seed(1)
        torch.manual_seed(1)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True

    output_root = Path(get_project_base())
    log_dir = output_root / cfg.aux.logdir
    checkpoint_dir = log_dir / "ckpts"
    save_dir = log_dir / "saves"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, test_dataset = create_dataset(cfg)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False
    )
    if cfg.Loss.nullw == -1:
        compute_null_weight(cfg, train_dataset)

    model = FACT(cfg, train_dataset.input_dimension, train_dataset.nclasses)
    model.mcriterion = MatchCriterion(
        cfg, train_dataset.nclasses, train_dataset.bg_class
    )
    model.to(device)
    print(model)
    print(f"Trainable parameters: {count_parameters(model) / 1e6:.2f}M")

    global_step, checkpoint = resume_ckpt(cfg, str(log_dir))
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state, strict=True)

    config_path = log_dir / "args.json"
    config_path.write_text(json.dumps(cfg, indent=2) + "\n")

    offline = (
        cfg.aux.wandb_offline
        or cfg.aux.debug
        or os.environ.get("WANDB_MODE", "").lower()
        in {"offline", "disabled", "dryrun"}
    )
    try:
        run = wandb.init(
            project=cfg.aux.wandb_project,
            entity=cfg.aux.wandb_entity,
            dir=str(log_dir),
            group=cfg.aux.exp,
            id=resume_wandb_runid(str(log_dir)),
            resume="allow",
            config=cfg2flatdict(cfg),
            reinit=True,
            save_code=False,
            mode="offline" if offline else "online",
        )
    except Exception:
        logging.exception("Wandb initialization failed; continuing without it")
        run = None

    if cfg.optimizer == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    else:
        optimizer = optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

    max_iter = cfg.max_iter
    if max_iter is None:
        max_iter = cfg.epoch * len(train_loader)
    train_result = Checkpoint(-1, bg_class=train_dataset.bg_class, eval_edit=False)
    model.train()
    while global_step < max_iter:
        for videos, sequences, labels, eval_labels in train_loader:
            sequences = [sequence.to(device) for sequence in sequences]
            labels = [label.to(device) for label in labels]
            optimizer.zero_grad()
            loss, saved = model(sequences, labels, compute_loss=True)
            loss.backward()
            if cfg.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.clip_grad_norm
                )
            optimizer.step()
            save_results(train_result, videos, eval_labels, saved)
            global_step += 1

            if global_step % cfg.aux.print_every == 0:
                train_result.average_losses()
                print(f"Iter {global_step}: loss={train_result.loss['loss']:.4f}")
                if run is not None:
                    run.log(
                        {"train-loss/loss": train_result.loss["loss"]},
                        step=global_step,
                    )
                train_result = Checkpoint(
                    -1, bg_class=train_dataset.bg_class, eval_edit=False
                )

            if global_step % cfg.aux.eval_every == 0:
                evaluate(global_step - 1, model, test_loader, device, run, save_dir)
                model.save_model(
                    checkpoint_dir / f"network.iter-{global_step}.net"
                )

            if global_step >= max_iter:
                break

    (log_dir / "FINISH_PROOF").touch()
    if args.export_checkpoint:
        export_path = Path(args.export_checkpoint)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(export_path)
        metadata_path = export_path.parent / "metadata.json"
        metadata = (
            json.loads(metadata_path.read_text())
            if metadata_path.is_file()
            else {}
        )
        metadata.update(
            {
                "status": "complete",
                "checkpoint": export_path.name,
                "size_bytes": export_path.stat().st_size,
                "sha256": file_sha256(export_path),
            }
        )
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
