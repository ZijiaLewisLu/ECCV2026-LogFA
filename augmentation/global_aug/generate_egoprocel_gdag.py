#!/usr/bin/env python3
"""Generate EgoProceL GDAG definitions through an OpenAI-compatible endpoint.

Mirrors ``augmentation/global_aug/generate_graphs.py`` (EgoPER): it prompts an
LLM to construct a ``{nodes, edges}`` GDAG (flat/hierarchical nodes, optional
``c1=0`` and repeated ``c2>1`` nodes) from a recipe's action list plus its
observed action sequences, and writes ``gdag/{recipe}.json`` for the loader's
``GDAGPlanner`` — the same format and sampler EgoPER uses.

The only difference from the EgoPER script: EgoProceL ships one shared
``mapping.txt`` across all recipes, so a recipe's action vocabulary is gathered
from that recipe's own groundTruth instead of the global mapping.

Usage (needs an OpenAI-compatible LLM endpoint):

    python -m src.augmentation.global_aug.generate_egoprocel_gdag \
      --data-root src/data/LogFA-EgoProceL-v1 \
      --base-url http://localhost:8000/v1 --api-key EMPTY
"""

import argparse
import json
import re
from pathlib import Path

from openai import OpenAI


DEFAULT_RECIPES = ("ETENT", "PC_assemble", "PC_disassemble", "MECANNO")
BACKGROUND = "background"


def read_videos(split_file):
    return [line for line in split_file.read_text().splitlines() if line.strip()]


def labels(annotations, video):
    return (annotations / "groundTruth" / f"{video}.txt").read_text().splitlines()


def action_sequence(frame_labels):
    result = []
    for label in frame_labels:
        label = label.split("|", 1)[0]
        if label != BACKGROUND and (not result or result[-1] != label):
            result.append(label)
    return result


def recipe_actions(annotations, videos):
    """Ordered unique non-background actions across the recipe's groundTruth."""
    seen, actions = set(), []
    for video in videos:
        for label in labels(annotations, video):
            label = label.split("|", 1)[0]
            if label != BACKGROUND and label not in seen:
                seen.add(label)
                actions.append(label)
    return actions


def prompt(actions, examples):
    action_text = "\n".join(
        f"{index}. {action}" for index, action in enumerate(actions, 1)
    )
    example_text = "\n".join(
        f"{index}. {'; '.join(sequence)}"
        for index, sequence in enumerate(examples, 1)
    )
    return f"""Construct a Generalized Directed Acyclic Graph (GDAG) for a
procedural task. The graph supports flat actions, hierarchical routines,
optional nodes (c1=0), and repeated nodes (c2>1).

For every flat node, copy the exact action string. Return JSON only:
{{
  "nodes": [
    {{"id": "exact action or routine", "type": "flat or hierarchical",
      "actions": [], "c1": 1, "c2": 1}}
  ],
  "edges": [["prerequisite", "successor"]]
}}

Actions:
{action_text}

Observed valid sequences:
{example_text}
"""


def parse_response(text):
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    return json.loads(fenced.group(1) if fenced else text)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", type=Path,
                        default=Path(__file__).resolve().parents[2]
                        / "data" / "LogFA-EgoProceL-v1")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: <data-root>/gdag")
    parser.add_argument("--recipes", nargs="*", default=list(DEFAULT_RECIPES))
    parser.add_argument("--split", default="split1")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")
    args = parser.parse_args()

    annotations = args.data_root / "annotations"
    output_dir = args.output_dir or (args.data_root / "gdag")
    output_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    for recipe in args.recipes:
        split_dir = annotations / "splits" / recipe
        train_videos = read_videos(split_dir / f"{args.split}.train")
        test_split = split_dir / f"{args.split}.test"
        all_videos = train_videos + (
            read_videos(test_split) if test_split.is_file() else []
        )
        actions = recipe_actions(annotations, all_videos)
        examples = [action_sequence(labels(annotations, v)) for v in train_videos]

        response = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": prompt(actions, examples)}],
        )
        graph = parse_response(response.choices[0].message.content)
        output = output_dir / f"{recipe}.json"
        output.write_text(json.dumps(graph, indent=2) + "\n")
        print(output)


if __name__ == "__main__":
    main()
