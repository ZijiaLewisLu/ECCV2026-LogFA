#!/usr/bin/env python3
"""Generate EgoPER GDAG definitions through an OpenAI-compatible endpoint."""

import argparse
import json
import re
from pathlib import Path

from openai import OpenAI


RECIPES = ("coffee", "oatmeal", "pinwheels", "quesadilla", "tea")
TRAIN_SPLITS = {
    "coffee": "a2_3v",
    "quesadilla": "a2_3v",
    "oatmeal": "a1_3v",
    "pinwheels": "a1_3v",
    "tea": "a1_3v",
}


def action_sequence(labels):
    result = []
    for label in labels:
        label = label.split("|", 1)[0]
        if label != "BG" and (not result or result[-1] != label):
            result.append(label)
    return result


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "data"
        / "LogFA-EgoPER-v1"
        / "gdag",
    )
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument(
        "--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
    )
    args = parser.parse_args()

    annotations = args.data_root / "annotations"
    actions = []
    for line in (annotations / "mapping.txt").read_text().splitlines():
        if line.strip():
            _, action = line.split(maxsplit=1)
            if action != "BG":
                actions.append(action)

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for recipe in RECIPES:
        split = (
            annotations
            / "splits"
            / recipe
            / f"{TRAIN_SPLITS[recipe]}.train"
        )
        videos = [line for line in split.read_text().splitlines() if line]
        examples = [
            action_sequence(
                (
                    annotations / "groundTruth" / f"{video}.txt"
                ).read_text().splitlines()
            )
            for video in videos
        ]
        response = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": prompt(actions, examples)}],
        )
        graph = parse_response(response.choices[0].message.content)
        output = args.output_dir / f"{recipe}.json"
        output.write_text(json.dumps(graph, indent=2) + "\n")
        print(output)


if __name__ == "__main__":
    main()
