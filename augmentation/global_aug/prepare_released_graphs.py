#!/usr/bin/env python3
"""Convert generated GDAG responses into compact training assets."""

import argparse
import json
import re
from pathlib import Path


RECIPES = ("coffee", "oatmeal", "pinwheels", "quesadilla", "tea")


def extract_graph(source: Path) -> dict:
    data = json.loads(source.read_text())
    if "nodes" in data and "edges" in data:
        return data

    response = data["response"]
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", response, re.DOTALL)
    return json.loads(fence.group(1) if fence else response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "data"
        / "LogFA-EgoPER-v1"
        / "gdag",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for recipe in RECIPES:
        source = (
            args.input_dir
            / f"{recipe}_task_prompt_v2_egoper_qwen3_v2.json"
        )
        graph = extract_graph(source)
        output = args.output_dir / f"{recipe}.json"
        output.write_text(json.dumps(graph, indent=2) + "\n")
        print(output)


if __name__ == "__main__":
    main()
