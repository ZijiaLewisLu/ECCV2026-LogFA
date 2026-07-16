#!/usr/bin/env python3
"""Summarize frame captions and generate four families of rewrites."""

import argparse
import asyncio
import json
from pathlib import Path

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm


SUMMARY_PROMPT = """Summarize this cooking-video frame caption into one short
sentence that preserves the visible action and objects. Return only the
sentence.

Caption: {caption}"""

REWRITE_PROMPT = """Create caption augmentation for the frame caption below.
Return JSON only with exactly these keys:
{{
  "Method1": ["12 meaning-preserving rewrites"],
  "Method2": ["12 rewrites changing object colors"],
  "Method3": ["12 rewrites changing object material or texture"],
  "Method4": ["12 rewrites adding reasonable unimportant context"]
}}
Keep every caption short and preserve the action.

Caption: {caption}"""


def parse_json(text):
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"Model did not return JSON: {text[:120]}")
    return json.loads(text[start:end + 1])


async def complete(client, model, prompt, semaphore):
    async with semaphore:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3072,
        )
    return response.choices[0].message.content


async def process_file(path, output, client, model, concurrency):
    items = [
        json.loads(line) for line in path.read_text().splitlines() if line
    ]
    semaphore = asyncio.Semaphore(concurrency)
    summaries = await tqdm.gather(
        *[
            complete(
                client,
                model,
                SUMMARY_PROMPT.format(caption=item["caption"]),
                semaphore,
            )
            for item in items
        ],
        desc=f"summarize:{path.stem}",
    )
    rewrites = await tqdm.gather(
        *[
            complete(
                client,
                model,
                REWRITE_PROMPT.format(caption=summary),
                semaphore,
            )
            for summary in summaries
        ],
        desc=f"rewrite:{path.stem}",
    )
    for item, summary, rewrite in zip(items, summaries, rewrites):
        item["caption"] = summary.strip()
        item["rewrite"] = parse_json(rewrite)
    output.write_text(
        "".join(json.dumps(item) + "\n" for item in items)
    )


async def run(args):
    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.input_dir.glob("*.jsonl")):
        output = args.output_dir / f"{path.stem}_rewritten.jsonl"
        await process_file(
            path, output, client, args.model, args.concurrency
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument(
        "--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
    )
    parser.add_argument("--concurrency", type=int, default=32)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
