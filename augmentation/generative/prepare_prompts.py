#!/usr/bin/env python3
"""Create a Qwen image-editing manifest from summarized frame captions."""

import argparse
import asyncio
import json
from pathlib import Path

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm


PROMPT = """Create {count} concise image-editing prompts that preserve the
depicted action while changing visually unimportant properties such as color,
texture, material, or background details. Return a JSON list of strings only.

Frame caption: {caption}"""


async def generate(client, model, caption, count, semaphore):
    async with semaphore:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": PROMPT.format(count=count, caption=caption),
                }
            ],
        )
    text = response.choices[0].message.content
    return json.loads(text[text.find("["):text.rfind("]") + 1])


async def run(args):
    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)
    semaphore = asyncio.Semaphore(args.concurrency)
    records = []
    for caption_file in sorted(args.captions_dir.glob("*.jsonl")):
        video = caption_file.stem.replace("_summarized", "")
        items = [
            json.loads(line)
            for line in caption_file.read_text().splitlines()
            if line
        ]
        prompts = await tqdm.gather(
            *[
                generate(
                    client,
                    args.model,
                    item["caption"],
                    args.rewrites,
                    semaphore,
                )
                for item in items
            ],
            desc=video,
        )
        for item, frame_prompts in zip(items, prompts):
            frame = Path(item["image_id"]).name
            image_path = args.frames_root / video / frame
            for index, edit_prompt in enumerate(frame_prompts):
                records.append(
                    {
                        "id": f"{video}_{Path(frame).stem}_rewrite_{index + 1}",
                        "image_path": str(image_path),
                        "prompt": edit_prompt,
                    }
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captions-dir", type=Path, required=True)
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument(
        "--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
    )
    parser.add_argument("--rewrites", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=32)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
