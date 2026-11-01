"""Real prompts from public datasets, so a draft model has text it can predict.

`sj-prompts` downloads them once into prompts/prompts.jsonl, which is committed:
- code: HumanEval (MIT), each function signature and docstring with a request to finish it
- chat: MT-Bench prompts (Apache-2.0), the first turn of each conversation
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

ROWS_URL = "https://datasets-server.huggingface.co/rows"
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "prompts.jsonl"
CODE_ASK = "Complete this Python function. Reply with only the code.\n\n"


@dataclass(frozen=True)
class Prompt:
    source: str
    text: str


def _rows(client: httpx.Client, dataset: str, config: str, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    while True:
        params: dict[str, str | int] = {"dataset": dataset, "config": config, "split": split}
        params |= {"offset": len(rows), "length": 100}
        page = client.get(ROWS_URL, params=params).raise_for_status().json()
        rows += [r["row"] for r in page["rows"]]
        if len(rows) >= page["num_rows_total"]:
            return rows


def download() -> list[Prompt]:
    with httpx.Client(timeout=60) as client:
        code = _rows(client, "openai/openai_humaneval", "openai_humaneval", "test")
        chat = _rows(client, "HuggingFaceH4/mt_bench_prompts", "default", "train")
    return [Prompt("humaneval", CODE_ASK + r["prompt"]) for r in code] + [
        Prompt("mt-bench", r["prompt"][0]) for r in chat
    ]


def read(path: Path = DEFAULT_PATH) -> list[Prompt]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [Prompt(r["source"], r["text"]) for r in rows]


def interleave(prompts: list[Prompt]) -> list[Prompt]:
    """Alternates sources so every stretch of requests mixes code and chat."""
    groups: dict[str, list[Prompt]] = {}
    for p in prompts:
        groups.setdefault(p.source, []).append(p)
    out: list[Prompt] = []
    for i in range(max(len(g) for g in groups.values())):
        out += [g[i % len(g)] for g in groups.values()]
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_PATH)
    args = p.parse_args(argv)
    prompts = download()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"source": p.source, "text": p.text}) for p in prompts]
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(prompts)} prompts to {args.out}")


if __name__ == "__main__":
    main()
