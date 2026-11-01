"""Closed-loop chat load against vLLM's OpenAI-compatible server, plus its draft counters."""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from sj.prompts import Prompt

# vLLM's Prometheus counters for speculative decoding.
DRAFTS = "vllm:spec_decode_num_drafts_total"
DRAFT_TOKENS = "vllm:spec_decode_num_draft_tokens_total"
ACCEPTED = "vllm:spec_decode_num_accepted_tokens_total"


@dataclass(frozen=True)
class Workload:
    batch: int
    duration_s: float
    max_tokens: int = 256


@dataclass
class LoadResult:
    requests: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    elapsed_s: float = 0.0
    latencies_s: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class Drafts:
    drafts: float = 0.0
    draft_tokens: float = 0.0
    accepted: float = 0.0

    def __sub__(self, other: Drafts) -> Drafts:
        return Drafts(
            self.drafts - other.drafts,
            self.draft_tokens - other.draft_tokens,
            self.accepted - other.accepted,
        )

    def acceptance_rate(self) -> float | None:
        """Share of drafted tokens the target model kept."""
        return self.accepted / self.draft_tokens if self.draft_tokens else None

    def acceptance_length(self) -> float | None:
        """Tokens produced per verification step: the accepted drafts plus one from the target."""
        return 1 + self.accepted / self.drafts if self.drafts else None


def parse_drafts(text: str) -> Drafts:
    """Sums each counter over its labels in Prometheus text format."""
    totals = {DRAFTS: 0.0, DRAFT_TOKENS: 0.0, ACCEPTED: 0.0}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        name = re.split(r"[{ ]", line, maxsplit=1)[0]
        if name in totals:
            totals[name] += float(line.rsplit(" ", 1)[1])
    return Drafts(totals[DRAFTS], totals[DRAFT_TOKENS], totals[ACCEPTED])


def read_drafts(url: str) -> Drafts:
    return parse_drafts(httpx.get(f"{url}/metrics", timeout=10).raise_for_status().text)


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile, `q` in [0, 1]."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


async def _request(
    client: httpx.AsyncClient, url: str, body: dict[str, Any], out: LoadResult
) -> None:
    start = time.perf_counter()
    response = await client.post(f"{url}/v1/chat/completions", json=body)
    response.raise_for_status()
    usage = response.json()["usage"]
    out.latencies_s.append(time.perf_counter() - start)
    out.requests += 1
    out.prompt_tokens += usage["prompt_tokens"]
    out.output_tokens += usage["completion_tokens"]


def request_body(model: str, prompt: Prompt, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt.text}],
        "max_tokens": max_tokens,
        "temperature": 0,
        # Qwen3 would otherwise think first; a direct answer is the common serving case.
        "chat_template_kwargs": {"enable_thinking": False},
    }


async def run(
    url: str, model: str, prompts: list[Prompt], workload: Workload, offset: int = 0
) -> LoadResult:
    """Keeps `batch` requests in flight for `duration_s`, then lets the last ones finish.

    Prompts are taken in order from `offset`, so every arm and clock sees the same text.
    """
    out = LoadResult()
    cursor = offset
    limits = httpx.Limits(max_connections=workload.batch + 4)

    def next_body() -> dict[str, Any]:
        nonlocal cursor
        prompt = prompts[cursor % len(prompts)]
        cursor += 1
        return request_body(model, prompt, workload.max_tokens)

    async with httpx.AsyncClient(timeout=900, limits=limits) as client:
        start = time.perf_counter()
        deadline = start + workload.duration_s

        async def worker() -> None:
            while time.perf_counter() < deadline:
                await _request(client, url, next_body(), out)

        await asyncio.gather(*(worker() for _ in range(workload.batch)))
        out.elapsed_s = time.perf_counter() - start
    return out


def served_model(url: str) -> tuple[str, str]:
    """The server's first model and its engine version."""
    with httpx.Client(base_url=url, timeout=10) as client:
        engine = client.get("/version").json().get("version", "unknown")
        model = client.get("/v1/models").json()["data"][0]["id"]
    return str(model), str(engine)


def dumps(record: dict[str, Any]) -> str:
    return json.dumps(record)
