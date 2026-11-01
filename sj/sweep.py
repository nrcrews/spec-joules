"""Sweeps a GPU's locked SM clock and batch size against one vLLM server.

Run it once against a plain server (`--arm plain`) and once against a server with a draft
model (`--arm spec`). Each run writes a `meta` line, then one `run` line per
(clock, batch, repeat). The default clock comes first, unlocked. The clock is unlocked again
with `nvidia-smi -rgc` on exit, Ctrl-C and SIGTERM included.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import signal
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TextIO, TypeVar

from sj import load, prompts
from sj.gpu import Gpu, NvmlGpu, Sampler, clock_steps

T = TypeVar("T")
Arm = Literal["plain", "spec"]
BATCHES = [1, 2, 4, 8, 16, 32, 64, 128]


@dataclass(frozen=True)
class Target:
    gpu: Gpu
    url: str
    model: str
    engine: str
    simulated: bool


def csv(parse: Callable[[str], T]) -> Callable[[str], list[T]]:
    return lambda text: [parse(item) for item in text.split(",")]


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=["plain", "spec"], required=True, help="what the server runs")
    p.add_argument("--url", default="http://localhost:8000", help="vLLM server")
    p.add_argument("--gpu", type=int, default=0, help="NVML index of the server's GPU")
    p.add_argument("--clocks", type=csv(float), help="MHz to lock (default: --steps from max)")
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--low-share", type=float, default=0.4, help="lowest clock as share of max")
    p.add_argument("--batches", type=csv(int), default=BATCHES, help="requests in flight")
    p.add_argument("--duration", type=float, default=15.0, help="seconds of load per run")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--settle", type=float, default=3.0, help="seconds after a clock change")
    p.add_argument("--prompts", type=Path, default=prompts.DEFAULT_PATH)
    p.add_argument("--out", type=Path, default=Path("results.jsonl"))
    p.add_argument("--simulate", action="store_true", help="dry run on a made-up GPU and engine")
    return p.parse_args(argv)


@contextmanager
def connect(args: argparse.Namespace) -> Iterator[Target]:
    if not args.simulate:
        model, engine = load.served_model(args.url)
        yield Target(NvmlGpu(args.gpu), args.url, model, engine, simulated=False)
        return

    from sj import sim

    gpu = sim.SimGpu(speculative=args.arm == "spec")
    url, server = sim.serve(gpu)
    try:
        yield Target(gpu, url, "simulated-model", "simulated", simulated=True)
    finally:
        server.shutdown()


def offset(batch: int, repeat: int) -> int:
    """Where in the prompt list a run starts: the same for every arm and clock."""
    return 97 * repeat + 13 * batch


def measure(
    target: Target, texts: list[prompts.Prompt], workload: load.Workload, start: int
) -> dict[str, Any]:
    drafts_before = load.read_drafts(target.url)
    with Sampler(target.gpu) as sampler:
        before_j = target.gpu.energy_j()
        result = asyncio.run(load.run(target.url, target.model, texts, workload, start))
        energy_j = target.gpu.energy_j() - before_j
    drafts = load.read_drafts(target.url) - drafts_before

    tokens = result.output_tokens
    return {
        "requests": result.requests,
        "prompt_tokens": result.prompt_tokens,
        "output_tokens": tokens,
        "elapsed_s": result.elapsed_s,
        "tokens_per_s": tokens / result.elapsed_s,
        "energy_j": energy_j,
        "j_per_token": energy_j / tokens if tokens else None,
        "mean_power_w": energy_j / result.elapsed_s,
        "peak_power_w": sampler.peak_power_w(),
        "median_sm_clock_mhz": sampler.median_clock_mhz(),
        "latency_p50_s": load.percentile(result.latencies_s, 0.5),
        "drafts": drafts.drafts,
        "draft_tokens": drafts.draft_tokens,
        "accepted_tokens": drafts.accepted,
        "acceptance_rate": drafts.acceptance_rate(),
        "acceptance_length": drafts.acceptance_length(),
    }


def check_arm(arm: Arm, row: dict[str, Any]) -> None:
    """Refuses to mislabel a run: a spec server drafts, a plain one doesn't."""
    if (row["drafts"] > 0) != (arm == "spec"):
        seen = "drafted" if row["drafts"] > 0 else "did not draft"
        raise SystemExit(f"--arm {arm} but the server {seen}; check its --speculative-config")


def set_clock(gpu: Gpu, mhz: float | None) -> None:
    if mhz is None:
        gpu.reset_clock()
    else:
        gpu.lock_clock(mhz)


def sweep(target: Target, args: argparse.Namespace, out: TextIO) -> None:
    info = target.gpu.info()
    locked = args.clocks or clock_steps(info, args.steps, args.low_share)
    clocks: list[float | None] = [None, *locked]
    texts = prompts.interleave(prompts.read(args.prompts))
    checked = False

    for repeat in range(args.repeats):
        for clock in clocks:
            set_clock(target.gpu, clock)
            time.sleep(args.settle)
            log(f"repeat {repeat}, clock {'default' if clock is None else f'{clock:.0f} MHz'}")

            for batch in args.batches:
                workload = load.Workload(batch, args.duration, args.max_tokens)
                warmup = dataclasses.replace(workload, duration_s=min(2.0, args.duration))
                asyncio.run(load.run(target.url, target.model, texts, warmup, 10_000))

                row = measure(target, texts, workload, offset(batch, repeat))
                if not checked:
                    check_arm(args.arm, row)
                    checked = True
                record = {
                    "type": "run",
                    "arm": args.arm,
                    "clock_mhz": clock,
                    "batch": batch,
                    "repeat": repeat,
                    **row,
                }
                out.write(json.dumps(record) + "\n")
                out.flush()
                rate = row["acceptance_rate"]
                accepted = "" if rate is None else f"  accept {rate:.0%}"
                log(
                    f"  b={batch:<3} {row['tokens_per_s']:8.1f} tok/s  "
                    f"{row['mean_power_w']:6.1f} W  {row['j_per_token']:.3f} J/tok{accepted}"
                )


def meta(target: Target, args: argparse.Namespace) -> dict[str, Any]:
    info = target.gpu.info()
    return {
        "type": "meta",
        "arm": args.arm,
        "simulated": target.simulated,
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "gpu": info.name,
        "driver": info.driver,
        "max_clock_mhz": info.max_clock_mhz,
        "model": target.model,
        "engine_version": target.engine,
        "max_tokens": args.max_tokens,
        "duration_s": args.duration,
    }


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: sys.exit(143))

    with connect(args) as target, args.out.open("w") as out:
        out.write(json.dumps(meta(target, args)) + "\n")
        try:
            sweep(target, args, out)
        finally:
            target.gpu.reset_clock()
            log("unlocked the SM clock")


if __name__ == "__main__":
    main()
