# spec-joules

Do speculative decoding and a locked GPU clock add up, or compete, for LLM decode energy per token?

On an H100 PCIe serving Qwen3-8B, they **stack up to 16 requests in flight, then compete**.

At small batches, an EAGLE-3 draft saves about half the energy per token and a locked clock of
about 900 MHz saves another quarter of what's left. Together that's 57–62% less energy than
plain decoding, and still 1.6× faster. From 32 requests in flight the GPU's spare compute runs
out: together they save less than independent savings would, and at 128 barely more than the
lower clock alone.

![Energy saved by a lower clock, by speculation, and by both, per batch size](results/h100-pcie/charts/light/headline.png)

## Results

One H100 PCIe (350 W, max SM clock 1755 MHz), vLLM 0.32.0, Qwen3-8B with
`RedHatAI/Qwen3-8B-speculator.eagle3` drafting 3 tokens. Savings are against plain decoding at
the default clock. Medians of 2 repeats, 10 s each.

| In flight | Plain | Clock alone | Speculation alone | Both | If independent | Sum | Speed, both |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2.64 J/token | 23% | 50% | **62%** | 61% | 73% | 1.65× |
| 4 | 0.72 J | 22% | 49% | **62%** | 60% | 70% | 1.74× |
| 16 | 0.22 J | 23% | 43% | **57%** | 57% | 67% | 1.65× |
| 32 | 0.13 J | 24% | 41% | **50%** | 55% | 65% | 1.50× |
| 64 | 0.08 J | 21% | 30% | **34%** | 45% | 51% | 1.21× |
| 128 | 0.05 J | 13% | 9% | **15%** | 22% | 23% | 0.95× |

- **The best clock barely moves.** With or without the draft, energy per token bottoms out
  around 915 MHz (705 MHz at a couple of batch sizes, within 1%). Turning speculation on is no
  reason to raise the clock.
- **Speculation never costs energy here,** even at 128 in flight, but its saving shrinks from
  50% to 9% as the batch fills the GPU's compute.
- **Acceptance was steady:** 43–50% of drafted tokens were kept, at every batch size and clock.
- **Below 915 MHz decode starts to slow faster than power falls,** so 705 MHz rarely helps.

Every batch size, light and dark charts, summary table and raw data:
[`results/h100-pcie`](results/h100-pcie). Settings: `scripts/run.sh --duration 10 --repeats 2`.
The whole run cost $2.72.

## Why ask

- **A lower clock saves decode energy.** Decode mostly waits on memory, so the cores can run
  slower at lower voltage while barely losing speed
  ([tokens-per-watt](https://github.com/nrcrews/tokens-per-watt)).
- **Speculation saves energy at small batches.** A draft model guesses a few tokens and the big
  model checks them in one pass, using compute that would otherwise sit idle. At large batches
  that compute is no longer idle, and speculation costs energy.
- **Both feed on the same idle compute.** If speculation uses up the slack, there is less for a
  lower clock to give back. The best clock could move up, the savings could stop adding up, or
  the answer could flip with batch size.

## What it measures

For each batch size (1 to 128 requests in flight) and each SM clock (the default, then six
locked steps from the card's max down to 40% of it), energy per output token with and without
an EAGLE-3 draft model. Four numbers per batch size, each compared with **plain** (no draft,
default clock):

- **Clock alone**: no draft, at the locked clock with the least energy per token
- **Speculation alone**: draft on, default clock
- **Both**: draft on, at its own best locked clock
- **Sum** and **if independent**: what the two alone would give together. Independent savings
  compound to `1 - (1 - a)(1 - b)`, a bit less than the sum `a + b`.

The charts call it **add up** when both together reach the sum, **stack** when they reach the
independent figure, and **compete** when they fall short of it (2-point tolerance). Chart
titles are computed from the data.

Energy comes from the GPU's own counter (NVML). Prompts are real: HumanEval function stubs and
MT-Bench first turns ([`prompts/prompts.jsonl`](prompts/prompts.jsonl), 244 prompts), so the
draft has text it can predict. Every arm and clock gets the same prompts in the same order, at
temperature 0. Acceptance rate comes from vLLM's `/metrics` counters.

## Run it

Needs root on a host that allows `nvidia-smi -lgc`, plus Docker with GPU support and uv.

```bash
OUT=out scripts/run.sh --duration 10 --repeats 2             # plain sweep, then speculative
uv run sj-plot out/plain.jsonl out/spec.jsonl --out charts   # --theme dark
```

`scripts/run.sh` checks that the clock can be locked, starts vLLM 0.32.0 serving Qwen3-8B, sweeps
it, then restarts it with `RedHatAI/Qwen3-8B-speculator.eagle3` (3 draft tokens) and sweeps again.
The sweep unlocks the clock (`nvidia-smi -rgc`) when it ends, fails, or gets Ctrl-C or SIGTERM.

To sweep a server you started yourself:

```bash
sudo .venv/bin/sj-sweep --arm spec --url http://localhost:8000 --out spec.jsonl
```

Without a GPU (made-up numbers, charts watermarked SIMULATED):

```bash
uv run sj-sweep --simulate --arm plain --duration 2 --repeats 1 --out plain.jsonl
uv run sj-sweep --simulate --arm spec --duration 2 --repeats 1 --out spec.jsonl
uv run sj-plot plain.jsonl spec.jsonl
```

## Caveats

- GPU power only: no CPU, memory, networking or cooling.
- Energy per output token includes the prefill of each prompt, the same in both arms.
- Closed-loop load: each run keeps a fixed number of requests in flight, then waits for the last
  ones to finish.
- One card, one model, one draft, one engine version, two repeats. The PCIe card runs near its
  350 W limit at large batches, so a 700 W SXM card may show bigger clock savings there.

## Prior art

- Ma et al., [*The Illusion of Power Capping in LLM Decode*](https://arxiv.org/abs/2605.11999)
  (2026): on an H200, a static SM clock lock saves up to 32% of decode energy where a power cap
  never engages.
- Fernandez et al., [*Energy Considerations of LLM Inference and Efficiency
  Optimizations*](https://aclanthology.org/2025.acl-long.1563/) (ACL 2025), and Dutta et al.,
  [*Benchmarking the Energy Savings with Speculative Decoding
  Strategies*](https://arxiv.org/abs/2602.09113) (EACL 2026 Findings): speculation's energy
  savings depend on the workload and batch size, and it can cost energy.
- Yang et al., [*PELM*](https://arxiv.org/abs/2609.09662) (2026): speculation and DVFS tuned
  together, on-device.

As far as I can find, nobody has measured the combination on a datacenter GPU.

## Development

```bash
uv run ruff format && uv run ruff check && uv run mypy && uv run pytest
```

MIT license. Prompts: HumanEval (MIT), MT-Bench prompts (Apache-2.0).
