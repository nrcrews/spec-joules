import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from sj import gpu, load, plot, prompts, sim, sweep


def info(*clocks: float) -> gpu.GpuInfo:
    return gpu.GpuInfo("test", "test", max(clocks), tuple(sorted(clocks, reverse=True)))


def test_clock_steps_snap_to_supported_clocks_from_the_max_down():
    card = info(*range(1980, 344, -15))
    assert gpu.clock_steps(card, 6, 0.4) == [1980, 1740, 1500, 1260, 1035, 795]
    with pytest.raises(ValueError):
        gpu.clock_steps(card, 1, 0.4)


def test_clock_steps_drop_duplicates_when_the_card_has_few_clocks():
    assert gpu.clock_steps(info(1980, 1000), 4, 0.5) == [1980, 1000]


def test_parse_drafts_sums_counters_over_labels():
    text = """# HELP vllm:spec_decode_num_drafts_total drafts
vllm:spec_decode_num_drafts_total{engine="0",model_name="m"} 10.0
vllm:spec_decode_num_drafts_total{engine="1",model_name="m"} 30.0
vllm:spec_decode_num_draft_tokens_total{model_name="m"} 120.0
vllm:spec_decode_num_accepted_tokens_total{model_name="m"} 60.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 35.0
"""
    drafts = load.parse_drafts(text)
    assert drafts == load.Drafts(40, 120, 60)
    assert drafts.acceptance_rate() == 0.5
    assert drafts.acceptance_length() == 2.5


def test_a_server_that_never_drafts_has_no_acceptance():
    nothing = load.parse_drafts("# none") - load.Drafts()
    assert nothing.acceptance_rate() is None and nothing.acceptance_length() is None


def test_interleave_alternates_sources():
    ps = [prompts.Prompt("a", "1"), prompts.Prompt("a", "2"), prompts.Prompt("b", "x")]
    assert [p.text for p in prompts.interleave(ps)] == ["1", "x", "2", "x"]


def test_the_committed_prompts_are_real_code_and_chat():
    ps = prompts.read()
    assert {p.source for p in ps} == {"humaneval", "mt-bench"}
    assert len(ps) >= 200 and all(p.text.strip() for p in ps)


def table(rows: dict[tuple[str, float | None], float], batch: int = 1) -> plot.Table:
    return {(arm, batch, clock): plot.Cell(j, 1.0, None) for (arm, clock), j in rows.items()}


def test_savings_compare_each_arm_at_its_own_best_clock_with_plain():
    t = table(
        {
            ("plain", None): 1.0,
            ("plain", 1980): 1.0,
            ("plain", 1200): 0.8,
            ("plain", 800): 0.9,
            ("spec", None): 0.6,
            ("spec", 1980): 0.6,
            ("spec", 1200): 0.5,
            ("spec", 800): 0.55,
        }
    )
    s = plot.savings(t, 1)
    assert (s.plain_best_mhz, s.spec_best_mhz) == (1200, 1200)
    assert (round(s.clock), round(s.spec), round(s.both), round(s.total)) == (20, 40, 50, 60)
    assert round(s.independent) == 52
    assert s.verdict == "stack"


def saving(batch: int, clock: float, spec: float, both: float) -> plot.Savings:
    return plot.Savings(batch, clock, spec, both, 1200, 1200)


def test_verdict_separates_adding_up_stacking_and_competing():
    assert saving(1, 20, 40, 59).verdict == "add up"
    assert saving(1, 20, 40, 51).verdict == "stack"
    assert saving(1, 20, 40, 45).verdict == "compete"


def test_headline_title_follows_the_verdicts():
    stack, compete = saving(1, 20, 40, 52), saving(1, 20, 40, 30)
    assert plot.headline_title([stack, stack]).endswith("stack at every batch size")
    rows = [stack, plot.Savings(8, 20, 40, 52, 0, 0), plot.Savings(64, 20, -5, 5, 0, 0)]
    assert plot.headline_title(rows) == (
        "Speculation and a lower clock stack up to batch 8, then compete"
    )
    mixed = [compete, plot.Savings(2, 20, 40, 52, 0, 0), plot.Savings(4, 20, 40, 30, 0, 0)]
    assert plot.headline_title(mixed).endswith("compete at batch 1, 4, and help elsewhere")


def test_best_clock_title_reports_which_way_it_moves():
    def moved(plain: float, spec: float) -> plot.Savings:
        return plot.Savings(1, 0, 0, 0, plain, spec)

    assert plot.best_clock_title([moved(1000, 1000), moved(1000, 1010)]).endswith(
        "move the best clock"
    )
    assert plot.best_clock_title([moved(1000, 1260)]).endswith("moves up")
    assert plot.best_clock_title([moved(1260, 1000)]).endswith("moves down")
    assert plot.best_clock_title([moved(1000, 1260), moved(1260, 1000)]).endswith("both ways")


def test_check_arm_refuses_a_mislabelled_server():
    with pytest.raises(SystemExit, match="did not draft"):
        sweep.check_arm("spec", {"drafts": 0})
    with pytest.raises(SystemExit, match="drafted"):
        sweep.check_arm("plain", {"drafts": 5})
    sweep.check_arm("spec", {"drafts": 5})


class Recorder(sim.SimGpu):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    def lock_clock(self, mhz: float) -> None:
        self.calls.append(f"lock {mhz:.0f}")
        super().lock_clock(mhz)

    def reset_clock(self) -> None:
        self.calls.append("reset")
        super().reset_clock()


def test_sweep_unlocks_the_clock_when_a_run_fails(monkeypatch, tmp_path: Path):
    card = Recorder()
    monkeypatch.setattr(sim, "SimGpu", lambda speculative: card)

    def fail(*_args: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(sweep, "measure", fail)
    argv = f"--simulate --arm plain --clocks 990 --settle 0 --out {tmp_path / 'r.jsonl'}"
    with pytest.raises(RuntimeError, match="boom"):
        sweep.main(argv.split())
    assert card.calls[-1] == "reset"


def test_sigterm_unlocks_the_clock(tmp_path: Path):
    """A real SIGTERM mid-sweep still reaches the reset."""
    script = f"""
import sys
from sj import sim, sweep
card = sim.SimGpu()
sim.SimGpu = lambda speculative: card
reset = card.reset_clock
def noisy_reset():
    reset()
    print("RESET", flush=True)
card.reset_clock = noisy_reset
sweep.main("--simulate --arm plain --clocks 990 --batches 1 --duration 30 --settle 0 "
           "--out {tmp_path / "r.jsonl"}".split())
"""
    child = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    assert child.stderr is not None
    while "clock 990" not in child.stderr.readline():
        pass
    time.sleep(0.5)
    child.send_signal(signal.SIGTERM)
    out, _ = child.communicate(timeout=30)
    assert child.returncode == 143
    assert out.splitlines()[-1] == "RESET"


def test_simulated_sweeps_record_every_run_and_plot_every_chart(tmp_path: Path):
    for arm in ("plain", "spec"):
        sweep.main(
            f"--simulate --arm {arm} --clocks 1980,990 --batches 1,4 --duration 0.4 "
            f"--repeats 1 --settle 0 --max-tokens 8 --out {tmp_path / arm}.jsonl".split()
        )
    lines = (tmp_path / "spec.jsonl").read_text().splitlines()
    meta, *runs = (json.loads(line) for line in lines)
    assert meta["type"] == "meta" and meta["simulated"] and meta["arm"] == "spec"
    assert len(runs) == 3 * 2  # default plus two locked clocks, two batch sizes
    assert all(r["energy_j"] > 0 and r["acceptance_rate"] > 0 for r in runs)
    assert [r["clock_mhz"] for r in runs[::2]] == [None, 1980, 990]

    charts = tmp_path / "charts"
    plot.main([str(tmp_path / "plain.jsonl"), str(tmp_path / "spec.jsonl"), "--out", str(charts)])
    names = {p.name for p in charts.iterdir()}
    assert names == {
        "energy-b1.png",
        "energy-b4.png",
        "best-clock.png",
        "headline.png",
        "summary.md",
    }


def test_plot_refuses_a_single_arm(tmp_path: Path):
    sweep.main(
        f"--simulate --arm plain --clocks 990 --batches 1 --duration 0.2 --repeats 1 "
        f"--settle 0 --max-tokens 4 --out {tmp_path / 'p.jsonl'}".split()
    )
    with pytest.raises(SystemExit, match="one plain and one spec"):
        plot.main([str(tmp_path / "p.jsonl"), "--out", str(tmp_path / "c")])
