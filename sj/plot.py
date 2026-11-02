"""Turns sweep results into charts whose titles are computed from the data.

Each batch size is compared four ways: plain (default clock), underclocked (best locked
clock), speculative (default clock) and both (draft model at its own best locked clock).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

SIZE_IN = (8.0, 4.5)  # 1600 x 900 at 200 dpi
DPI = 200
TOLERANCE = 2.0  # percentage points of slack before calling it competition
SAME_CLOCK_MHZ = 30.0  # best clocks closer than this count as the same
TIE = 0.01  # energies this close count as equal when picking the best clock
MARKERS = {"Sum", "If independent"}  # legend entries that go after the bars

FONTS = ["Inter", "Arial", "DejaVu Sans"]
LINE: dict[str, Any] = {"linewidth": 2, "marker": "o", "markersize": 6.5, "zorder": 3}
LEGEND: dict[str, Any] = {"loc": "upper left", "bbox_to_anchor": (0.03, 0.835), "frameon": False}
SUM_MARK: dict[str, Any] = {"marker": "_", "s": 260, "linewidths": 2.5, "zorder": 4}
RING_MARK: dict[str, Any] = {"marker": "o", "s": 34, "linewidths": 1.5, "zorder": 5}
WATERMARK: dict[str, Any] = {"fontsize": 32, "fontweight": "bold", "alpha": 0.22, "rotation": 18}


@dataclass(frozen=True)
class Theme:
    surface: str
    ink: str
    ink2: str
    muted: str
    grid: str
    axis: str
    plain: str
    spec: str
    both: str


THEMES = {
    "light": Theme(
        surface="#fcfcfb",
        ink="#0b0b0b",
        ink2="#52514e",
        muted="#898781",
        grid="#e1e0d9",
        axis="#c3c2b7",
        plain="#2a78d6",
        spec="#eb6834",
        both="#1baf7a",
    ),
    "dark": Theme(
        surface="#1a1a19",
        ink="#f0efec",
        ink2="#c3c2b7",
        muted="#898781",
        grid="#2c2c2a",
        axis="#383835",
        plain="#3987e5",
        spec="#d95926",
        both="#199e70",
    ),
}


@dataclass(frozen=True)
class Cell:
    """Medians of the repeats at one (arm, batch, clock); clock None is the default."""

    j_per_token: float
    tokens_per_s: float
    acceptance_rate: float | None


Table = dict[tuple[str, int, float | None], Cell]


@dataclass(frozen=True)
class Savings:
    """Energy per token saved against plain at one batch size, in %."""

    batch: int
    clock: float
    spec: float
    both: float
    plain_best_mhz: float
    spec_best_mhz: float

    @property
    def total(self) -> float:
        return self.clock + self.spec

    @property
    def independent(self) -> float:
        """What both would save if each cut its own share of what's left."""
        return 100 * (1 - (1 - self.clock / 100) * (1 - self.spec / 100))

    @property
    def clock_moved(self) -> bool:
        return abs(self.spec_best_mhz - self.plain_best_mhz) >= SAME_CLOCK_MHZ

    @property
    def verdict(self) -> str:
        """`add up` to the sum, `stack` like independent savings, or `compete`."""
        if self.both >= self.total - TOLERANCE:
            return "add up"
        if self.both >= self.independent - TOLERANCE:
            return "stack"
        return "compete"


def load(paths: list[Path]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metas, runs = [], []
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        metas += [row for row in rows if row["type"] == "meta"]
        runs += [row for row in rows if row["type"] == "run"]
    if len({(m["gpu"], m["model"], m["simulated"]) for m in metas}) != 1:
        raise SystemExit("every results file must come from the same GPU type and model")
    if {m["arm"] for m in metas} != {"plain", "spec"}:
        raise SystemExit("pass one plain and one spec results file")
    return metas[0], runs


def _median(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return float(statistics.median(present)) if present else None


def summarize(runs: list[dict[str, Any]]) -> Table:
    groups: defaultdict[tuple[str, int, float | None], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        groups[(run["arm"], run["batch"], run["clock_mhz"])].append(run)
    table: Table = {}
    for key, rows in groups.items():
        j = _median([r["j_per_token"] for r in rows])
        speed = _median([r["tokens_per_s"] for r in rows])
        table[key] = Cell(
            math.nan if j is None else j,
            math.nan if speed is None else speed,
            _median([r["acceptance_rate"] for r in rows]),
        )
    return table


def batches(table: Table) -> list[int]:
    by_arm = [{b for a, b, _ in table if a == arm} for arm in ("plain", "spec")]
    return sorted(by_arm[0] & by_arm[1])


def mj(table: Table, arm: str, batch: int, clock: float | None) -> float:
    return 1000 * table[(arm, batch, clock)].j_per_token


def locked_clocks(table: Table, arm: str, batch: int) -> list[float]:
    return sorted(c for a, b, c in table if a == arm and b == batch and c is not None)


def best_clock(table: Table, arm: str, batch: int) -> float:
    """The fastest locked clock within `TIE` of the least energy per token."""
    clocks = locked_clocks(table, arm, batch)
    least = min(table[(arm, batch, c)].j_per_token for c in clocks)
    return max(c for c in clocks if table[(arm, batch, c)].j_per_token <= least * (1 + TIE))


def savings(table: Table, batch: int) -> Savings:
    base = table[("plain", batch, None)].j_per_token
    plain_best, spec_best = best_clock(table, "plain", batch), best_clock(table, "spec", batch)

    def saved(arm: str, clock: float | None) -> float:
        return 100 * (1 - table[(arm, batch, clock)].j_per_token / base)

    return Savings(
        batch,
        clock=saved("plain", plain_best),
        spec=saved("spec", None),
        both=saved("spec", spec_best),
        plain_best_mhz=plain_best,
        spec_best_mhz=spec_best,
    )


def _runs_of(labels: list[str]) -> list[tuple[str, int, int]]:
    """Consecutive stretches of equal labels as (label, first index, last index)."""
    out: list[tuple[str, int, int]] = []
    for i, label in enumerate(labels):
        if out and out[-1][0] == label:
            out[-1] = (label, out[-1][1], i)
        else:
            out.append((label, i, i))
    return out


def headline_title(rows: list[Savings]) -> str:
    runs = _runs_of([r.verdict for r in rows])
    lead = "Speculation and a lower clock"
    if len(runs) == 1:
        return f"{lead} {runs[0][0]} at every batch size"
    if len(runs) == 2:
        first, _, edge = runs[0]
        return f"{lead} {first} up to batch {rows[edge].batch}"
    competing = ", ".join(str(r.batch) for r in rows if r.verdict == "compete")
    return f"{lead} compete at batch {competing or 'none'}, and help elsewhere"


def best_clock_title(rows: list[Savings]) -> str:
    moves = [r.spec_best_mhz - r.plain_best_mhz for r in rows]
    up = sum(m >= SAME_CLOCK_MHZ for m in moves)
    down = sum(m <= -SAME_CLOCK_MHZ for m in moves)
    if not up and not down:
        return "Speculation doesn't move the best clock"
    if 2 * (up + down) < len(rows):
        return "Speculation barely moves the best clock"
    if up and not down:
        return "With speculation on, the best clock moves up"
    if down and not up:
        return "With speculation on, the best clock moves down"
    return "Speculation moves the best clock both ways"


def energy_title(table: Table, batch: int) -> str:
    clocks = locked_clocks(table, "plain", batch)
    cheaper = [
        table[("spec", batch, c)].j_per_token < table[("plain", batch, c)].j_per_token
        for c in clocks
        if ("spec", batch, c) in table
    ]
    if all(cheaper):
        return f"Batch {batch}: speculation saves energy at every clock"
    if not any(cheaper):
        return f"Batch {batch}: speculation costs energy at every clock"
    return f"Batch {batch}: speculation saves energy at some clocks, not others"


class Canvas:
    def __init__(self, meta: dict[str, Any], theme: Theme) -> None:
        self.meta = meta
        self.t = theme
        plt.rcParams["font.sans-serif"] = FONTS

    def chart(self, title: str, subtitle: str) -> tuple[Figure, Axes]:
        fig = plt.figure(figsize=SIZE_IN, dpi=DPI, facecolor=self.t.surface)
        self.source(fig)
        fig.text(0.04, 0.92, title, fontsize=15, fontweight="bold", color=self.t.ink)
        fig.text(0.04, 0.86, subtitle, fontsize=10, color=self.t.ink2)
        ax = fig.add_axes((0.1, 0.15, 0.85, 0.6), facecolor=self.t.surface)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(self.t.axis)
        ax.tick_params(colors=self.t.muted, length=0, labelsize=9)
        ax.grid(axis="y", color=self.t.grid, linewidth=0.75)
        ax.set_axisbelow(True)
        return fig, ax

    def source(self, fig: Figure) -> None:
        """The footer naming what was measured, and a watermark on simulated data."""
        m = self.meta
        footer = (
            f"{m['gpu']} · {m['model']} · vLLM {m['engine_version']} · "
            "GPU power from NVML · output tokens only in the denominator"
        )
        fig.text(0.04, 0.035, footer, fontsize=7.5, color=self.t.muted)
        if m["simulated"]:
            mark = "SIMULATED — NOT MEASURED"
            fig.text(0.5, 0.48, mark, color=self.t.muted, ha="center", va="center", **WATERMARK)

    def legend(self, fig: Figure, ax: Axes) -> None:
        handles, labels = ax.get_legend_handles_labels()
        order = sorted(range(len(labels)), key=lambda i: labels[i] in MARKERS)
        handles, labels = [handles[i] for i in order], [labels[i] for i in order]
        legend = fig.legend(handles, labels, ncol=len(labels), fontsize=8.5, **LEGEND)
        for text in legend.get_texts():
            text.set_color(self.t.ink2)

    def xlabel(self, ax: Axes, text: str) -> None:
        ax.set_xlabel(text, color=self.t.muted, fontsize=8.5)

    def save(self, fig: Figure, path: Path) -> None:
        fig.savefig(path, facecolor=self.t.surface)
        plt.close(fig)


def energy_chart(c: Canvas, table: Table, batch: int, out: Path) -> None:
    s = savings(table, batch)
    subtitle = (
        f"Energy per output token, {batch} in flight. "
        f"Plain: {mj(table, 'plain', batch, None):,.0f} mJ. "
        f"Saved: clock {s.clock:.0f}%, speculation {s.spec:.0f}%, both {s.both:.0f}%."
    )
    fig, ax = c.chart(energy_title(table, batch), subtitle)
    dashed: dict[str, Any] = {"linewidth": 1.25, "linestyle": (0, (4, 3))}
    arms = (
        ("plain", c.t.plain, "Plain", "Underclocked", 9),
        ("spec", c.t.spec, "Speculative", "Both", -15),
    )

    for arm, color, default_label, locked_label, label_dy in arms:
        clocks = locked_clocks(table, arm, batch)
        ax.axhline(mj(table, arm, batch, None), color=color, zorder=2, **dashed)
        ax.plot([], [], color=color, label=default_label, **dashed)
        ys = [mj(table, arm, batch, clock) for clock in clocks]
        ax.plot(clocks, ys, color=color, mec=c.t.surface, label=locked_label, **LINE)

        best = best_clock(table, arm, batch)
        xy = (best, mj(table, arm, batch, best))
        offset: dict[str, Any] = {"xytext": (0, label_dy), "textcoords": "offset points"}
        ax.annotate(f"{best:.0f} MHz", xy, ha="center", fontsize=8.5, color=c.t.ink2, **offset)

    c.legend(fig, ax)
    clocks = locked_clocks(table, "plain", batch)
    ax.set_xticks(clocks, [f"{clock:,.0f}" for clock in clocks])
    c.xlabel(ax, "Locked SM clock (MHz); dashed lines: default clock")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:,.0f} mJ")
    c.save(fig, out / f"energy-b{batch}.png")


def best_clock_chart(c: Canvas, rows: list[Savings], out: Path) -> None:
    moved = [str(r.batch) for r in rows if r.clock_moved]
    subtitle = (
        "Fastest locked clock within 1% of the least energy per token. "
        f"Moves at batch: {', '.join(moved) or 'none'}."
    )
    fig, ax = c.chart(best_clock_title(rows), subtitle)
    xs = list(range(len(rows)))
    lines = (
        ("Without speculation", c.t.plain, [r.plain_best_mhz for r in rows]),
        ("With speculation", c.t.spec, [r.spec_best_mhz for r in rows]),
    )
    for label, color, ys in lines:
        ax.plot(xs, ys, color=color, mec=c.t.surface, label=label, **LINE)

    c.legend(fig, ax)
    ax.set_xticks(xs, [str(r.batch) for r in rows])
    c.xlabel(ax, "Requests in flight")
    top = max(y for _, _, ys in lines for y in ys)
    ax.set_ylim(0, top * 1.15)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:,.0f} MHz")
    c.save(fig, out / "best-clock.png")


def headline_chart(c: Canvas, rows: list[Savings], out: Path) -> None:
    subtitle = (
        "% of plain energy per token saved. Independent savings compound "
        "to 1 - (1 - a)(1 - b), which is less than the sum."
    )
    fig, ax = c.chart(headline_title(rows), subtitle)
    width = 0.26
    xs = list(range(len(rows)))
    bars = (
        ("Lower clock alone", c.t.plain, [r.clock for r in rows], -width),
        ("Speculation alone", c.t.spec, [r.spec for r in rows], 0.0),
        ("Both together", c.t.both, [r.both for r in rows], width),
    )
    for label, color, values, shift in bars:
        ax.bar([x + shift for x in xs], values, width * 0.92, color=color, label=label, zorder=3)

    together = [x + width for x in xs]
    sums = [r.total for r in rows]
    independent = [r.independent for r in rows]
    ax.scatter(together, sums, label="Sum", color=c.t.ink, **SUM_MARK)
    ring: dict[str, Any] = {**RING_MARK, "color": c.t.surface, "edgecolors": c.t.ink}
    ax.scatter(together, independent, label="If independent", **ring)

    ax.axhline(0, color=c.t.axis, linewidth=0.75, zorder=2)
    c.legend(fig, ax)
    ax.set_xticks(xs, [str(r.batch) for r in rows])
    c.xlabel(ax, "Requests in flight")
    values = [v for r in rows for v in (r.clock, r.spec, r.both, r.total)]
    ax.set_ylim(min(0.0, min(values) * 1.2), max(values) * 1.25)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
    c.save(fig, out / "headline.png")


def markdown(table: Table, rows: list[Savings]) -> str:
    lines = [
        "| Batch | Plain mJ/tok | Clock alone | Spec alone | Sum | Both | "
        "Best clock (plain → spec) | Acceptance |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        base = mj(table, "plain", r.batch, None)
        rate = table[("spec", r.batch, None)].acceptance_rate
        accepted = "n/a" if rate is None else f"{100 * rate:.0f}%"
        lines.append(
            f"| {r.batch} | {base:,.0f} | {r.clock:.0f}% | {r.spec:.0f}% | {r.total:.0f}% | "
            f"{r.both:.0f}% | {r.plain_best_mhz:.0f} → {r.spec_best_mhz:.0f} MHz | {accepted} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path, nargs="+", help="plain and spec results files")
    p.add_argument("--out", type=Path, default=Path("charts"))
    p.add_argument("--theme", choices=sorted(THEMES), default="light")
    args = p.parse_args(argv)

    meta, runs = load(args.results)
    table = summarize(runs)
    rows = [savings(table, b) for b in batches(table)]
    canvas = Canvas(meta, THEMES[args.theme])
    args.out.mkdir(parents=True, exist_ok=True)

    for batch in batches(table):
        energy_chart(canvas, table, batch, args.out)
    best_clock_chart(canvas, rows, args.out)
    headline_chart(canvas, rows, args.out)
    (args.out / "summary.md").write_text(markdown(table, rows))
    print(f"wrote charts to {args.out}/")


if __name__ == "__main__":
    main()
