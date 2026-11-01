"""Energy, power and clock readings for one NVIDIA GPU, and SM clock locking."""

from __future__ import annotations

import statistics
import subprocess
import threading
from dataclasses import dataclass
from typing import Protocol

import pynvml


@dataclass(frozen=True)
class GpuInfo:
    name: str
    driver: str
    max_clock_mhz: float
    clocks_mhz: tuple[float, ...]  # supported SM clocks at the top memory clock, descending


class Gpu(Protocol):
    def info(self) -> GpuInfo: ...
    def lock_clock(self, mhz: float) -> None: ...
    def reset_clock(self) -> None: ...
    def energy_j(self) -> float: ...
    def power_w(self) -> float: ...
    def sm_clock_mhz(self) -> float: ...


class ClockLockDenied(RuntimeError):
    pass


def _text(value: str | bytes) -> str:
    return value.decode() if isinstance(value, bytes) else value


class NvmlGpu:
    def __init__(self, index: int) -> None:
        pynvml.nvmlInit()
        self._index = index
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(index)

    def info(self) -> GpuInfo:
        memory = max(pynvml.nvmlDeviceGetSupportedMemoryClocks(self._handle))
        supported = pynvml.nvmlDeviceGetSupportedGraphicsClocks(self._handle, memory)
        clocks = tuple(sorted((float(c) for c in supported), reverse=True))
        return GpuInfo(
            name=_text(pynvml.nvmlDeviceGetName(self._handle)),
            driver=_text(pynvml.nvmlSystemGetDriverVersion()),
            max_clock_mhz=clocks[0],
            clocks_mhz=clocks,
        )

    def _smi(self, *args: str) -> None:
        done = subprocess.run(
            ["nvidia-smi", "-i", str(self._index), *args], capture_output=True, text=True
        )
        if done.returncode != 0:
            raise ClockLockDenied(f"nvidia-smi {' '.join(args)}: {done.stdout}{done.stderr}")

    def lock_clock(self, mhz: float) -> None:
        self._smi("-lgc", f"{mhz:.0f},{mhz:.0f}")

    def reset_clock(self) -> None:
        self._smi("-rgc")

    def energy_j(self) -> float:
        """The card's cumulative energy counter, more exact than integrating sampled power."""
        return float(pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)) / 1000

    def power_w(self) -> float:
        return float(pynvml.nvmlDeviceGetPowerUsage(self._handle)) / 1000

    def sm_clock_mhz(self) -> float:
        return float(pynvml.nvmlDeviceGetClockInfo(self._handle, pynvml.NVML_CLOCK_SM))


def nearest(clocks: tuple[float, ...], mhz: float) -> float:
    return min(clocks, key=lambda c: abs(c - mhz))


def clock_steps(info: GpuInfo, steps: int, low_share: float) -> list[float]:
    """`steps` supported clocks evenly spaced from the max down to `low_share` of it."""
    if steps < 2:
        raise ValueError("need at least two clocks")
    top, low = info.max_clock_mhz, info.max_clock_mhz * low_share
    wanted = [top - (top - low) * i / (steps - 1) for i in range(steps)]
    picked = [nearest(info.clocks_mhz, w) for w in wanted]
    return sorted(set(picked), reverse=True)


class Sampler:
    """Samples power and SM clock in the background while a load runs."""

    def __init__(self, gpu: Gpu, interval_s: float = 0.1) -> None:
        self._gpu = gpu
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.power_w: list[float] = []
        self.clock_mhz: list[float] = []

    def __enter__(self) -> Sampler:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            self.power_w.append(self._gpu.power_w())
            self.clock_mhz.append(self._gpu.sm_clock_mhz())

    def peak_power_w(self) -> float | None:
        return max(self.power_w, default=None)

    def median_clock_mhz(self) -> float | None:
        return statistics.median(self.clock_mhz) if self.clock_mhz else None
