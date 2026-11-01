"""A made-up GPU and engine for dry runs and tests. Nothing it produces is a measurement.

A decode step costs fixed memory time plus compute that scales with tokens verified and
inversely with the clock. Speculation verifies more tokens per step and keeps some of them.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from sj import load
from sj.gpu import GpuInfo

MAX_CLOCK_MHZ = 1980.0
CLOCKS_MHZ = tuple(float(c) for c in range(1980, 344, -15))
IDLE_W = 75.0
STATIC_W = 110.0  # memory and board while busy
ACTIVE_W = 140.0  # SMs busy waiting on memory, at full clock and voltage
COMPUTE_W = 420.0  # SMs doing math, at full clock and voltage
MEMORY_S = 0.008  # per step, whatever the batch
TOKEN_S = 0.00011  # per verified token, at full clock
DRAFT_S = 0.00003  # per drafted token, at full clock
DRAFT_MEMORY = 0.08  # the draft model's share of the memory cost
DRAFT_LEN = 3
ACCEPTANCE = 0.7
MEMORY_BOUND_BELOW = 0.45  # clock share under which the memory cost itself grows


def voltage_scaled(share: float) -> float:
    """Dynamic power at a clock share: frequency times voltage squared, roughly."""
    return float(share**2.6)


def accepted_per_step(acceptance: float, draft_len: int) -> float:
    """Expected tokens produced per verification step, the target's own token included."""
    return (1 - acceptance ** (draft_len + 1)) / (1 - acceptance)


class SimGpu:
    def __init__(self, speculative: bool = False) -> None:
        self.speculative = speculative
        self._lock = threading.Lock()
        self._share = 1.0
        self._active = 0
        self._energy_j = 0.0
        self._last = time.perf_counter()
        self.drafts = 0.0
        self.draft_tokens = 0.0
        self.accepted = 0.0

    def info(self) -> GpuInfo:
        return GpuInfo("Simulated GPU", "simulated", MAX_CLOCK_MHZ, CLOCKS_MHZ)

    def lock_clock(self, mhz: float) -> None:
        with self._lock:
            self._advance()
            self._share = mhz / MAX_CLOCK_MHZ

    def reset_clock(self) -> None:
        with self._lock:
            self._advance()
            self._share = 1.0

    def energy_j(self) -> float:
        with self._lock:
            self._advance()
            return self._energy_j

    def power_w(self) -> float:
        if not self._active:
            return IDLE_W
        memory_s, compute_s = self._step_parts()
        busy = compute_s / (memory_s + compute_s)
        dynamic = ACTIVE_W * (1 - busy) + COMPUTE_W * busy
        return IDLE_W + STATIC_W + dynamic * voltage_scaled(self._share)

    def sm_clock_mhz(self) -> float:
        return MAX_CLOCK_MHZ * self._share if self._active else 345.0

    def begin(self) -> None:
        with self._lock:
            self._advance()
            self._active += 1

    def end(self) -> None:
        with self._lock:
            self._advance()
            self._active -= 1

    def tokens_per_step(self) -> float:
        return accepted_per_step(ACCEPTANCE, DRAFT_LEN) if self.speculative else 1.0

    def step_s(self) -> float:
        memory_s, compute_s = self._step_parts()
        return memory_s + compute_s

    def count_step(self) -> None:
        if self.speculative:
            with self._lock:
                self.drafts += 1
                self.draft_tokens += DRAFT_LEN
                self.accepted += self.tokens_per_step() - 1

    def _step_parts(self) -> tuple[float, float]:
        batch, share = max(1, self._active), self._share
        memory_s = MEMORY_S * max(1.0, MEMORY_BOUND_BELOW / share)
        if not self.speculative:
            return memory_s, batch * TOKEN_S / share
        verify = batch * (DRAFT_LEN + 1) * TOKEN_S
        draft = batch * DRAFT_LEN * DRAFT_S
        return memory_s * (1 + DRAFT_MEMORY), (verify + draft) / share

    def _advance(self) -> None:
        now = time.perf_counter()
        self._energy_j += self.power_w() * (now - self._last)
        self._last = now


def answer_tokens(text: str, max_tokens: int) -> int:
    """A stable answer length per prompt, so every arm and clock does the same work."""
    digest = int(hashlib.sha256(text.encode()).hexdigest(), 16)
    return min(max_tokens, max_tokens // 2 + digest % max_tokens)


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 512  # the default of 5 resets connections under load


def _handler(gpu: SimGpu) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_: Any) -> None:
            pass

        def _send(self, body: bytes, content_type: str = "application/json") -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/metrics":
                counters = {
                    load.DRAFTS: gpu.drafts,
                    load.DRAFT_TOKENS: gpu.draft_tokens,
                    load.ACCEPTED: gpu.accepted,
                }
                lines = [f"{name} {value}" for name, value in counters.items()]
                body = "\n".join(lines) if gpu.speculative else "# no drafts"
                self._send(body.encode(), "text/plain")
            elif self.path == "/v1/models":
                self._send(json.dumps({"data": [{"id": "simulated-model"}]}).encode())
            else:
                self._send(json.dumps({"version": "simulated"}).encode())

        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            text = request["messages"][0]["content"]
            target = answer_tokens(text, request["max_tokens"])
            made = 0.0
            gpu.begin()
            try:
                while made < target:
                    time.sleep(gpu.step_s())
                    gpu.count_step()
                    made += gpu.tokens_per_step()
            finally:
                gpu.end()
            usage = {"prompt_tokens": len(text) // 4, "completion_tokens": target}
            self._send(json.dumps({"choices": [], "usage": usage}).encode())

    return Handler


def serve(gpu: SimGpu) -> tuple[str, ThreadingHTTPServer]:
    """Starts the simulated engine on a free local port."""
    server = _Server(("127.0.0.1", 0), _handler(gpu))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}", server
