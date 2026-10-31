"""Time model for the whole system.

RAGLedger mixes two kinds of time, and conflating them was a real defect:

* **Production** uses real POSIX seconds (``SystemClock``).  Source age, burst
  windows and trust decay are all expressed in seconds, so wall-clock values are
  the natural unit.
* **Simulation** uses a deterministic logical clock
  (``LogicalClock``): every query-stream step maps to
  ``epoch + step * step_seconds``.  Nothing in a simulation reads the current
  wall clock, so a run on two machines (or on the same machine next week)
  produces identical ages, burst counts and decay decisions.

Rules enforced by this module:

* every public API that accepts a timestamp accepts an explicit ``float`` and
  falls back to an injected ``Clock`` - never directly to ``time.time()``;
* conversion between a step and a timestamp happens in exactly one place
  (:meth:`LogicalClock.timestamp` / :meth:`Clock.step_of`);
* the evaluation harness records both the step and its timestamp so metrics can
  use one timeline.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable

__all__ = ["Clock", "SystemClock", "LogicalClock", "build_clock"]


@runtime_checkable
class Clock(Protocol):
    """Minimal clock protocol used by ingestion, signals, trust and policy."""

    mode: str

    def now(self) -> float:
        """Current time in POSIX seconds."""

    def timestamp(self, step: int) -> float:
        """Timestamp of a simulation step (wall clocks ignore the step)."""

    def step_of(self, timestamp: float) -> int:
        """Best-effort inverse of :meth:`timestamp`."""

    def set_step(self, step: int) -> None:
        """Record the current stream step (a no-op for wall clocks)."""


class SystemClock:
    """Real wall clock.  Production mode."""

    mode = "wall"

    def now(self) -> float:
        return time.time()

    def timestamp(self, step: int = 0) -> float:  # noqa: ARG002 - steps do not map to wall time
        return self.now()

    def step_of(self, timestamp: float) -> int:  # noqa: ARG002
        # A wall-clock run has no step grid.  Return 0 rather than a false step.
        return 0

    def set_step(self, step: int) -> None:  # noqa: ARG002
        return None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "SystemClock()"


class LogicalClock:
    """Deterministic step -> timestamp mapping for simulations.

    ``timestamp(step) == epoch + step * step_seconds``.  The clock also tracks
    the current step so components that ask for ``now()`` without a step
    (trust ``get_trust`` snapshots) see the same instant as the query being
    processed.
    """

    mode = "logical"

    def __init__(self, epoch: float = 1_700_000_000.0, step_seconds: float = 3600.0,
                 step: int = 0) -> None:
        if step_seconds <= 0:
            raise ValueError("step_seconds must be positive")
        self.epoch = float(epoch)
        self.step_seconds = float(step_seconds)
        self._step = int(step)

    def timestamp(self, step: int) -> float:
        return self.epoch + int(step) * self.step_seconds

    def now(self) -> float:
        return self.timestamp(self._step)

    def step_of(self, timestamp: float) -> int:
        return int(round((float(timestamp) - self.epoch) / self.step_seconds))

    def set_step(self, step: int) -> None:
        self._step = int(step)

    @property
    def current_step(self) -> int:
        return self._step

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"LogicalClock(epoch={self.epoch}, step_seconds={self.step_seconds}, "
                f"step={self._step})")


def build_clock(config) -> Clock:
    """Build the clock described by a :class:`TimeConfig` (duck-typed)."""
    mode = getattr(config, "mode", "wall")
    if mode == "logical":
        return LogicalClock(epoch=getattr(config, "epoch", 1_700_000_000.0),
                            step_seconds=getattr(config, "step_seconds", 3600.0))
    if mode == "wall":
        return SystemClock()
    raise ValueError(f"unknown time mode: {mode!r} (expected 'wall' or 'logical')")
