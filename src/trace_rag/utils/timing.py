from __future__ import annotations

import time
from types import TracebackType
from typing import Optional, Type


class Stopwatch:
    """Context manager measuring wall-clock milliseconds."""

    def __init__(self) -> None:
        self.elapsed_ms: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Stopwatch":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc: Optional[BaseException],
                 tb: Optional[TracebackType]) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
