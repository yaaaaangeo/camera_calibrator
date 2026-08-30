"""Bounded latest-frame handoff for high-rate live camera streams.

ROS callbacks must never enqueue an unbounded number of frames for the Qt event
loop.  This small, Qt-independent buffer keeps exactly one frame: a producer
replaces stale data and the GUI timer periodically takes the newest value.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrameBufferStats:
    received: int
    replaced: int
    delivered: int


class LatestFrameBuffer:
    """Thread-safe single-slot buffer with observable drop statistics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: tuple[np.ndarray, float] | None = None
        self._received = 0
        self._replaced = 0
        self._delivered = 0

    def put(self, frame: np.ndarray, timestamp_sec: float) -> None:
        with self._lock:
            self._received += 1
            if self._latest is not None:
                self._replaced += 1
            self._latest = (frame, timestamp_sec)

    def take(self) -> tuple[np.ndarray, float] | None:
        with self._lock:
            value = self._latest
            self._latest = None
            if value is not None:
                self._delivered += 1
            return value

    def clear(self, *, reset_stats: bool = False) -> None:
        with self._lock:
            self._latest = None
            if reset_stats:
                self._received = 0
                self._replaced = 0
                self._delivered = 0

    def has_pending(self) -> bool:
        """Return whether the single slot currently contains a frame."""
        with self._lock:
            return self._latest is not None

    def stats(self) -> FrameBufferStats:
        with self._lock:
            return FrameBufferStats(
                received=self._received,
                replaced=self._replaced,
                delivered=self._delivered,
            )
