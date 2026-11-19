from __future__ import annotations

import threading

import numpy as np

from calibration.latest_frame import LatestFrameBuffer


def test_latest_frame_replaces_stale_frames_without_queue_growth():
    buffer = LatestFrameBuffer()
    for index in range(100):
        buffer.put(np.array([index]), float(index))

    frame, timestamp = buffer.take()
    assert int(frame[0]) == 99
    assert timestamp == 99.0
    assert buffer.take() is None

    stats = buffer.stats()
    assert stats.received == 100
    assert stats.replaced == 99
    assert stats.delivered == 1


def test_latest_frame_is_thread_safe_for_live_producer_consumer():
    buffer = LatestFrameBuffer()

    def produce():
        for index in range(1000):
            buffer.put(np.array([index]), float(index))

    thread = threading.Thread(target=produce)
    thread.start()
    while thread.is_alive():
        buffer.take()
    thread.join()

    stats = buffer.stats()
    assert stats.received == 1000
    assert stats.delivered <= stats.received
    assert stats.replaced + stats.delivered <= stats.received


def test_clear_discards_pending_frame_and_can_reset_stats():
    buffer = LatestFrameBuffer()
    buffer.put(np.zeros(1), 1.0)
    buffer.clear(reset_stats=True)

    assert buffer.take() is None
    assert buffer.stats().received == 0
