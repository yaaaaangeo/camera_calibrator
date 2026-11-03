"""Bounded retry of Windows replace sharing races; persistent errors still fail."""
import os
import time
from automation.run_manifest import atomic_write_json


def write_json(path, data):
    for attempt in range(5):
        try:
            return atomic_write_json(path, data)
        except PermissionError as exc:
            if os.name != 'nt' or getattr(exc, 'winerror', None) not in (5, 32, 33) or attempt == 4:
                raise
            time.sleep(.01 * 2 ** attempt)
