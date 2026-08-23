"""
Small performance helpers shared by expensive calibration analyses.
"""

from __future__ import annotations

import os


def resolve_worker_count(n_jobs: int | None, task_count: int) -> int:
    """Normalize user-facing job counts to a safe worker count.

    `1` keeps the historical sequential path. `0` means "auto".
    """
    if task_count <= 1:
        return 1
    if n_jobs is None:
        return 1
    if n_jobs == 0:
        return max(1, min(task_count, os.cpu_count() or 1))
    return max(1, min(task_count, int(n_jobs)))
