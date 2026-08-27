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
        # 코어를 전부 다 쓰면(특히 detect_dataset처럼 수백 장을 병렬 처리할 때)
        # GUI 프로세스가 스케줄링을 못 받아 OS가 "응답 없음"으로 오판하는
        # 실사용자 버그가 있었다. 코어 하나는 항상 GUI/OS 몫으로 남겨둔다.
        cpu = os.cpu_count() or 1
        return max(1, min(task_count, cpu - 1 if cpu > 1 else 1))
    return max(1, min(task_count, int(n_jobs)))
