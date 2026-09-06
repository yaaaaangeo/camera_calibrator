"""
camera_calibrator.calibration.project_codecs.common
==============================================================

Phase D-2 안정화 - 모든 도메인 codec(intrinsic/windshield/reflection/ghost)이
공유하는 최소 유틸. `calibration/project_io.py`에서 그대로 옮겨왔다(로직
변경 없음).
"""

from __future__ import annotations

from datetime import datetime

import numpy as np


def _arr(d, dtype) -> np.ndarray | None:
    """_json_safe가 만든 {"__ndarray__": True, "data": [...]} 구조 -> np.ndarray.
    구버전 파일 호환을 위해 그냥 리스트로 저장된 경우도 받아준다.
    """
    if d is None:
        return None
    if isinstance(d, dict) and d.get("__ndarray__"):
        return np.array(d["data"], dtype=dtype)
    return np.array(d, dtype=dtype)


def _dt(d) -> datetime:
    if d is None:
        return datetime.now()
    if isinstance(d, dict) and d.get("__datetime__"):
        return datetime.fromisoformat(d["iso"])
    return datetime.fromisoformat(d)
