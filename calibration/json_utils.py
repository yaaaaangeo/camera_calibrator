"""
camera_calibrator.calibration.json_utils
=============================================

dataclass(+ numpy 배열, Enum, datetime이 섞인 구조)를 JSON으로 안전하게
바꾸는 공용 유틸리티. calibration/project_io.py(.ccproj 저장/불러오기)와
export/json_export.py(외부 도구용 JSON export) 둘 다 이 함수를 쓴다 -
전에는 project_io.py 안에 똑같은 로직이 들어있었는데, 두 번째로 필요한
곳이 생겨서 공용 모듈로 뽑아냈다 (하나만 고치고 다른 곳은 안 고치는 실수를
막기 위함).

두 용도가 numpy 배열을 표현하는 방식이 다르다:
- .ccproj(project_io.py)는 "나중에 정확히 원래 dtype으로 복원"해야 해서
  {"__ndarray__": True, "dtype": ..., "data": [...]} 래퍼를 쓴다.
- 외부 JSON export(export/json_export.py)는 이 프로젝트를 모르는 다른
  도구/스크립트가 읽을 걸 상정하므로, 그런 래퍼 없이 그냥 중첩 리스트로
  평평하게 편다 - 훨씬 널리 읽히는 형태다.

ndarray_wrapper 인자로 둘을 고른다.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
from typing import Any

import numpy as np


def json_safe(obj: Any, ndarray_wrapper: bool = True) -> Any:
    """dataclasses.asdict() 결과 등을 재귀적으로 순회하며 numpy/Enum/datetime을
    JSON 표준 타입으로 바꾼다. 필드 이름에 의존하지 않는 범용 변환이라
    구조가 바뀌어도(필드 추가 등) 이 함수는 수정할 필요가 없다.

    dataclass 인스턴스(중첩된 RegionalError, RadialErrorProfile 등)를 직접
    받는 경우도 처리한다 - 호출부가 미리 dataclasses.asdict()로 dict화해서
    넘겨줄 필요 없이, 이 함수 하나로 임의 깊이의 dataclass 중첩을 다 풀 수 있다.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return json_safe(
            {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}, ndarray_wrapper
        )
    if isinstance(obj, np.ndarray):
        if ndarray_wrapper:
            return {"__ndarray__": True, "dtype": str(obj.dtype), "data": obj.tolist()}
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, datetime):
        if ndarray_wrapper:
            return {"__datetime__": True, "iso": obj.isoformat()}
        return obj.isoformat()
    if isinstance(obj, dict):
        return {
            (k.value if hasattr(k, "value") and isinstance(k, str) else k): json_safe(v, ndarray_wrapper)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [json_safe(v, ndarray_wrapper) for v in obj]
    if hasattr(obj, "value") and isinstance(obj, str):
        # str, Enum 서브클래스(CameraModelType 등) - .value로 순수 문자열화.
        return obj.value
    return obj
