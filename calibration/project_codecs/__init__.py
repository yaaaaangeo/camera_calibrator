"""
camera_calibrator.calibration.project_codecs
==============================================================

Phase D-2 안정화 - `calibration/project_io.py`(God file)를 도메인별
디코더(dict -> dataclass) 모듈로 분리한다.

    intrinsic.py  - Camera Intrinsic Calibration/Validation/Dataset 관련
    windshield.py - Windshield Calibration 관련
    reflection.py - Reflection 평가 관련
    ghost.py      - Ghost/Double Image 평가 관련
    common.py     - 위 네 모듈이 공유하는 최소 유틸(_arr/_dt)

`calibration/project_io.py`는 이 패키지의 함수들을 그대로 재-export해서
오케스트레이션(`project_to_dict`/`project_from_dict`/`migrate_v1_to_v2`/
`save_project`/`load_project`)만 담당한다 - 기존에
`from calibration.project_io import _windshield_calibration_result_from_dict`
처럼 private 함수를 직접 가져다 쓰던 기존 테스트/코드가 전혀 바뀌지 않고
그대로 동작한다(공개 API를 이유 없이 바꾸지 않는다는 원칙).

각 모듈의 함수 로직 자체는 이 분리 작업에서 단 한 줄도 바뀌지 않았다 -
순수한 파일 이동(+ import 경로 조정)만 수행했다.
"""

from __future__ import annotations
