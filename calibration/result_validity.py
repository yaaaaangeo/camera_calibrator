"""
camera_calibrator.calibration.result_validity
==============================================================

Phase D-3 안정화 - 공통 ResultValidity enum.

이 프로젝트 전반에 걸쳐 "성공/실패"를 넘어서는 세부 상태들이 서로 다른
문자열로 흩어져 있다(예: Reflection alignment_status="good"/"warning"/
"invalid", Phase B-6 HoldoutEvidenceGate.status="sufficient"/
"insufficient_evidence"/"not_evaluated"). 이 모듈은 그 상태들을 하나의
공통 enum으로 "설명"할 수 있게 해주는 얇은 read-only 계층이다.

중요한 설계 제약(사용자 스펙 D-3번, "기존 공개 직렬화를 깨지 않는다"):
  - 기존 문자열 필드(alignment_status, HoldoutEvidenceGate.status 등) 자체는
    이름/타입/값을 전혀 바꾸지 않는다 - `.ccproj` 저장 포맷, project_io
    round-trip, 기존 테스트가 읽는 문자열이 전부 그대로 유지된다.
  - `ResultValidity`는 그 문자열들을 표준화해서 "읽는" 새 헬퍼일 뿐, 기존
    필드를 대체하는 새 저장 필드가 아니다(각 결과 dataclass의 `validity`
    property가 매번 이 함수를 호출해 계산할 뿐, 별도로 저장/직렬화되지
    않는다).
  - 이번 라운드에서 프로젝트 전체(모든 서브시스템)의 상태 문자열을 이
    enum으로 강제 통일하는 전면 마이그레이션은 하지 않는다(리스크 대비
    효과가 낮고, "한 번에 크게 갈아엎지 않는다"는 이번 작업 전체의 원칙에
    위배된다) - 대신 Phase B-6(HoldoutEvidenceGate)과 Reflection
    Alignment처럼 이미 명확한 상태 문자열을 가진 대표적인 곳에만
    read-only 변환 프로퍼티를 추가해, 다른 서브시스템도 향후 같은 패턴을
    점진적으로 따라올 수 있게 한다.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional


class ResultValidity(Enum):
    """FAILED(성공/실패)와는 다른 축의 세부 상태. NOT_RUN과
    INSUFFICIENT_EVIDENCE를 서로 다른 값으로 명확히 구분하는 것이 핵심
    목표다(사용자 스펙 - "FAILED/INVALID/INSUFFICIENT DATA/NOT RUN을
    명확히 구분한다")."""
    VALID = "valid"
    WARNING = "warning"
    INVALID = "invalid"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NOT_RUN = "not_run"


# 기존 서브시스템이 이미 쓰고 있는 문자열 -> ResultValidity 매핑. 새 상태
# 문자열을 추가로 매핑해야 하면 여기 한 곳만 갱신한다(중복 매핑 테이블을
# 여기저기 만들지 않는다).
_LEGACY_STATUS_TO_VALIDITY: dict[str, ResultValidity] = {
    # Reflection alignment (calibration/windshield/reflection/alignment.py)
    "good": ResultValidity.VALID,
    "warning": ResultValidity.WARNING,
    "invalid": ResultValidity.INVALID,
    # Phase B-6 Hold-out Evidence Gate (calibration/holdout_evidence.py)
    "sufficient": ResultValidity.VALID,
    "insufficient_evidence": ResultValidity.INSUFFICIENT_EVIDENCE,
    "not_evaluated": ResultValidity.NOT_RUN,
    # 범용
    "not_run": ResultValidity.NOT_RUN,
}


def validity_from_legacy_status(status: Optional[str]) -> ResultValidity:
    """기존 서브시스템의 문자열 status를 `ResultValidity`로 변환한다.

    `None`은 "아직 실행되지 않음"으로 간주해 `NOT_RUN`을 반환한다. 매핑에
    없는 문자열은 조용히 기본값으로 떨어뜨리지 않고 `ValueError`를 낸다 -
    알 수 없는 새 상태 문자열이 생기면(예: 다른 서브시스템의 오타나 새
    status 값 추가) 여기서 바로 드러나야, 매핑 누락이 "이상하게 항상
    NOT_RUN으로 보이는" 조용한 버그로 남지 않는다.
    """
    if status is None:
        return ResultValidity.NOT_RUN
    key = status.lower()
    if key not in _LEGACY_STATUS_TO_VALIDITY:
        raise ValueError(
            f"알 수 없는 legacy status 문자열: {status!r} - "
            "calibration/result_validity.py의 _LEGACY_STATUS_TO_VALIDITY 매핑에 추가해야 합니다."
        )
    return _LEGACY_STATUS_TO_VALIDITY[key]
