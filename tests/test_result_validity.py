"""
tests/test_result_validity.py
==================================

Phase D-3 - 공통 ResultValidity enum 검증.

기존 문자열 status 필드 자체(직렬화 대상)는 절대 바뀌면 안 되고,
`.validity` read-only 프로퍼티만 새로 추가됐는지 확인한다.
"""

from __future__ import annotations

import pytest

from calibration.result_validity import ResultValidity, validity_from_legacy_status
from calibration.types import HoldoutEvidenceGate


def test_validity_from_legacy_status_maps_known_strings():
    assert validity_from_legacy_status("good") is ResultValidity.VALID
    assert validity_from_legacy_status("warning") is ResultValidity.WARNING
    assert validity_from_legacy_status("invalid") is ResultValidity.INVALID
    assert validity_from_legacy_status("sufficient") is ResultValidity.VALID
    assert validity_from_legacy_status("insufficient_evidence") is ResultValidity.INSUFFICIENT_EVIDENCE
    assert validity_from_legacy_status("not_evaluated") is ResultValidity.NOT_RUN
    assert validity_from_legacy_status("not_run") is ResultValidity.NOT_RUN


def test_validity_from_legacy_status_is_case_insensitive():
    assert validity_from_legacy_status("GOOD") is ResultValidity.VALID


def test_validity_from_legacy_status_none_is_not_run():
    assert validity_from_legacy_status(None) is ResultValidity.NOT_RUN


def test_validity_from_legacy_status_rejects_unknown_string_instead_of_silently_defaulting():
    """매핑에 없는 새 문자열은 조용히 기본값으로 떨어지면 안 된다 - 매핑
    누락을 즉시 드러내야 한다."""
    with pytest.raises(ValueError):
        validity_from_legacy_status("some_new_status_nobody_mapped_yet")


def test_holdout_evidence_gate_validity_property_does_not_change_status_field():
    gate = HoldoutEvidenceGate(status="insufficient_evidence")
    assert gate.validity is ResultValidity.INSUFFICIENT_EVIDENCE
    # 기존 문자열 필드 자체는 그대로여야 한다(직렬화 호환성).
    assert gate.status == "insufficient_evidence"
    assert isinstance(gate.status, str)


def test_holdout_evidence_gate_sufficient_maps_to_valid():
    gate = HoldoutEvidenceGate(status="sufficient")
    assert gate.validity is ResultValidity.VALID


def test_holdout_evidence_gate_not_evaluated_maps_to_not_run():
    gate = HoldoutEvidenceGate(status="not_evaluated")
    assert gate.validity is ResultValidity.NOT_RUN


def test_alignment_result_validity_property_does_not_change_status_field():
    from calibration.windshield.reflection.alignment import AlignmentResult
    import numpy as np

    result = AlignmentResult(
        aligned_reference=np.zeros((2, 2)), warp_matrix=np.eye(2, 3),
        score=0.99, error_px=0.1, status="warning", method="translation",
    )
    assert result.validity is ResultValidity.WARNING
    assert result.status == "warning"
