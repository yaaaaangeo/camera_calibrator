"""
tests/test_vehicle_validation.py
=====================================

Phase C-4 - Real Vehicle Day/Night/Session Validation Framework 검증.

이 저장소에는 실제 차량 데이터가 없으므로, 이 테스트들은 프레임워크
자체(메타데이터 검증/집계/Geometry-Photometric 분리/미검증 disclaimer)만
확인한다 - 실제 "검증 결과"를 지어내서 테스트하지 않는다.
"""

from __future__ import annotations

import pytest

from calibration.residual_stats import compute_residual_stats
from calibration.types import HoldoutEvidenceGate, ValidationResult
from calibration.vehicle_validation import (
    REAL_VEHICLE_NOT_YET_VALIDATED_NOTE,
    SessionGeometrySummary,
    SessionPhotometricSummary,
    VehicleSessionMetadata,
    build_session_geometry_summary,
    build_session_photometric_summary,
    compare_day_night_geometry,
    compare_day_night_photometric,
    format_session_geometry_table,
    format_session_photometric_table,
    real_vehicle_validation_disclaimer,
)
from calibration.windshield.ghost.types import GhostDatasetResult
from calibration.windshield.reflection.types import ReflectionDatasetResult


def _metadata(**overrides) -> VehicleSessionMetadata:
    base = dict(vehicle_id="car01", session_id="s01", day_night="day", data_provenance="synthetic")
    base.update(overrides)
    return VehicleSessionMetadata(**base)


def test_metadata_rejects_unsupported_day_night():
    with pytest.raises(ValueError):
        _metadata(day_night="dusk")


def test_metadata_rejects_unsupported_data_provenance():
    with pytest.raises(ValueError):
        _metadata(data_provenance="made_up")


def test_disclaimer_is_explicit_when_no_real_vehicle_session_exists():
    sessions = [_metadata(data_provenance="synthetic"), _metadata(session_id="s02", data_provenance="unknown")]
    assert real_vehicle_validation_disclaimer(sessions) == REAL_VEHICLE_NOT_YET_VALIDATED_NOTE
    assert "NOT YET VALIDATED" in real_vehicle_validation_disclaimer(sessions)


def test_disclaimer_lists_real_vehicle_sessions_when_present():
    sessions = [_metadata(data_provenance="real_vehicle", vehicle_id="carX", session_id="sess1")]
    disclaimer = real_vehicle_validation_disclaimer(sessions)
    assert "carX/sess1" in disclaimer
    assert disclaimer != REAL_VEHICLE_NOT_YET_VALIDATED_NOTE


def test_geometry_summary_from_missing_validation_result_is_not_evaluated():
    summary = build_session_geometry_summary(_metadata(), None)
    assert summary.evaluated is False
    assert summary.test_rms is None
    assert summary.test_p95 is None


def test_geometry_summary_from_failed_validation_result_is_not_evaluated():
    failed = ValidationResult(success=False, error_message="Train failed")
    summary = build_session_geometry_summary(_metadata(), failed)
    assert summary.evaluated is False


def test_geometry_summary_copies_values_without_recomputation():
    stats = compute_residual_stats([0.1, 0.2, 0.3, 0.4, 0.5])
    gate = HoldoutEvidenceGate(status="sufficient", test_frame_count=10)
    validation = ValidationResult(
        success=True, test_rms=0.42, edge_rms=0.51, test_residual_stats=stats, evidence_gate=gate,
    )
    summary = build_session_geometry_summary(_metadata(), validation)
    assert summary.evaluated is True
    assert summary.test_rms == 0.42
    assert summary.edge_rms == 0.51
    assert summary.test_p95 == stats.p95
    assert summary.test_p99 == stats.p99
    assert summary.evidence_gate_status == "sufficient"


def test_photometric_summary_never_mixes_geometry_fields():
    """SessionPhotometricSummary는 test_rms/edge_rms 같은 geometry 필드를
    아예 갖지 않아야 한다(사용자 스펙 - Geometry/Photometric 절대 분리)."""
    summary = SessionPhotometricSummary(metadata=_metadata())
    assert not hasattr(summary, "test_rms")
    assert not hasattr(summary, "edge_rms")


def test_geometry_summary_never_has_photometric_fields():
    summary = SessionGeometrySummary(metadata=_metadata())
    assert not hasattr(summary, "reflection_mean_strength")
    assert not hasattr(summary, "ghost_mean_strength")


def test_photometric_summary_keeps_ghost_likelihood_separate_from_ghost_strength():
    """Ghost General(No-Reference) Likelihood 모드는 heuristic이므로
    ghost_mean_strength가 아니라 ghost_mean_likelihood에만 채워야 한다
    (STEP 8 semantic fix 1번 원칙을 이 프레임워크에서도 유지)."""
    ghost_likelihood_result = GhostDatasetResult(mode="general_likelihood", mean_ghost_likelihood=0.42)
    summary = build_session_photometric_summary(_metadata(), ghost_result=ghost_likelihood_result)
    assert summary.ghost_mean_likelihood == 0.42
    assert summary.ghost_mean_strength is None


def test_photometric_summary_point_source_ghost_fills_strength_not_likelihood():
    ghost_point_result = GhostDatasetResult(mode="point_source", mean_strength=0.15, p95_strength=0.22)
    summary = build_session_photometric_summary(_metadata(), ghost_result=ghost_point_result)
    assert summary.ghost_mean_strength == 0.15
    assert summary.ghost_mean_likelihood is None


def test_photometric_summary_includes_reflection_when_successful():
    reflection_result = ReflectionDatasetResult(
        mode="reference", success=True, mean_strength=0.05, p95_strength=0.1, coverage=0.3,
    )
    summary = build_session_photometric_summary(_metadata(), reflection_result=reflection_result)
    assert summary.evaluated is True
    assert summary.reflection_mean_strength == 0.05
    assert summary.reflection_mode == "reference"


def test_photometric_summary_excludes_failed_reflection_result():
    failed = ReflectionDatasetResult(mode="reference", success=False, error_message="all pairs invalid")
    summary = build_session_photometric_summary(_metadata(), reflection_result=failed)
    assert summary.reflection_mean_strength is None


def test_compare_day_night_geometry_excludes_unevaluated_and_unknown_sessions():
    day = build_session_geometry_summary(
        _metadata(day_night="day"), ValidationResult(success=True, test_rms=0.3, edge_rms=0.4),
    )
    night = build_session_geometry_summary(
        _metadata(day_night="night"), ValidationResult(success=True, test_rms=0.5, edge_rms=0.6),
    )
    unknown = build_session_geometry_summary(_metadata(day_night="unknown"), None)  # not evaluated
    comparison = compare_day_night_geometry([day, night, unknown])
    assert comparison.day_mean_test_rms == pytest.approx(0.3)
    assert comparison.night_mean_test_rms == pytest.approx(0.5)
    assert len(comparison.day_sessions) == 1
    assert len(comparison.night_sessions) == 1


def test_compare_day_night_photometric_averages_only_evaluated_sessions():
    day = build_session_photometric_summary(
        _metadata(day_night="day"),
        reflection_result=ReflectionDatasetResult(mode="reference", success=True, mean_strength=0.1),
    )
    night = build_session_photometric_summary(
        _metadata(day_night="night"),
        reflection_result=ReflectionDatasetResult(mode="reference", success=True, mean_strength=0.3),
    )
    comparison = compare_day_night_photometric([day, night])
    assert comparison.day_mean_reflection_strength == pytest.approx(0.1)
    assert comparison.night_mean_reflection_strength == pytest.approx(0.3)


def test_format_tables_never_crash_on_empty_input():
    assert "no sessions" in format_session_geometry_table([]).lower()
    assert "no sessions" in format_session_photometric_table([]).lower()


def test_format_geometry_table_includes_provenance_and_evidence_status():
    validation = ValidationResult(
        success=True, test_rms=0.3, edge_rms=0.4,
        evidence_gate=HoldoutEvidenceGate(status="insufficient_evidence"),
    )
    summary = build_session_geometry_summary(_metadata(data_provenance="synthetic"), validation)
    table = format_session_geometry_table([summary])
    assert "synthetic" in table
    assert "insufficient_evidence" in table
