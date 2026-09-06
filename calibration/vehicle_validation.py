"""
camera_calibrator.calibration.vehicle_validation
==============================================================

Phase C-4 - Real Vehicle Day/Night/Session Validation Framework.

이 저장소에는 실제 차량에서 촬영한 캡처 데이터셋이 없다. 이 모듈은 그런
데이터가 "생기면" 곧바로 쓸 수 있는 메타데이터/집계/비교 프레임워크만
제공할 뿐, 어떤 검증 결과도 스스로 만들어내지 않는다 - 절대 실측 없이
"검증됨"을 주장하지 않기 위해, 모든 세션에 `data_provenance`
("real_vehicle" | "synthetic" | "unknown")를 명시적으로 붙이고,
`real_vehicle_validation_disclaimer()`가 real_vehicle 세션이 하나도 없으면
그 사실을 항상 명시적으로 보고한다.

핵심 설계 원칙(이 프로젝트 전체의 절대 원칙, 여기서도 그대로 지킨다):
Geometry 지표(Hold-out RMS/P95/P99/Regional/Edge/Stability)와 Photometric
지표(Reflection/Glare/Saturation/Ghost)는 절대 하나의 점수로 합치지 않는다.
`SessionGeometrySummary`/`SessionPhotometricSummary`는 서로 다른
데이터클래스이고, 비교 함수도 두 축을 독립적으로만 다룬다.

이 모듈은 이미 계산된 `ValidationResult`/`ReflectionDatasetResult`/
`GhostDatasetResult`를 "집계"할 뿐, Hold-out/Reflection/Ghost 평가 로직
자체를 재구현하지 않는다(calibration/validation.py,
calibration/windshield/reflection/evaluator.py,
calibration/windshield/ghost/evaluator.py를 그대로 재사용).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from calibration.types import ValidationResult
from calibration.windshield.ghost.types import GhostDatasetResult
from calibration.windshield.reflection.types import ReflectionDatasetResult

SUPPORTED_DAY_NIGHT = ("day", "night", "unknown")
SUPPORTED_DATA_PROVENANCE = ("real_vehicle", "synthetic", "unknown")

REAL_VEHICLE_NOT_YET_VALIDATED_NOTE = (
    "NOT YET VALIDATED ON REAL VEHICLE - this repository does not contain any "
    "real-vehicle capture session yet. All numbers below (if any) come from "
    "synthetic or bench/test data only."
)


@dataclass
class VehicleSessionMetadata:
    """한 촬영 세션을 식별/설명하는 메타데이터(사용자 스펙 C-4번). 이
    프레임워크가 다루는 최소 단위다 - 한 세션 = 한 차량 x 한 시간대(day/
    night) x 한 windshield 상태의 데이터 묶음."""
    vehicle_id: str
    session_id: str
    day_night: str = "unknown"  # "day" | "night" | "unknown"
    resolution: Optional[tuple[int, int]] = None
    camera_mode: Optional[str] = None
    exposure_mode: Optional[str] = None
    windshield_state: Optional[str] = None  # 예: "clean", "dirty", "wipers_on", "tinted"
    weather_note: Optional[str] = None
    timestamp: Optional[str] = None  # ISO 8601 문자열 권장
    notes: Optional[str] = None
    # 이 세션이 실제 차량 캡처인지, synthetic/test fixture인지 항상 명시
    # (사용자 스펙 - "실차 데이터가 없으면 결과를 지어내지 않는다"를
    # 데이터 모델 레벨에서 강제하기 위한 필드).
    data_provenance: str = "unknown"

    def __post_init__(self) -> None:
        if self.day_night not in SUPPORTED_DAY_NIGHT:
            raise ValueError(
                f"지원하지 않는 day_night 값: {self.day_night!r} (지원: {SUPPORTED_DAY_NIGHT})"
            )
        if self.data_provenance not in SUPPORTED_DATA_PROVENANCE:
            raise ValueError(
                f"지원하지 않는 data_provenance 값: {self.data_provenance!r} "
                f"(지원: {SUPPORTED_DATA_PROVENANCE})"
            )


@dataclass
class SessionGeometrySummary:
    """한 세션의 Geometry(Hold-out) 지표만 담는다 - Photometric과 절대
    섞지 않는다. `validation_result`가 없거나 실패했으면 모든 지표가
    None으로 남는다(0으로 채우지 않음 - "측정 안 됨"과 "0으로 측정됨"을
    구분, Phase A-4/B-6과 동일한 원칙)."""
    metadata: VehicleSessionMetadata
    test_rms: Optional[float] = None
    test_p95: Optional[float] = None
    test_p99: Optional[float] = None
    edge_rms: Optional[float] = None
    evidence_gate_status: Optional[str] = None
    evaluated: bool = False


@dataclass
class SessionPhotometricSummary:
    """한 세션의 Photometric(Reflection/Ghost) 지표만 담는다 - Geometry와
    절대 섞지 않는다. Reflection/Ghost 각각 결과가 없으면 그쪽 필드만
    None으로 남는다(둘 다 없어도 되고, 하나만 있어도 된다)."""
    metadata: VehicleSessionMetadata
    reflection_mean_strength: Optional[float] = None
    reflection_p95_strength: Optional[float] = None
    reflection_coverage: Optional[float] = None
    reflection_mode: Optional[str] = None
    ghost_mean_strength: Optional[float] = None
    ghost_p95_strength: Optional[float] = None
    ghost_mean_likelihood: Optional[float] = None
    ghost_num_valid_frames: Optional[int] = None
    ghost_mode: Optional[str] = None
    evaluated: bool = False


def build_session_geometry_summary(
    metadata: VehicleSessionMetadata,
    validation_result: Optional[ValidationResult],
) -> SessionGeometrySummary:
    """이미 계산된 `ValidationResult`에서 값을 그대로 옮겨 담는다 - 이
    함수는 어떤 새 숫자도 계산하지 않는다(재계산 금지, calibration/
    validation.py의 결과를 그대로 신뢰)."""
    if validation_result is None or not validation_result.success:
        return SessionGeometrySummary(metadata=metadata, evaluated=False)

    test_p95 = validation_result.test_residual_stats.p95 if validation_result.test_residual_stats else None
    test_p99 = validation_result.test_residual_stats.p99 if validation_result.test_residual_stats else None
    evidence_status = (
        validation_result.evidence_gate.status if validation_result.evidence_gate else None
    )
    return SessionGeometrySummary(
        metadata=metadata,
        test_rms=validation_result.test_rms,
        test_p95=test_p95,
        test_p99=test_p99,
        edge_rms=validation_result.edge_rms,
        evidence_gate_status=evidence_status,
        evaluated=True,
    )


def build_session_photometric_summary(
    metadata: VehicleSessionMetadata,
    reflection_result: Optional[ReflectionDatasetResult] = None,
    ghost_result: Optional[GhostDatasetResult] = None,
) -> SessionPhotometricSummary:
    """이미 계산된 `ReflectionDatasetResult`/`GhostDatasetResult`에서 값을
    그대로 옮겨 담는다 - 재계산하지 않는다. Ghost의 General(No-Reference)
    Likelihood 모드는 `ghost_mean_likelihood`에만 채우고
    `ghost_mean_strength`는 절대 채우지 않는다(STEP 8 semantic fix 1번,
    "Ghost Likelihood != Ghost Strength" 원칙을 여기서도 유지)."""
    summary = SessionPhotometricSummary(metadata=metadata)

    if reflection_result is not None and reflection_result.success:
        summary.reflection_mean_strength = reflection_result.mean_strength
        summary.reflection_p95_strength = reflection_result.p95_strength
        summary.reflection_coverage = reflection_result.coverage
        summary.reflection_mode = reflection_result.mode
        summary.evaluated = True

    if ghost_result is not None and ghost_result.mode != "general_likelihood":
        if ghost_result.mean_strength is not None:
            summary.ghost_mean_strength = ghost_result.mean_strength
            summary.ghost_p95_strength = ghost_result.p95_strength
            summary.ghost_mode = ghost_result.mode
            summary.evaluated = True
    elif ghost_result is not None and ghost_result.mode == "general_likelihood":
        summary.ghost_mean_likelihood = ghost_result.mean_ghost_likelihood
        summary.ghost_mode = ghost_result.mode
        summary.evaluated = True

    if ghost_result is not None:
        summary.ghost_num_valid_frames = ghost_result.num_valid_frames

    return summary


@dataclass
class DayNightGeometryComparison:
    day_sessions: list[SessionGeometrySummary] = field(default_factory=list)
    night_sessions: list[SessionGeometrySummary] = field(default_factory=list)
    day_mean_test_rms: Optional[float] = None
    night_mean_test_rms: Optional[float] = None
    day_mean_edge_rms: Optional[float] = None
    night_mean_edge_rms: Optional[float] = None


@dataclass
class DayNightPhotometricComparison:
    day_sessions: list[SessionPhotometricSummary] = field(default_factory=list)
    night_sessions: list[SessionPhotometricSummary] = field(default_factory=list)
    day_mean_reflection_strength: Optional[float] = None
    night_mean_reflection_strength: Optional[float] = None
    day_mean_ghost_strength: Optional[float] = None
    night_mean_ghost_strength: Optional[float] = None


def _mean_or_none(values: list[Optional[float]]) -> Optional[float]:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def compare_day_night_geometry(sessions: list[SessionGeometrySummary]) -> DayNightGeometryComparison:
    """세션들을 day/night로 나눠 Geometry 지표 평균을 비교한다. 세션이
    평가되지 않았거나(evaluated=False) day_night가 "unknown"이면 비교에서
    제외한다(억지로 한쪽에 끼워 넣지 않음)."""
    day = [s for s in sessions if s.evaluated and s.metadata.day_night == "day"]
    night = [s for s in sessions if s.evaluated and s.metadata.day_night == "night"]
    return DayNightGeometryComparison(
        day_sessions=day,
        night_sessions=night,
        day_mean_test_rms=_mean_or_none([s.test_rms for s in day]),
        night_mean_test_rms=_mean_or_none([s.test_rms for s in night]),
        day_mean_edge_rms=_mean_or_none([s.edge_rms for s in day]),
        night_mean_edge_rms=_mean_or_none([s.edge_rms for s in night]),
    )


def compare_day_night_photometric(sessions: list[SessionPhotometricSummary]) -> DayNightPhotometricComparison:
    day = [s for s in sessions if s.evaluated and s.metadata.day_night == "day"]
    night = [s for s in sessions if s.evaluated and s.metadata.day_night == "night"]
    return DayNightPhotometricComparison(
        day_sessions=day,
        night_sessions=night,
        day_mean_reflection_strength=_mean_or_none([s.reflection_mean_strength for s in day]),
        night_mean_reflection_strength=_mean_or_none([s.reflection_mean_strength for s in night]),
        day_mean_ghost_strength=_mean_or_none([s.ghost_mean_strength for s in day]),
        night_mean_ghost_strength=_mean_or_none([s.ghost_mean_strength for s in night]),
    )


def real_vehicle_validation_disclaimer(sessions: list[VehicleSessionMetadata]) -> str:
    """`sessions` 중 하나라도 `data_provenance == "real_vehicle"`이면 그
    세션 id들을 알려주는 짧은 문구를, 하나도 없으면 표준
    `REAL_VEHICLE_NOT_YET_VALIDATED_NOTE`를 반환한다. 보고서/README가
    실측 없이 "실차 검증됨"이라고 말하지 않도록 이 함수를 항상 거치게
    한다(사용자 스펙 - 절대 실차 검증 결과를 지어내지 않는다)."""
    real_vehicle = [s for s in sessions if s.data_provenance == "real_vehicle"]
    if not real_vehicle:
        return REAL_VEHICLE_NOT_YET_VALIDATED_NOTE
    ids = ", ".join(f"{s.vehicle_id}/{s.session_id}" for s in real_vehicle)
    return f"Real-vehicle sessions present: {ids}. (Framework does not itself verify capture authenticity.)"


def format_session_geometry_table(sessions: list[SessionGeometrySummary]) -> str:
    """콘솔/로그용 세션별 Geometry 표. Photometric 지표는 이 표에 절대
    섞이지 않는다."""
    if not sessions:
        return "Vehicle session geometry comparison: no sessions."

    def fmt(v: Optional[float]) -> str:
        return f"{v:.3f}" if v is not None else "N/A"

    lines = [
        "Vehicle Session Geometry Comparison (Hold-out; fixed intrinsics):",
        f"{'Vehicle':<12}{'Session':<12}{'Day/Night':<10}{'Provenance':<14}"
        f"{'TestRMS':>10}{'P95':>10}{'P99':>10}{'EdgeRMS':>10}{'Evidence':>22}",
    ]
    for s in sessions:
        m = s.metadata
        lines.append(
            f"{m.vehicle_id:<12}{m.session_id:<12}{m.day_night:<10}{m.data_provenance:<14}"
            f"{fmt(s.test_rms):>10}{fmt(s.test_p95):>10}{fmt(s.test_p99):>10}{fmt(s.edge_rms):>10}"
            f"{(s.evidence_gate_status or 'not_evaluated'):>22}"
        )
    return "\n".join(lines)


def format_session_photometric_table(sessions: list[SessionPhotometricSummary]) -> str:
    """콘솔/로그용 세션별 Photometric 표. Geometry 지표는 이 표에 절대
    섞이지 않는다."""
    if not sessions:
        return "Vehicle session photometric comparison: no sessions."

    def fmt(v: Optional[float]) -> str:
        return f"{v:.3f}" if v is not None else "N/A"

    lines = [
        "Vehicle Session Photometric Comparison (Reflection/Ghost; heuristic where no-reference):",
        f"{'Vehicle':<12}{'Session':<12}{'Day/Night':<10}{'Provenance':<14}"
        f"{'ReflMean':>10}{'ReflP95':>10}{'GhostMean':>10}{'GhostLik':>10}",
    ]
    for s in sessions:
        m = s.metadata
        lines.append(
            f"{m.vehicle_id:<12}{m.session_id:<12}{m.day_night:<10}{m.data_provenance:<14}"
            f"{fmt(s.reflection_mean_strength):>10}{fmt(s.reflection_p95_strength):>10}"
            f"{fmt(s.ghost_mean_strength):>10}{fmt(s.ghost_mean_likelihood):>10}"
        )
    return "\n".join(lines)
