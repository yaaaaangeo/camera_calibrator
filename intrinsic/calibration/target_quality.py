"""
camera_calibrator.calibration.target_quality
================================================

설계 문서 3-2번 - Calibration Target(보드) 품질 검사.

detector.py가 검출 시점에 계산해 DetectionResult에 채워주는 값들
(num_corners, board_area_ratio, board_center_px, board_tilt_deg,
corner_confidence, min_edge_margin_px)을 해석해서 "이 검출 결과를 그대로
캘리브레이션에 써도 괜찮은가"에 대한 경고를 만든다.

이 모듈은 계산을 하지 않는다(계산은 detector.py의 몫) - 오직 임계값 판정만
한다. 그래야 "왜 검출 시점에 이미 계산해둔 값과 여기서 판정한 결과가
다르지?" 같은 혼란이 생기지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from calibration.types import DetectionResult, PatternConfig

# ---------------------------------------------------------------------------
# 판정 기준값
# ---------------------------------------------------------------------------

MIN_RECOMMENDED_CORNERS = 6      # detector.py와 동일한 기준 재사용(문서 일관성)
_CORNER_CONFIDENCE_WARN = 0.5    # 이론상 코너의 50% 미만만 검출되면 경고

_EDGE_MARGIN_WARN_PX = 15.0      # 코너가 이미지 경계에서 이보다 가까우면 경고
_EDGE_MARGIN_CUTOFF_PX = 3.0     # 이보다 가까우면 "잘렸을 가능성" 격상

_TILT_WARN_DEG = 55.0            # minAreaRect 각도(대략 in-plane 회전) 절대값이 이보다 크면 경고
                                  # (참고: 이 값은 "완전히 옆으로 누운" 정도의 극단치를 잡기 위함
                                  #  - 일반적인 원근 기울기는 board_area_ratio/hull 형태로 더 잘 보임)

# board_area_ratio 선호 구간 - frame_quality.py의 _area_preference_score와
# 정확히 같은 값을 쓴다 (같은 "적정 크기" 기준이 두 모듈에서 다르면 혼란스러움)
AREA_SWEET_LOW = 0.10
AREA_SWEET_HIGH = 0.55
AREA_TOO_SMALL_HARD = 0.02   # 이보다 작으면 경고가 아니라 사실상 무의미한 검출로 봄
AREA_TOO_LARGE_HARD = 0.85   # 이보다 크면 보드가 프레임을 거의 다 채워 잘릴 위험이 매우 큼


class TargetQualitySeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass
class TargetQualityIssue:
    code: str
    severity: TargetQualitySeverity
    message: str


@dataclass
class TargetQualityReport:
    image_id: str
    issues: list[TargetQualityIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == TargetQualitySeverity.ERROR for i in self.issues)


def _max_possible_corners(pattern: PatternConfig) -> int:
    return max(1, (pattern.squares_x - 1) * (pattern.squares_y - 1))


def evaluate_target_quality(
    detection: DetectionResult | None,
    pattern: PatternConfig,
) -> TargetQualityReport:
    """DetectionResult 하나를 검사해 경고 목록을 만든다."""
    image_id = detection.image_id if detection else "unknown"
    issues: list[TargetQualityIssue] = []

    if detection is None or not detection.success:
        issues.append(TargetQualityIssue(
            "detection_failed", TargetQualitySeverity.ERROR,
            (detection.failure_reason if detection else None) or "검출 실패 (원인 불명).",
        ))
        return TargetQualityReport(image_id=image_id, issues=issues)

    max_corners = _max_possible_corners(pattern)
    if detection.num_corners < MIN_RECOMMENDED_CORNERS:
        issues.append(TargetQualityIssue(
            "too_few_corners", TargetQualitySeverity.WARNING,
            f"검출된 코너 {detection.num_corners}개 - 권장 최소치"
            f"({MIN_RECOMMENDED_CORNERS}개)보다 적습니다.",
        ))

    if detection.corner_confidence is not None and detection.corner_confidence < _CORNER_CONFIDENCE_WARN:
        issues.append(TargetQualityIssue(
            "low_corner_confidence", TargetQualitySeverity.WARNING,
            f"코너 검출 신뢰도 {detection.corner_confidence:.0%} "
            f"(이론상 최대 {max_corners}개 중 {detection.num_corners}개만 검출) - "
            "보드가 부분적으로 가려졌거나 각도가 심할 수 있습니다.",
        ))

    if detection.min_edge_margin_px is not None:
        if detection.min_edge_margin_px < _EDGE_MARGIN_CUTOFF_PX or detection.likely_cut_off:
            issues.append(TargetQualityIssue(
                "board_likely_cut_off", TargetQualitySeverity.WARNING,
                f"코너가 이미지 경계에서 {detection.min_edge_margin_px:.1f}px 떨어져 있어 "
                "보드가 프레임 밖으로 잘렸을 가능성이 있습니다.",
            ))
        elif detection.min_edge_margin_px < _EDGE_MARGIN_WARN_PX:
            issues.append(TargetQualityIssue(
                "corner_near_edge", TargetQualitySeverity.WARNING,
                f"일부 코너가 이미지 경계에서 {detection.min_edge_margin_px:.1f}px밖에 "
                "떨어져 있지 않습니다 - 여유를 두고 촬영하는 것을 권장합니다.",
            ))

    if detection.board_area_ratio is not None:
        ratio = detection.board_area_ratio
        if ratio < AREA_TOO_SMALL_HARD:
            issues.append(TargetQualityIssue(
                "board_too_small", TargetQualitySeverity.WARNING,
                f"보드 면적 비율 {ratio:.1%} - 너무 작게 찍혀 코너 정밀도가 떨어질 수 있습니다"
                " (더 가까이서 촬영 권장).",
            ))
        elif ratio > AREA_TOO_LARGE_HARD:
            issues.append(TargetQualityIssue(
                "board_too_large", TargetQualitySeverity.WARNING,
                f"보드 면적 비율 {ratio:.1%} - 너무 크게 찍혀 보드 일부가 잘렸을 위험이 큽니다"
                " (더 멀리서 촬영 권장).",
            ))
        elif not (AREA_SWEET_LOW <= ratio <= AREA_SWEET_HIGH):
            issues.append(TargetQualityIssue(
                "board_area_suboptimal", TargetQualitySeverity.WARNING,
                f"보드 면적 비율 {ratio:.1%} - 권장 구간"
                f"({AREA_SWEET_LOW:.0%}~{AREA_SWEET_HIGH:.0%}) 밖입니다.",
            ))

    if detection.board_tilt_deg is not None and abs(detection.board_tilt_deg) > _TILT_WARN_DEG:
        issues.append(TargetQualityIssue(
            "board_tilt_extreme", TargetQualitySeverity.WARNING,
            f"보드 기울기 추정치 {detection.board_tilt_deg:.1f}\u00b0 - 매우 큰 각도로 "
            "촬영되어 코너 검출/재투영 정밀도가 떨어질 수 있습니다.",
        ))

    return TargetQualityReport(image_id=image_id, issues=issues)


def evaluate_dataset_target_quality(
    frames, pattern: PatternConfig
) -> dict[str, TargetQualityReport]:
    """Dataset.frames(또는 임의의 Frame 이터러블)에 대해 일괄 실행."""
    return {
        f.image_info.image_id: evaluate_target_quality(f.detection, pattern)
        for f in frames
    }


def format_target_quality_summary(reports: dict[str, TargetQualityReport]) -> str:
    flagged = {k: r for k, r in reports.items() if r.issues}
    if not flagged:
        return "Target 품질 경고 없음."
    lines = [f"Target 품질 경고 {len(flagged)}장:"]
    for image_id, report in flagged.items():
        for issue in report.issues:
            mark = "\u2716" if issue.severity == TargetQualitySeverity.ERROR else "\u26a0"
            lines.append(f"  {mark} [{image_id}] {issue.message}")
    return "\n".join(lines)
