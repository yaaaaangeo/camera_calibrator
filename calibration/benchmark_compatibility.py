"""
camera_calibrator.calibration.benchmark_compatibility
=====================================================

Reference / Candidate calibration을 같은 validation dataset에서 공정하게
비교하기 전에 통과해야 하는 compatibility 검사.

포맷 로딩(calibration_io.py)은 "파일을 내부 표준 객체로 읽는 것"까지만 담당하고,
이 모듈은 "두 객체를 같은 benchmark pipeline에 넣어도 되는가"를 검사한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from calibration.calibration_io import StandardCalibration
from calibration.types import CameraModelType


class CompatibilitySeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass
class CompatibilityIssue:
    code: str
    severity: CompatibilitySeverity
    side: str
    message: str


@dataclass
class CalibrationCompatibilityReport:
    reference_label: str
    candidate_label: str
    issues: list[CompatibilityIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == CompatibilitySeverity.ERROR for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == CompatibilitySeverity.WARNING for i in self.issues)

    @property
    def compatible(self) -> bool:
        return not self.has_errors

    @property
    def status(self) -> str:
        if self.has_errors:
            return "incompatible"
        if self.has_warnings:
            return "compatible_with_warnings"
        return "compatible"

    def format(self) -> str:
        if not self.issues:
            return "Reference/Candidate calibration compatibility: OK"
        lines = [f"Reference/Candidate calibration compatibility: {self.status}"]
        for issue in self.issues:
            prefix = "ERROR" if issue.severity == CompatibilitySeverity.ERROR else "WARNING"
            lines.append(f"- [{prefix}] {issue.side}: {issue.message}")
        return "\n".join(lines)


_DISTORTION_MODEL_FOR_MODEL = {
    CameraModelType.PINHOLE: {"none", "no_distortion", "plumb_bob", "radtan"},
    CameraModelType.EXTENDED_PINHOLE: {"plumb_bob", "radtan", "rational_polynomial", "radtan8"},
    CameraModelType.FISHEYE: {"equidistant", "fisheye"},
}

_EXTENDED_COUNTS = {4, 5, 8, 12, 14}
_FOCAL_MIN_RATIO = 0.05
_FOCAL_MAX_RATIO = 20.0
_DISTORTION_WARN_ABS = {
    CameraModelType.EXTENDED_PINHOLE: 5.0,
    CameraModelType.FISHEYE: 2.5,
}


def _issue(
    issues: list[CompatibilityIssue],
    code: str,
    severity: CompatibilitySeverity,
    side: str,
    message: str,
) -> None:
    issues.append(CompatibilityIssue(code=code, severity=severity, side=side, message=message))


def _check_matrix_and_values(cal: StandardCalibration, side: str, issues: list[CompatibilityIssue]) -> None:
    K = np.asarray(cal.camera_matrix)
    D = np.asarray(cal.distortion)
    if K.shape != (3, 3):
        _issue(issues, "camera_matrix_shape", CompatibilitySeverity.ERROR, side, f"camera_matrix shape가 3x3이 아닙니다: {K.shape}")
        return
    if D.size == 0:
        _issue(issues, "distortion_empty", CompatibilitySeverity.ERROR, side, "distortion coefficient가 비어 있습니다.")
    if not np.all(np.isfinite(K)):
        _issue(issues, "camera_matrix_non_finite", CompatibilitySeverity.ERROR, side, "camera_matrix에 NaN 또는 Inf가 있습니다.")
    if not np.all(np.isfinite(D)):
        _issue(issues, "distortion_non_finite", CompatibilitySeverity.ERROR, side, "distortion 계수에 NaN 또는 Inf가 있습니다.")
    if not np.allclose(K[2], [0.0, 0.0, 1.0], atol=1e-9):
        _issue(issues, "camera_matrix_bottom_row", CompatibilitySeverity.ERROR, side, "camera_matrix 마지막 행은 [0, 0, 1]이어야 합니다.")
    if abs(float(K[0, 1])) > 1e-9:
        _issue(issues, "camera_matrix_skew_nonzero", CompatibilitySeverity.WARNING, side, f"skew K[0,1]={float(K[0,1]):.6g}가 0이 아닙니다.")


def _check_image_size(
    cal: StandardCalibration,
    side: str,
    validation_image_size: tuple[int, int] | None,
    issues: list[CompatibilityIssue],
) -> None:
    if cal.width is None or cal.height is None:
        _issue(issues, "image_size_missing", CompatibilitySeverity.WARNING, side, "image width/height 정보가 없습니다.")
        return
    if cal.width <= 0 or cal.height <= 0:
        _issue(issues, "image_size_non_positive", CompatibilitySeverity.ERROR, side, "image width/height는 0보다 커야 합니다.")
        return
    if validation_image_size and (cal.width, cal.height) != validation_image_size:
        _issue(
            issues,
            "validation_resolution_mismatch",
            CompatibilitySeverity.ERROR,
            side,
            f"calibration 해상도 {cal.width}x{cal.height}가 validation 이미지 해상도 "
            f"{validation_image_size[0]}x{validation_image_size[1]}와 다릅니다.",
        )


def _check_model_and_distortion(cal: StandardCalibration, side: str, issues: list[CompatibilityIssue]) -> None:
    model = cal.model_name
    D = np.asarray(cal.distortion).reshape(-1)
    if model is None:
        _issue(issues, "camera_model_missing", CompatibilitySeverity.ERROR, side, "camera model이 명시되지 않았습니다.")
        return

    distortion_model = cal.distortion_model.lower() if cal.distortion_model else None
    if distortion_model:
        allowed = _DISTORTION_MODEL_FOR_MODEL.get(model, set())
        if distortion_model not in allowed:
            _issue(
                issues,
                "distortion_model_mismatch",
                CompatibilitySeverity.ERROR,
                side,
                f"camera model {model.value}와 distortion_model {cal.distortion_model!r} 조합이 맞지 않습니다.",
            )
    else:
        _issue(issues, "distortion_model_missing", CompatibilitySeverity.WARNING, side, "distortion_model 정보가 없습니다.")

    count = int(D.size)
    if model == CameraModelType.FISHEYE and count != 4:
        _issue(issues, "distortion_count_fisheye", CompatibilitySeverity.ERROR, side, f"Fisheye는 distortion 계수 4개가 필요합니다: 현재 {count}개")
    elif model == CameraModelType.EXTENDED_PINHOLE and count not in _EXTENDED_COUNTS:
        _issue(
            issues,
            "distortion_count_extended",
            CompatibilitySeverity.ERROR,
            side,
            f"Extended/Pinhole-radtan 계수 개수는 {sorted(_EXTENDED_COUNTS)} 중 하나여야 합니다: 현재 {count}개",
        )
    elif model == CameraModelType.PINHOLE and count not in (1, 4, 5):
        _issue(issues, "distortion_count_pinhole", CompatibilitySeverity.WARNING, side, f"Pinhole 계수 개수가 일반적이지 않습니다: {count}개")

    if model == CameraModelType.PINHOLE and D.size and not np.allclose(D, 0.0, atol=1e-12):
        _issue(issues, "pinhole_distortion_nonzero", CompatibilitySeverity.ERROR, side, "Pinhole 모델인데 distortion 계수가 0이 아닙니다.")

    limit = _DISTORTION_WARN_ABS.get(model)
    if limit is not None and D.size and np.all(np.isfinite(D)):
        max_abs = float(np.max(np.abs(D)))
        if max_abs > limit:
            _issue(
                issues,
                "distortion_parameter_range",
                CompatibilitySeverity.WARNING,
                side,
                f"distortion 계수 절대값 최대치 {max_abs:.3g}가 통상 범위(|d| <= {limit})를 넘습니다.",
            )


def _check_intrinsic_range(cal: StandardCalibration, side: str, issues: list[CompatibilityIssue]) -> None:
    K = np.asarray(cal.camera_matrix)
    if K.shape != (3, 3) or not np.all(np.isfinite(K)):
        return
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    if fx <= 0 or fy <= 0:
        _issue(issues, "focal_non_positive", CompatibilitySeverity.ERROR, side, f"fx/fy는 양수여야 합니다: fx={fx}, fy={fy}")
        return

    if cal.width and cal.height:
        ref = max(cal.width, cal.height)
        min_f = ref * _FOCAL_MIN_RATIO
        max_f = ref * _FOCAL_MAX_RATIO
        if fx < min_f or fx > max_f:
            _issue(issues, "fx_parameter_range", CompatibilitySeverity.WARNING, side, f"fx={fx:.1f}px가 해상도 대비 통상 범위({min_f:.0f}~{max_f:.0f}px)를 벗어났습니다.")
        if fy < min_f or fy > max_f:
            _issue(issues, "fy_parameter_range", CompatibilitySeverity.WARNING, side, f"fy={fy:.1f}px가 해상도 대비 통상 범위({min_f:.0f}~{max_f:.0f}px)를 벗어났습니다.")
        if not (-0.2 * cal.width <= cx <= 1.2 * cal.width):
            _issue(issues, "cx_parameter_range", CompatibilitySeverity.WARNING, side, f"cx={cx:.1f}px가 이미지 폭 기준 범위를 크게 벗어났습니다.")
        if not (-0.2 * cal.height <= cy <= 1.2 * cal.height):
            _issue(issues, "cy_parameter_range", CompatibilitySeverity.WARNING, side, f"cy={cy:.1f}px가 이미지 높이 기준 범위를 크게 벗어났습니다.")


def validate_single_calibration(
    calibration: StandardCalibration,
    *,
    side: str,
    validation_image_size: tuple[int, int] | None = None,
) -> list[CompatibilityIssue]:
    """calibration 하나의 구조/범위/모델 정보를 검사한다."""
    issues: list[CompatibilityIssue] = []
    _check_matrix_and_values(calibration, side, issues)
    _check_image_size(calibration, side, validation_image_size, issues)
    _check_model_and_distortion(calibration, side, issues)
    _check_intrinsic_range(calibration, side, issues)
    return issues


def validate_calibration_pair_compatibility(
    reference: StandardCalibration,
    candidate: StandardCalibration,
    *,
    validation_image_size: tuple[int, int] | None = None,
) -> CalibrationCompatibilityReport:
    """Reference/Candidate를 한 번에 검사해 benchmark 가능 여부를 반환한다."""
    issues: list[CompatibilityIssue] = []
    issues.extend(validate_single_calibration(reference, side="reference", validation_image_size=validation_image_size))
    issues.extend(validate_single_calibration(candidate, side="candidate", validation_image_size=validation_image_size))

    if reference.width and candidate.width and reference.height and candidate.height:
        if (reference.width, reference.height) != (candidate.width, candidate.height):
            _issue(
                issues,
                "reference_candidate_resolution_mismatch",
                CompatibilitySeverity.ERROR,
                "pair",
                f"Reference 해상도 {reference.width}x{reference.height}와 Candidate 해상도 "
                f"{candidate.width}x{candidate.height}가 다릅니다.",
            )

    if reference.model_name and candidate.model_name and reference.model_name != candidate.model_name:
        _issue(
            issues,
            "different_camera_models",
            CompatibilitySeverity.ERROR,
            "pair",
            f"Reference 모델({reference.model_name.value})과 Candidate 모델({candidate.model_name.value})이 다릅니다.",
        )

    if reference.distortion_model and candidate.distortion_model:
        if reference.distortion_model.lower() != candidate.distortion_model.lower():
            _issue(
                issues,
                "different_distortion_models",
                CompatibilitySeverity.ERROR,
                "pair",
                f"Reference distortion_model({reference.distortion_model})과 Candidate distortion_model({candidate.distortion_model})이 다릅니다.",
            )

    if reference.distortion.size != candidate.distortion.size:
        _issue(
            issues,
            "different_distortion_coefficient_count",
            CompatibilitySeverity.ERROR,
            "pair",
            f"Reference 계수 {reference.distortion.size}개와 Candidate 계수 {candidate.distortion.size}개가 다릅니다.",
        )

    return CalibrationCompatibilityReport(
        reference_label=reference.label,
        candidate_label=candidate.label,
        issues=issues,
    )
