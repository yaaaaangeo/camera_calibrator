"""
camera_calibrator.calibration.sanity_check
==============================================

설계 문서 8번 - "Calibration 결과 sanity check 추가".

캘리브레이션이 "성공"(cv2가 예외 없이 리턴)했다고 해서 결과가 물리적으로
말이 되는 건 아니다. 이 모듈은 계산된 CalibrationResult가 다음을 만족하는지
검사한다 (문서 8번 체크리스트 그대로):

    - fx > 0
    - fy > 0
    - principal point가 이미지 밖으로 나가지 않는지
    - aspect ratio 이상 여부
    - distortion coefficient 이상 여부
    - FOV 이상 여부 (문서 38번 FOV Sanity Check와 동일한 개념, 여기서 통합)
    - focal length 이상 여부
    - calibration RMS 이상 여부
    - parameter가 비정상적으로 큰지/작은지
    - NaN/Inf 결과 검사

이 모듈은 "계산이 틀렸다"를 증명하지 않는다 - 사람이 결과를 받아들이기 전에
훑어볼 경고 목록을 만드는 게 목적이다 (문서 43번 "Score는 요약이고 원본
metric이 근거" 원칙과 같은 철학: 여기서도 판정이 아니라 근거를 보여준다).
그래서 대부분의 항목은 ERROR가 아니라 WARNING이다 - 광각 렌즈, 크롭 센서
등 정상적인 이유로 통상 범위를 벗어나는 경우가 실제로 있기 때문에, 최종
판단은 여전히 사용자의 몫으로 남긴다. NaN/Inf, fx<=0, Pinhole인데 distortion
이 0이 아닌 경우처럼 "물리적으로 불가능"한 경우에만 ERROR로 분류한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from calibration.types import CalibrationResult, CameraConfig, CameraModelType


class SanitySeverity(str, Enum):
    ERROR = "error"      # 물리적으로 불가능 - 결과를 신뢰할 수 없음
    WARNING = "warning"  # 가능은 하지만 통상 범위를 벗어남 - 확인 권장


@dataclass
class SanityIssue:
    code: str
    severity: SanitySeverity
    message: str


@dataclass
class SanityCheckResult:
    model_name: CameraModelType
    issues: list[SanityIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == SanitySeverity.ERROR for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == SanitySeverity.WARNING for i in self.issues)

    @property
    def passed(self) -> bool:
        """ERROR가 하나도 없으면 통과. WARNING만 있어도 통과지만 확인을 권장한다."""
        return not self.has_errors

    def format(self) -> str:
        label = self.model_name.value if hasattr(self.model_name, "value") else str(self.model_name)
        if not self.issues:
            return f"[{label}] sanity check 이상 없음"
        lines = [f"[{label}] sanity check: 경고 {len(self.issues)}건"]
        for issue in self.issues:
            mark = "\u2716" if issue.severity == SanitySeverity.ERROR else "\u26a0"
            lines.append(f"  {mark} {issue.message}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 판정 기준값 (실측 통상 범위를 기준으로 여유를 두고 설정, 절대 기준 아님)
# ---------------------------------------------------------------------------

# 모델별 distortion 계수 절대값 상한. 초과한다고 무조건 틀린 건 아니지만
# (특히 초광각), 대부분의 정상적인 캘리브레이션 결과에서 관찰되는 범위를
# 크게 벗어나면 발산했거나 데이터가 잘못됐을 가능성이 높다.
_DISTORTION_ABS_LIMIT = {
    CameraModelType.EXTENDED_PINHOLE: 3.0,  # k1~k6(rational)+p1,p2 포함해도 보통 이 범위 안
    CameraModelType.FISHEYE: 1.5,           # k1~k4 (equidistant)
}

# focal length(px)가 이미지 크기 대비 몇 배~몇 분의 1배를 벗어나면 비정상으로
# 볼지. 초광각이면 작은 쪽, 망원이면 큰 쪽 - 이 범위를 완전히 벗어나면
# 캘리브레이션 자체가 잘못됐을 가능성이 매우 높다.
_FOCAL_LENGTH_MIN_RATIO = 0.15
_FOCAL_LENGTH_MAX_RATIO = 8.0

_ASPECT_RATIO_WARN_PCT = 5.0   # fx/fy가 서로 이 % 이상 차이나면 경고
_RMS_WARNING_PX = 1.0
_RMS_ERROR_PX = 3.0
_DISTORTION_ABS_MIN = 1e-4   # 이보다 작으면 "사실상 왜곡 없음"으로 본다 (parameter가 비정상적으로 작은 경우)

# FOV 스펙(camera_config.hfov_deg/vfov_deg, 문서 38번 "제조사 spec 입력")과
# 추정치 차이가 이 이상이면 경고. 상대오차(spec의 15%)와 절대오차(10도) 중
# 큰 쪽을 기준으로 삼는다 - 광각일수록 절대오차 허용폭을 넓히기 위함.
_FOV_SPEC_DIFF_ABS_DEG = 10.0
_FOV_SPEC_DIFF_REL = 0.15


def _finite(x) -> bool:
    try:
        return bool(np.isfinite(x))
    except (TypeError, ValueError):
        return False


def _check_finite_arrays(result: CalibrationResult, issues: list[SanityIssue]) -> bool:
    """camera_matrix/distortion/RMS에 NaN/Inf가 있으면 이후 검사가 전부
    무의미해지므로 가장 먼저 확인하고, 문제가 있으면 True 대신 False를
    반환해 호출자가 나머지 검사를 건너뛰게 한다.
    """
    ok = True
    if result.camera_matrix is not None and not np.all(np.isfinite(result.camera_matrix)):
        issues.append(SanityIssue(
            "camera_matrix_non_finite", SanitySeverity.ERROR,
            "camera_matrix에 NaN 또는 Inf 값이 있습니다 - 캘리브레이션이 발산했을 가능성이 높습니다.",
        ))
        ok = False
    if result.distortion is not None and not np.all(np.isfinite(result.distortion)):
        issues.append(SanityIssue(
            "distortion_non_finite", SanitySeverity.ERROR,
            "distortion 계수에 NaN 또는 Inf 값이 있습니다 - 캘리브레이션이 발산했을 가능성이 높습니다.",
        ))
        ok = False
    if result.rms_error is not None and not _finite(result.rms_error):
        issues.append(SanityIssue(
            "rms_non_finite", SanitySeverity.ERROR,
            "RMS 재투영 오차가 NaN 또는 Inf입니다.",
        ))
        ok = False
    return ok


def _check_focal_length(
    K: np.ndarray, image_size: tuple[int, int], issues: list[SanityIssue]
) -> tuple[float, float]:
    fx, fy = float(K[0, 0]), float(K[1, 1])
    w, h = image_size

    if fx <= 0:
        issues.append(SanityIssue(
            "fx_non_positive", SanitySeverity.ERROR,
            f"fx={fx:.2f} - focal length는 반드시 양수여야 합니다.",
        ))
    if fy <= 0:
        issues.append(SanityIssue(
            "fy_non_positive", SanitySeverity.ERROR,
            f"fy={fy:.2f} - focal length는 반드시 양수여야 합니다.",
        ))
    if fx <= 0 or fy <= 0:
        return fx, fy

    ref = max(w, h)
    min_focal = ref * _FOCAL_LENGTH_MIN_RATIO
    max_focal = ref * _FOCAL_LENGTH_MAX_RATIO
    if fx < min_focal or fx > max_focal:
        issues.append(SanityIssue(
            "fx_out_of_range", SanitySeverity.WARNING,
            f"fx={fx:.1f}px가 이미지 크기({w}x{h}) 대비 통상 범위"
            f"({min_focal:.0f}~{max_focal:.0f}px)를 벗어났습니다.",
        ))
    if fy < min_focal or fy > max_focal:
        issues.append(SanityIssue(
            "fy_out_of_range", SanitySeverity.WARNING,
            f"fy={fy:.1f}px가 이미지 크기({w}x{h}) 대비 통상 범위"
            f"({min_focal:.0f}~{max_focal:.0f}px)를 벗어났습니다.",
        ))

    aspect_pct = abs(fx - fy) / max(fx, fy) * 100
    if aspect_pct > _ASPECT_RATIO_WARN_PCT:
        issues.append(SanityIssue(
            "aspect_ratio_off", SanitySeverity.WARNING,
            f"fx/fy 비율이 {aspect_pct:.1f}% 차이납니다 (fx={fx:.1f}, fy={fy:.1f}) "
            "- 정사각형 픽셀 센서라면 통상 1~2% 이내여야 합니다.",
        ))

    return fx, fy


def _check_principal_point(
    K: np.ndarray, image_size: tuple[int, int], issues: list[SanityIssue]
) -> None:
    cx, cy = float(K[0, 2]), float(K[1, 2])
    w, h = image_size
    # 완전히 0~w / 0~h 안이 아니어도(크롭·오프셋 렌즈면 벗어날 수 있음) 바로
    # 에러로 보지 않지만, 이미지 폭/높이의 20%를 넘겨 벗어나면 비정상으로 본다.
    margin_x, margin_y = w * 0.2, h * 0.2
    if cx < -margin_x or cx > w + margin_x:
        issues.append(SanityIssue(
            "cx_out_of_bounds", SanitySeverity.WARNING,
            f"주점 cx={cx:.1f}가 이미지 폭(0~{w}px)을 크게 벗어났습니다.",
        ))
    if cy < -margin_y or cy > h + margin_y:
        issues.append(SanityIssue(
            "cy_out_of_bounds", SanitySeverity.WARNING,
            f"주점 cy={cy:.1f}가 이미지 높이(0~{h}px)를 크게 벗어났습니다.",
        ))


def _check_distortion(
    model: CameraModelType, D: np.ndarray | None, issues: list[SanityIssue]
) -> None:
    if model == CameraModelType.PINHOLE:
        if D is not None and D.size > 0 and not np.all(D == 0):
            issues.append(SanityIssue(
                "pinhole_distortion_nonzero", SanitySeverity.ERROR,
                "Pinhole 모델인데 distortion 계수가 0이 아닙니다 (구현 오류 가능성).",
            ))
        return

    if D is None or D.size == 0:
        return
    limit = _DISTORTION_ABS_LIMIT.get(model)
    if limit is None:
        return
    max_abs = float(np.max(np.abs(D)))
    if max_abs > limit:
        idx = int(np.argmax(np.abs(D)))
        issues.append(SanityIssue(
            "distortion_magnitude_large", SanitySeverity.WARNING,
            f"distortion 계수 절대값 최대치가 {max_abs:.3f}(계수 인덱스 {idx})로 "
            f"통상 범위(|d|<={limit})를 넘었습니다 - 발산에 가까운 결과일 수 있습니다.",
        ))
    elif max_abs < _DISTORTION_ABS_MIN:
        # k1(주 방사왜곡 계수)이 사실상 0에 가까우면 "왜곡이 정말 없는 렌즈"일
        # 수도 있지만, Extended/Fisheye 모델을 굳이 쓸 이유가 없다는 신호이기도
        # 하다 - Pinhole로도 충분할 가능성이 높으므로 참고용으로 알려준다.
        issues.append(SanityIssue(
            "distortion_magnitude_tiny", SanitySeverity.WARNING,
            f"distortion 계수 절대값 최대치가 {max_abs:.5f}로 매우 작습니다 "
            "- 이 렌즈는 왜곡이 거의 없어 Pinhole 모델로도 충분할 수 있습니다.",
        ))


def _estimate_fov_deg(
    model: CameraModelType, fx: float, fy: float, image_size: tuple[int, int]
) -> tuple[float, float]:
    """모델에 맞는 근사식으로 수평/수직 FOV(도)를 추정한다.

    Pinhole/Extended Pinhole은 표준 perspective 모델이라 2*atan(size/2f)를
    쓰지만, Fisheye(Kannala-Brandt 계열)는 r = f*theta인 equidistant 근사가
    맞다 - atan 공식을 그대로 쓰면 광각일수록 FOV가 크게 과소평가된다.
    """
    w, h = image_size
    if model == CameraModelType.FISHEYE:
        hfov = math.degrees(2 * (w / 2) / fx)
        vfov = math.degrees(2 * (h / 2) / fy)
    else:
        hfov = math.degrees(2 * math.atan((w / 2) / fx))
        vfov = math.degrees(2 * math.atan((h / 2) / fy))
    return hfov, vfov


def _check_fov(
    model: CameraModelType,
    fx: float,
    fy: float,
    image_size: tuple[int, int],
    camera_config: CameraConfig,
    issues: list[SanityIssue],
) -> None:
    """문서 38번 FOV Sanity Check. camera_config에 제조사 스펙(hfov_deg/
    vfov_deg)이 입력돼 있으면 추정치와 비교하고, 없으면 "비상식적인 값
    (0도 이하 또는 220도 이상)"인지만 확인한다.
    """
    hfov, vfov = _estimate_fov_deg(model, fx, fy, image_size)

    if not (1.0 < hfov < 220.0):
        issues.append(SanityIssue(
            "hfov_implausible", SanitySeverity.WARNING,
            f"추정 수평 FOV={hfov:.1f}\u00b0 - 비상식적인 값입니다.",
        ))
    if not (1.0 < vfov < 220.0):
        issues.append(SanityIssue(
            "vfov_implausible", SanitySeverity.WARNING,
            f"추정 수직 FOV={vfov:.1f}\u00b0 - 비상식적인 값입니다.",
        ))

    if camera_config.hfov_deg:
        diff = abs(hfov - camera_config.hfov_deg)
        if diff > max(_FOV_SPEC_DIFF_ABS_DEG, camera_config.hfov_deg * _FOV_SPEC_DIFF_REL):
            issues.append(SanityIssue(
                "hfov_spec_mismatch", SanitySeverity.WARNING,
                f"추정 수평 FOV={hfov:.1f}\u00b0가 입력한 스펙"
                f"({camera_config.hfov_deg:.1f}\u00b0)과 {diff:.1f}\u00b0 차이납니다.",
            ))
    if camera_config.vfov_deg:
        diff = abs(vfov - camera_config.vfov_deg)
        if diff > max(_FOV_SPEC_DIFF_ABS_DEG, camera_config.vfov_deg * _FOV_SPEC_DIFF_REL):
            issues.append(SanityIssue(
                "vfov_spec_mismatch", SanitySeverity.WARNING,
                f"추정 수직 FOV={vfov:.1f}\u00b0가 입력한 스펙"
                f"({camera_config.vfov_deg:.1f}\u00b0)과 {diff:.1f}\u00b0 차이납니다.",
            ))


def _check_rms(result: CalibrationResult, issues: list[SanityIssue]) -> None:
    if result.rms_error is None:
        return
    if result.rms_error > _RMS_ERROR_PX:
        issues.append(SanityIssue(
            "rms_very_high", SanitySeverity.WARNING,
            f"Train RMS={result.rms_error:.3f}px - 매우 높습니다 "
            "(검출 오류/이상치/모델 불일치 가능성이 있습니다).",
        ))
    elif result.rms_error > _RMS_WARNING_PX:
        issues.append(SanityIssue(
            "rms_high", SanitySeverity.WARNING,
            f"Train RMS={result.rms_error:.3f}px - 다소 높은 편입니다.",
        ))


def run_sanity_check(
    result: CalibrationResult,
    camera_config: CameraConfig,
    image_size: tuple[int, int] | None = None,
) -> SanityCheckResult:
    """캘리브레이션 결과 하나(CalibrationResult)를 검사해 경고 목록을 만든다.

    calibrate_pinhole/calibrate_extended_pinhole/calibrate_fisheye가 리턴한
    결과라면 어느 것이든 그대로 넣을 수 있다. 이 함수는 result를 수정하지
    않는다 - 순수하게 "보여주기용" 진단 결과만 만든다.
    """
    issues: list[SanityIssue] = []

    if not result.success or result.camera_matrix is None:
        issues.append(SanityIssue(
            "calibration_failed", SanitySeverity.ERROR,
            result.error_message or "캘리브레이션이 실패했습니다.",
        ))
        return SanityCheckResult(model_name=result.model_name, issues=issues)

    if not _check_finite_arrays(result, issues):
        # camera_matrix/distortion/RMS 중 하나라도 non-finite면 그 값을 쓰는
        # 나머지 검사(FOV, aspect ratio 등)는 의미가 없으므로 여기서 끝낸다.
        return SanityCheckResult(model_name=result.model_name, issues=issues)

    w, h = image_size if image_size else (camera_config.width, camera_config.height)

    fx, fy = _check_focal_length(result.camera_matrix, (w, h), issues)
    _check_principal_point(result.camera_matrix, (w, h), issues)
    _check_distortion(result.model_name, result.distortion, issues)
    if fx > 0 and fy > 0:
        _check_fov(result.model_name, fx, fy, (w, h), camera_config, issues)
    _check_rms(result, issues)

    return SanityCheckResult(model_name=result.model_name, issues=issues)


def run_sanity_checks(
    results: list[CalibrationResult],
    camera_config: CameraConfig,
    image_size: tuple[int, int] | None = None,
) -> list[SanityCheckResult]:
    """run_all_models()가 리턴한 여러 모델 결과를 한 번에 검사."""
    return [run_sanity_check(r, camera_config, image_size) for r in results]


def format_sanity_checks(checks: list[SanityCheckResult]) -> str:
    return "\n".join(c.format() for c in checks)
