"""
camera_calibrator.calibration.models.common
==============================================

pinhole / extended_pinhole / fisheye 세 모델이 공통으로 쓰는 로직.

원래 pinhole.py 안에 있던 헬퍼들을 여기로 옮겼다. 세 모델이 "같은 구조"를
갖도록 강제하는 목적도 있다 - 여기 정의된 함수만 쓰면 세 모델의 결과가
자동으로 같은 방식(영역 구분 기준, 최소 프레임 조건 등)으로 계산된다.
"""

from __future__ import annotations

import cv2
import numpy as np

from calibration.types import (
    CalibrationResult,
    CameraConfig,
    CameraModelType,
    Dataset,
    Frame,
    RegionalError,
)  # noqa: F401 (RegionalError used in type hints)

# calibrateCamera류 함수가 최소한으로 요구하는 뷰(이미지) 개수.
# 이론상 3장부터 동작하지만, 안정적인 초점거리 추정을 위해 최소치로 둔다.
# Fisheye는 파라미터가 더 많아(k1~k4) 이론적으로는 더 많은 뷰가 필요하지만,
# "최소 조건"은 세 모델 동일하게 두고 부족하면 개별 함수가 실패로 반환하게 한다.
MIN_FRAMES_REQUIRED = 3
MIN_CORNERS_PER_FRAME = 4  # cv2.calibrateCamera 계열 최소 요구사항

# 설계 문서 7번 "calibration termination criteria 통일" - 세 모델
# (calibrateCameraExtended / calibrateCameraExtended+RATIONAL / fisheye.calibrate)이
# 전부 같은 반복 종료 조건을 쓰도록 여기 하나로 고정한다. OpenCV 기본값
# (30회, DBL_EPSILON)보다 반복 횟수를 늘리고 종료 오차는 느슨하게 잡았다 -
# Fisheye처럼 파라미터가 많고 비선형성이 강한 모델은 30회 안에 못 끝나는
# 경우가 실측에서 있었기 때문에(대화 중 확인), 세 모델 모두 여유를 준다.
DEFAULT_TERM_CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6
)


def expected_free_param_count(
    model_name: CameraModelType,
    use_rational_model: bool = False,
    fix_tangent_dist: bool = False,
) -> int:
    """설계 문서 7번 "각 모델의 parameter 수 명확화" - fx/fy/cx/cy 4개를 뺀,
    distortion 계수 중 실제로 "자유도"(추정 대상)인 개수를 모델/옵션별로
    명시한다. self_check.py가 이미 실측 기반으로 검증해둔 값과 반드시 일치해야
    한다 - free_param_count(=np.count_nonzero(D))와 이 값이 다르면 계산이나
    플래그 설정이 잘못됐다는 신호다 (OpenCV 4.13이 rational model에서 배열을
    8이 아니라 14로 주는 것도 이 값이 아니라 배열 길이가 다른 것뿐이라는 점에
    유의 - self_check.py docstring 참고).
    """
    if model_name == CameraModelType.PINHOLE:
        return 0  # k1,k2,p1,p2,k3 전부 0으로 고정
    if model_name == CameraModelType.FISHEYE:
        return 4  # k1,k2,k3,k4 (Kannala-Brandt, 항상 4개)
    # Extended Pinhole
    count = 8 if use_rational_model else 5  # k1,k2,p1,p2,k3 [,k4,k5,k6]
    if fix_tangent_dist:
        count -= 2  # p1, p2 제외
    return count


def validate_finite_calibration_output(
    camera_matrix: np.ndarray, distortion: np.ndarray
) -> str | None:
    """설계 문서 7번 "numerical error 처리 / NaN/Inf 결과 검사" - cv2가
    예외 없이 "성공"으로 리턴해도 결과에 NaN/Inf가 섞여 있을 수 있으므로,
    각 모델의 calibrate_*() 함수가 CalibrationResult를 만들기 직전에 항상
    이 함수를 거치게 한다. 여기서 걸러지면 success=False로 조기 반환하고,
    (sanity_check.py의 사후 검사는 "성공했지만 이상한 결과"까지 잡아내는
    두 번째 방어선이다 - 이 함수는 더 근본적인 "숫자 자체가 깨진" 경우를
    아예 파이프라인 뒤로 못 넘어가게 막는 첫 번째 방어선).
    """
    if camera_matrix is None or not np.all(np.isfinite(camera_matrix)):
        return "camera_matrix에 NaN 또는 Inf가 포함되어 있습니다 (계산 발산)."
    if distortion is None or not np.all(np.isfinite(distortion)):
        return "distortion 계수에 NaN 또는 Inf가 포함되어 있습니다 (계산 발산)."
    if camera_matrix[0, 0] <= 0 or camera_matrix[1, 1] <= 0:
        return f"fx/fy가 양수가 아닙니다 (fx={camera_matrix[0,0]}, fy={camera_matrix[1,1]})."
    return None


def collect_calibration_inputs(
    dataset: Dataset,
) -> tuple[list[Frame], list[np.ndarray], list[np.ndarray]]:
    """검출 성공 + 활성화(enabled) + 최소 코너 수를 만족하는 프레임만 골라
    calibrateCamera 입력 형태(object_points 리스트, image_points 리스트)로 변환.

    설계 문서 16번 - corner-level outlier가 표시해둔 excluded_corner_indices가
    있으면, 프레임 자체는 살리되 그 인덱스에 해당하는 코너만 입력에서 뺀다.
    제외하고 남은 코너가 MIN_CORNERS_PER_FRAME 밑으로 떨어지면 그 프레임은
    통째로 빠진다 (calibrateCamera가 애초에 요구하는 최소 조건).
    """
    usable_frames: list[Frame] = []
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []

    for f in dataset.enabled_frames:
        det = f.detection
        if not det or not det.success:
            continue

        obj, img = det.object_points, det.corners
        excluded = det.excluded_corner_indices
        if excluded:
            mask = np.ones(obj.shape[0], dtype=bool)
            valid_idx = [i for i in excluded if 0 <= i < obj.shape[0]]
            mask[valid_idx] = False
            obj, img = obj[mask], img[mask]

        if obj.shape[0] < MIN_CORNERS_PER_FRAME:
            continue

        usable_frames.append(f)
        object_points.append(obj)
        image_points.append(img)

    return usable_frames, object_points, image_points


def infer_image_size(dataset: Dataset, camera_config: CameraConfig) -> tuple[int, int]:
    """CameraConfig에 해상도가 없으면 첫 번째 프레임에서 유추."""
    if camera_config.width and camera_config.height:
        return camera_config.width, camera_config.height

    for f in dataset.frames:
        if f.image_info.width and f.image_info.height:
            return f.image_info.width, f.image_info.height

    raise ValueError(
        "이미지 해상도를 알 수 없습니다. CameraConfig 또는 프레임에 width/height가 필요합니다."
    )


def classify_regions(cx: float, cy: float, w: int, h: int) -> list[str]:
    """보드 중심 좌표를 기준으로 이 프레임이 속하는 영역들을 반환.
    한 프레임이 여러 영역(예: left + top + corner)에 동시에 속할 수 있다.
    """
    x_third, y_third = w / 3, h / 3
    regions: list[str] = []

    horiz = "left" if cx < x_third else ("right" if cx > 2 * x_third else "center_x")
    vert = "top" if cy < y_third else ("bottom" if cy > 2 * y_third else "center_y")

    if horiz == "center_x" and vert == "center_y":
        regions.append("center")
    if horiz == "left":
        regions.append("left")
    if horiz == "right":
        regions.append("right")
    if vert == "top":
        regions.append("top")
    if vert == "bottom":
        regions.append("bottom")
    if horiz in ("left", "right") and vert in ("top", "bottom"):
        regions.append("corner")

    return regions


def compute_regional_error(
    frames: list[Frame],
    per_frame_error: dict[str, float],
    image_size: tuple[int, int],
) -> RegionalError:
    """설계 문서 4번 - Center/Left/Right/Top/Bottom/Corner RMS.

    세 모델(Pinhole/Extended/Fisheye) 모두 이 함수를 그대로 재사용하므로,
    영역 구분 기준이 모델마다 달라질 걱정 없이 공정하게 비교할 수 있다.
    """
    w, h = image_size
    buckets: dict[str, list[float]] = {
        "center": [], "left": [], "right": [], "top": [], "bottom": [], "corner": [],
    }

    for frame in frames:
        error = per_frame_error.get(frame.image_info.image_id)
        center = frame.detection.board_center_px if frame.detection else None
        if error is None or center is None:
            continue
        for region in classify_regions(center[0], center[1], w, h):
            buckets[region].append(error)

    def _avg(values: list[float]) -> float | None:
        return float(np.mean(values)) if values else None

    return RegionalError(
        center=_avg(buckets["center"]),
        left=_avg(buckets["left"]),
        right=_avg(buckets["right"]),
        top=_avg(buckets["top"]),
        bottom=_avg(buckets["bottom"]),
        corner=_avg(buckets["corner"]),
    )


def fmt_optional(v: float | None) -> str:
    return f"{v:.3f}" if v is not None else "N/A"


def distortion_coeff_labels(model_name: CameraModelType, count: int) -> list[str]:
    """왜곡 계수 벡터(distortion.ravel())의 각 원소 이름을 순서대로 반환.

    OpenCV의 계수 순서는 모델/플래그에 따라 원소 "개수"가 달라도 앞부분
    순서는 고정이다 (뒤에 이어붙는 방식):
      - Pinhole/Extended Pinhole (calibrateCamera 계열):
        k1, k2, p1, p2, [k3, [k4, k5, k6, [s1, s2, s3, s4, [taux, tauy]]]]
        (5개면 k3까지, CALIB_RATIONAL_MODEL을 쓰면 8개로 k4~k6까지 늘어난다)
      - Fisheye (cv2.fisheye, Kannala-Brandt): k1, k2, k3, k4 (항상 4개)

    개수가 알려진 패턴과 다르면(예: 향후 s1~s4/tau 확장) 안전하게
    "d{i}" 형태로 채운다 - 라벨이 없다고 값 자체를 숨기지 않는다.
    """
    if model_name == CameraModelType.FISHEYE:
        base = ["k1", "k2", "k3", "k4"]
    else:
        base = ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4", "taux", "tauy"]

    if count <= len(base):
        return base[:count]
    return base + [f"d{i}" for i in range(len(base), count)]


def regional_edge_average(regional_error: RegionalError) -> float | None:
    """RegionalError에서 외곽(left/right/top/bottom/corner)만 평균낸 값.
    compare.py와 validation.py가 '외곽 오차'를 정의할 때 같은 기준을 쓰도록 공용화.
    """
    edge_vals = [
        v
        for v in (
            regional_error.left,
            regional_error.right,
            regional_error.top,
            regional_error.bottom,
            regional_error.corner,
        )
        if v is not None
    ]
    return float(np.mean(edge_vals)) if edge_vals else None


def undistort_image(
    image: np.ndarray,
    result: CalibrationResult,
    camera_config: CameraConfig,
    balance: float = 0.0,
) -> np.ndarray:
    """캘리브레이션 결과로 이미지를 보정(undistort). UI의 preview.py가 이 함수를
    호출한다 - 실제 OpenCV 왜곡 보정 로직은 여기(backend)에만 있고, UI는
    화면에 그리는 것만 담당한다 (백엔드/UI 분리 원칙, 설계 문서 16번).

    Fisheye는 cv2.undistort가 아니라 별도 remap 경로가 필요해 분기한다.
    balance: fisheye 전용, 0=최대 크롭(왜곡 없는 중심만) ~ 1=원본 화각 최대 보존.
    """
    if not result.success or result.camera_matrix is None or result.distortion is None:
        raise ValueError(f"실패한 CalibrationResult는 undistort할 수 없습니다: {result.error_message}")

    K, D = result.camera_matrix, result.distortion
    size = (camera_config.width, camera_config.height)

    if result.model_name == CameraModelType.FISHEYE:
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, size, np.eye(3), balance=balance
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), new_K, size, cv2.CV_16SC2
        )
        return cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)

    return cv2.undistort(image, K, D)


def compute_mad_threshold(errors: list[float], k: float = 3.0, mad_scale: float = 1.0) -> float:
    """threshold = median(error) + k * MAD

    원래 outlier.py에 있던 함수를 여기(models/common.py)로 옮겼다 - outlier.py는
    pinhole/extended_pinhole/fisheye 세 모델 함수를 import하는 "상위" 모듈이라,
    residual_stats.py(모델 함수들이 CalibrationResult를 만들 때 바로 호출)가
    outlier.py를 다시 import하면 순환 참조가 생긴다. common.py는 모델 함수들의
    "하위" 의존성이라 이 방향의 순환이 생기지 않는다 - outlier.py는 하위
    호환을 위해 이 함수를 그대로 재노출(import)한다.

    mad_scale: 정규분포 가정 하에서 MAD를 표준편차와 비슷한 스케일로 맞추려면
    1.4826을 곱하는 게 통계적으로 흔한 관례다. 문서 원문 공식은 스케일링 없이
    그대로("median + 3*MAD")이므로 기본값은 1.0으로 두되, 필요하면
    UI 고급 옵션에서 1.4826으로 바꿀 수 있게 열어둔다.
    """
    if not errors:
        return 0.0
    arr = np.asarray(errors, dtype=float)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    return median + k * mad_scale * mad
