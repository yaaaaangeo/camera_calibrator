"""
camera_calibrator.calibration.models.common
==============================================

pinhole / extended_pinhole / fisheye가 직접 쓰고, brown_conrady는
extended_pinhole을 내부적으로 재사용하므로 간접적으로 공유하는 - 즉 Standard
4모델 전체가 공통으로 쓰는 로직.

원래 pinhole.py 안에 있던 헬퍼들을 여기로 옮겼다. 네 모델이 "같은 구조"를
갖도록 강제하는 목적도 있다 - 여기 정의된 함수만 쓰면 네 모델의 결과가
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
# "최소 조건"은 모델 전체에 동일하게 두고 부족하면 개별 함수가 실패로 반환하게 한다.
MIN_FRAMES_REQUIRED = 3
MIN_CORNERS_PER_FRAME = 4  # cv2.calibrateCamera 계열 최소 요구사항

# 설계 문서 7번 "calibration termination criteria 통일" - Standard 4모델의
# 계산 경로(calibrateCameraExtended / calibrateCameraExtended+RATIONAL /
# fisheye.calibrate)가 전부 같은 반복 종료 조건을 쓰도록 여기 하나로 고정한다.
# OpenCV 기본값(30회, DBL_EPSILON)보다 반복 횟수를 늘리고 종료 오차는 느슨하게
# 잡았다 - Fisheye처럼 파라미터가 많고 비선형성이 강한 모델은 30회 안에 못
# 끝나는 경우가 실측에서 있었기 때문에(대화 중 확인), 모든 모델에 여유를 준다.
DEFAULT_TERM_CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6
)


def expected_free_param_count(
    model_name: CameraModelType,
    fix_tangent_dist: bool = False,
) -> int:
    """설계 문서 7번 "각 모델의 parameter 수 명확화" - fx/fy/cx/cy 4개를 뺀,
    distortion 계수 중 실제로 "자유도"(추정 대상)인 개수를 모델/옵션별로
    명시한다. self_check.py가 이미 실측 기반으로 검증해둔 값과 반드시 일치해야
    한다 - free_param_count(=np.count_nonzero(D))와 이 값이 다르면 계산이나
    플래그 설정이 잘못됐다는 신호다 (OpenCV 4.13이 rational model에서 배열을
    8이 아니라 14로 주는 것도 이 값이 아니라 배열 길이가 다른 것뿐이라는 점에
    유의 - self_check.py docstring 참고).

    모델 의미가 고정이므로(EXTENDED_PINHOLE=항상 Rational 8계수,
    BROWN_CONRADY=항상 5계수) runtime toggle 파라미터는 없다.
    """
    if model_name == CameraModelType.PINHOLE:
        return 0  # k1,k2,p1,p2,k3 전부 0으로 고정
    if model_name == CameraModelType.FISHEYE:
        return 4  # k1,k2,k3,k4 (Kannala-Brandt, 항상 4개)
    if model_name == CameraModelType.BROWN_CONRADY:
        return 3 if fix_tangent_dist else 5
    # Extended Pinhole (Rational) - 항상 8계수
    count = 8  # k1,k2,p1,p2,k3,k4,k5,k6
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
    *,
    require_full_target: bool = False,
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
        if obj is None or img is None:
            continue
        expected_count = None
        if det.object_points is not None:
            expected_count = det.object_points.shape[0]
        if require_full_target and expected_count is not None and img.shape[0] != expected_count:
            continue

        excluded = [] if require_full_target else det.excluded_corner_indices
        if excluded:
            mask = np.ones(obj.shape[0], dtype=bool)
            valid_idx = [i for i in excluded if 0 <= i < obj.shape[0]]
            mask[valid_idx] = False
            obj, img = obj[mask], img[mask]

        if not _is_valid_calibration_view(obj, img):
            continue

        usable_frames.append(f)
        object_points.append(obj)
        image_points.append(img)

    return usable_frames, object_points, image_points


def _is_valid_calibration_view(obj: np.ndarray, img: np.ndarray) -> bool:
    """OpenCV의 planar homography 초기화가 가능한 한 뷰인지 검사한다.

    ``calibrateCameraExtended``는 각 뷰에서 먼저 3x3 homography를 구한다.
    점이 4개여도 중복되거나 한 직선 위에 있으면 homography가 비어 있고,
    OpenCV 5.x에서는 ``matH0.size() == Size(3,3)`` assertion으로 종료된다.
    라이브 검출 및 corner-level 제외 후에 생길 수 있는 이 입력을 호출 전에
    걸러서, 나머지 정상 프레임만으로 캘리브레이션하도록 한다.
    """
    if obj is None or img is None:
        return False

    try:
        obj_pts = np.asarray(obj).reshape(-1, 3)
        img_pts = np.asarray(img).reshape(-1, 2)
    except (TypeError, ValueError):
        return False
    if (
        len(obj_pts) != len(img_pts)
        or len(obj_pts) < MIN_CORNERS_PER_FRAME
        or not np.all(np.isfinite(obj_pts))
        or not np.all(np.isfinite(img_pts))
    ):
        return False

    # 같은 좌표를 여러 번 센 것은 homography의 독립 대응점이 아니다.
    if len(np.unique(obj_pts, axis=0)) < MIN_CORNERS_PER_FRAME:
        return False
    if len(np.unique(img_pts, axis=0)) < MIN_CORNERS_PER_FRAME:
        return False

    # ChArUco/체스보드/AprilGrid의 object point는 평면 위에 있으므로 PCA로
    # 평면의 두 축을 얻는다. object/image 모두 2차원 rank여야 homography를
    # 유일하게 초기화할 수 있다.
    obj_centered = obj_pts - np.mean(obj_pts, axis=0)
    img_centered = img_pts - np.mean(img_pts, axis=0)
    return bool(
        np.linalg.matrix_rank(obj_centered) >= 2
        and np.linalg.matrix_rank(img_centered) >= 2
    )


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
    xs: np.ndarray | list[float],
    ys: np.ndarray | list[float],
    errors: np.ndarray | list[float],
    image_size: tuple[int, int],
) -> RegionalError:
    """Corner-level Center/Left/Right/Top/Bottom/Corner RMS.

    각 검출 코너의 관측 픽셀 좌표로 영역을 분류하고, 영역별 값은
    per-frame RMS 평균이 아니라 해당 코너들의 pooled RMS로 계산한다.
    """
    w, h = image_size
    buckets: dict[str, list[float]] = {
        "center": [], "left": [], "right": [], "top": [], "bottom": [], "corner": [],
    }

    point_xs = np.asarray(xs, dtype=np.float64).reshape(-1)
    point_ys = np.asarray(ys, dtype=np.float64).reshape(-1)
    point_errors = np.asarray(errors, dtype=np.float64).reshape(-1)
    if not (point_xs.size == point_ys.size == point_errors.size):
        point_count = min(point_xs.size, point_ys.size, point_errors.size)
        point_xs = point_xs[:point_count]
        point_ys = point_ys[:point_count]
        point_errors = point_errors[:point_count]

    for x, y, error in zip(point_xs, point_ys, point_errors):
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(error)):
            continue
        for region in classify_regions(float(x), float(y), w, h):
            buckets[region].append(float(error))

    def _rms(values: list[float]) -> float | None:
        return float(np.sqrt(np.mean(np.square(values)))) if values else None

    return RegionalError(
        center=_rms(buckets["center"]),
        left=_rms(buckets["left"]),
        right=_rms(buckets["right"]),
        top=_rms(buckets["top"]),
        bottom=_rms(buckets["bottom"]),
        corner=_rms(buckets["corner"]),
    )


def fmt_optional(v: float | None) -> str:
    return f"{v:.3f}" if v is not None else "N/A"


def expected_distortion_coeff_count(model_name: CameraModelType) -> int:
    if model_name == CameraModelType.FISHEYE:
        return 4
    if model_name == CameraModelType.EXTENDED_PINHOLE:
        return 8
    return 5


def normalize_distortion_coefficients(model_name: CameraModelType, distortion: np.ndarray) -> np.ndarray:
    count = expected_distortion_coeff_count(model_name)
    flat = np.asarray(distortion, dtype=np.float64).reshape(-1)
    if flat.size >= count:
        return flat[:count].reshape(-1, 1).copy()
    padded = np.zeros(count, dtype=np.float64)
    padded[: flat.size] = flat
    return padded.reshape(-1, 1)


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


def solve_pnp_for_model(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    model: CameraModelType,
) -> tuple[bool, np.ndarray, np.ndarray]:
    """cv2.solvePnP vs cv2.fisheye.solvePnP 분기를 한 곳에 모은다.

    validation.py::_test_reprojection_errors가 원래 이 분기를 직접 갖고
    있었고, calibration/windshield 패키지(고정 K,D로 포즈만 다시 구하는
    용도)도 정확히 같은 분기가 필요해서 여기로 뽑았다 - fisheye는 float64를
    요구한다(calibration.models.fisheye와 동일한 주의사항).
    """
    if model == CameraModelType.FISHEYE:
        obj64 = np.asarray(object_points, dtype=np.float64)
        img64 = np.asarray(image_points, dtype=np.float64)
        ok, rvec, tvec = cv2.fisheye.solvePnP(obj64, img64, camera_matrix, distortion)
        return bool(ok), rvec, tvec
    ok, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, distortion)
    return bool(ok), rvec, tvec


def _finite_pose(rvec, tvec) -> bool:
    return (
        rvec is not None and tvec is not None
        and np.all(np.isfinite(rvec)) and np.all(np.isfinite(tvec))
    )


def _solve_pnp_fisheye_robust(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> tuple[bool, np.ndarray, np.ndarray, str]:
    """Fisheye test-pose 추정 전용 fallback chain (Hold-out 원칙 준수: 이 함수
    안 어디에서도 camera_matrix/distortion을 다시 추정하거나 수정하지 않는다 -
    세 단계 모두 "이미 확정된 K,D로 pose만 구한다"는 동일한 문제를 푸는
    서로 다른 방법일 뿐이다).

    cv2.fisheye.solvePnP()는 내부적으로 반복 Newton 방식 undistort +
    solvePnP를 쓰는데, 실제 검출 코너(노이즈, 이미지 주변부 강한 왜곡)에서는
    합성 데이터보다 훨씬 쉽게 수렴 실패(cv2.error) 또는 ok=False를 반환한다.
    이 함수는 그 단일 실패 지점을 3단계로 완화한다:

        1) cv2.fisheye.solvePnP(obj, img, K, D) - 1차 선택지, Fisheye 왜곡
           모델을 직접 반영한다.
        2) cv2.fisheye.undistortPoints(img, K, D, P=K)로 코너를 point-wise로
           pinhole-domain pixel 좌표로 편 뒤(undistortPoints 자체도 K,D를
           고정값으로만 쓴다), distCoeffs=0인 cv2.solvePnP(ITERATIVE)로 재시도.
           1번이 발산한 코너에서도 undistort는 코너별로 독립적이라 성공할 수
           있다.
        3) 같은 undistort된 좌표에 SOLVEPNP_IPPE(평면 타겟 전용 closed-form,
           ChArUco/Chessboard는 항상 평면이므로 적용 가능)로 후보 pose(들)를
           구한 뒤, 각 후보를 cv2.fisheye.projectPoints()로 원래 Fisheye
           왜곡 도메인에 재투영해 원본 검출 코너와 RMS를 비교한다. IPPE는
           평면 타겟의 pose ambiguity 때문에 반사(mirror) 후보를 낼 수 있어,
           대상 포인트가 카메라 앞(양의 depth)에 있는 후보만 candidate로
           인정하고 그중 RMS가 가장 작은 것을 채택한다.

    Returns:
        (success, rvec, tvec, reason) - reason은 " | "로 이어붙인 단계별
        실패 사유(성공 시에는 어떤 단계에서 성공했는지)를 담은 사람이 읽을
        수 있는 요약 문자열. 세세한 로그를 켜켜이 쌓기보다는, 실패한
        frame별로 "어디서 왜 실패했는지" 한 줄로 판단할 수 있게 하는 데
        목적이 있다(ValidationResult.failed_test_frame_reasons에 그대로
        저장된다).
    """
    K = np.asarray(camera_matrix, dtype=np.float64)
    D = np.asarray(distortion, dtype=np.float64).reshape(-1, 1)
    obj64 = np.asarray(object_points, dtype=np.float64).reshape(-1, 1, 3)
    img64 = np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2)
    zero3 = np.zeros((3, 1), dtype=np.float64)

    # --- 1단계: cv2.fisheye.solvePnP 직접 -----------------------------
    try:
        ok1, rvec1, tvec1 = cv2.fisheye.solvePnP(obj64, img64, K, D)
        if ok1 and _finite_pose(rvec1, tvec1):
            return True, rvec1, tvec1, "fisheye.solvePnP"
        stage1_reason = (
            "fisheye.solvePnP: returned False" if not ok1
            else "fisheye.solvePnP: non-finite result"
        )
    except cv2.error as exc:
        stage1_reason = f"fisheye.solvePnP: {exc}"

    # --- 2/3단계 공통 준비: fisheye 왜곡을 point-wise로 제거 -----------
    # (undistortPoints는 K,D를 "그대로" 쓴다 - 재추정이 아니라 좌표 변환)
    try:
        undistorted = cv2.fisheye.undistortPoints(img64, K, D, P=K)
        undistorted = np.asarray(undistorted, dtype=np.float64).reshape(-1, 1, 2)
        if not np.all(np.isfinite(undistorted)):
            raise cv2.error("undistortPoints produced non-finite points")
    except cv2.error as exc:
        return False, zero3, zero3, f"{stage1_reason} | undistortPoints: {exc}"

    # --- 2단계: undistort된 pixel 좌표 + 일반 ITERATIVE solvePnP -------
    try:
        ok2, rvec2, tvec2 = cv2.solvePnP(
            obj64, undistorted, K, None, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if ok2 and _finite_pose(rvec2, tvec2):
            return True, rvec2, tvec2, "undistort+ITERATIVE"
        stage2_reason = (
            "undistort+ITERATIVE: returned False" if not ok2
            else "undistort+ITERATIVE: non-finite result"
        )
    except cv2.error as exc:
        stage2_reason = f"undistort+ITERATIVE: {exc}"

    # --- 3단계: planar IPPE 후보 + Fisheye 재투영 RMS로 최선 후보 선택 --
    try:
        n_solutions, rvecs, tvecs, _errs = cv2.solvePnPGeneric(
            obj64, undistorted, K, None, flags=cv2.SOLVEPNP_IPPE
        )
    except cv2.error as exc:
        return False, zero3, zero3, f"{stage1_reason} | {stage2_reason} | IPPE: {exc}"

    obj_flat = obj64.reshape(-1, 3)
    img_flat = img64.reshape(-1, 2)
    best_pose: tuple[np.ndarray, np.ndarray] | None = None
    best_rms = float("inf")
    for rvec_c, tvec_c in zip(rvecs or [], tvecs or []):
        rvec_c = np.asarray(rvec_c, dtype=np.float64).reshape(3, 1)
        tvec_c = np.asarray(tvec_c, dtype=np.float64).reshape(3, 1)
        if not _finite_pose(rvec_c, tvec_c):
            continue
        R, _ = cv2.Rodrigues(rvec_c)
        cam_depth = (R @ obj_flat.T + tvec_c).T[:, 2]
        if np.any(cam_depth <= 0):
            continue  # target이 카메라 뒤/평면에 걸치는 물리적으로 불가능한 후보
        try:
            proj, _ = cv2.fisheye.projectPoints(obj_flat.reshape(-1, 1, 3), rvec_c, tvec_c, K, D)
        except cv2.error:
            continue
        proj = proj.reshape(-1, 2)
        if not np.all(np.isfinite(proj)):
            continue
        rms = float(np.sqrt(np.mean(np.sum((proj - img_flat) ** 2, axis=1))))
        if np.isfinite(rms) and rms < best_rms:
            best_rms = rms
            best_pose = (rvec_c, tvec_c)

    if best_pose is None:
        return (
            False, zero3, zero3,
            f"{stage1_reason} | {stage2_reason} | IPPE: no positive-depth solution",
        )
    return True, best_pose[0], best_pose[1], "IPPE"


def solve_pnp_for_model_robust(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    model: CameraModelType,
) -> tuple[bool, np.ndarray, np.ndarray, str]:
    """solve_pnp_for_model()과 같은 문제(Test pose 추정, K/D는 고정하고 절대
    바꾸지 않음)를 풀지만, 실패 진단 문자열을 함께 반환하고 Fisheye에는
    3단계 fallback chain(_solve_pnp_fisheye_robust 참고)을 적용한다.

    solve_pnp_for_model()은 다른 모듈(calibration/windshield 등)에서 이미
    쓰고 있어 기존 동작(단발성 호출, 3-튜플 반환)을 그대로 유지해야 하므로
    건드리지 않는다 - 이 함수는 그 위에 추가된 새 진입점이다. Fisheye가
    아닌 모델은 fallback 없이 solve_pnp_for_model()과 동일하게 1회만
    시도한다 (이번 변경의 목적은 Fisheye Hold-out 0/12 실패 문제이지,
    다른 모델의 동작을 바꾸는 게 아니다).
    """
    if model == CameraModelType.FISHEYE:
        return _solve_pnp_fisheye_robust(object_points, image_points, camera_matrix, distortion)
    try:
        ok, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, distortion)
    except cv2.error as exc:
        return False, np.zeros((3, 1)), np.zeros((3, 1)), f"solvePnP: {exc}"
    if ok and _finite_pose(rvec, tvec):
        return True, rvec, tvec, "solvePnP"
    return False, rvec, tvec, ("solvePnP: returned False" if not ok else "solvePnP: non-finite result")


def project_points_for_model(
    object_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    model: CameraModelType,
) -> np.ndarray:
    """cv2.projectPoints vs cv2.fisheye.projectPoints 분기를 한 곳에 모은다.

    원래 radial_profile.py::_project에 있던 분기를 그대로 옮겼다 -
    calibration/windshield 패키지도 동일한 분기가 필요하다.
    """
    if model == CameraModelType.FISHEYE:
        obj = np.asarray(object_points, dtype=np.float64).reshape(1, -1, 3)
        D = np.asarray(distortion, dtype=np.float64).reshape(-1, 1)
        projected, _ = cv2.fisheye.projectPoints(obj, rvec, tvec, camera_matrix, D)
    else:
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distortion)
    return projected.reshape(-1, 2)


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
    pinhole/brown_conrady/extended_pinhole/fisheye 모델 함수를 import하는
    "상위" 모듈이라,
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
