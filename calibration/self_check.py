"""
camera_calibrator.calibration.self_check
============================================

합성 데이터(알려진 정답 카메라 파라미터)로 캘리브레이션 파이프라인의 정확도를
독립적으로 검증하는 모듈.

설계 원칙: 이 모듈은 순수 계산 로직만 담고 UI에 의존하지 않는다
(백엔드/UI 분리 원칙, README 16번 폴더 구조). GUI의 "자체 진단" 버튼
(ui/worker.py의 SelfCheckWorker)과 tests/test_calibration_accuracy.py가
여기 정의된 함수를 그대로 재사용한다 - 정확도 검증 로직이 두 곳에 중복되면
한쪽만 고치고 다른 쪽을 깜빡하는 사고가 나기 쉬우므로 반드시 한 곳에만 둔다.

--- 이 파라미터들을 고르게 된 과정 (대화 중 실측, 요약) ---

1. conftest.py의 기존 합성 픽스처(perspective-warp로만 자세를 바꿈)로
   정확도를 재보니 fx 오차가 160~270%까지 벌어졌다. 원인은 "진짜 3D 회전"이
   부족해서(Zhang's method는 이런 경우 초점거리 추정이 근본적으로 불안정) -
   그래서 여기서는 rvec/tvec을 직접 무작위로 뽑아 3D 투영한다.

2. 처음엔 목표 화면 좌표(예: 프레임 구석)를 강하게 지정해서 커버리지를
   극대화하려 했으나, 그 결과 오히려 오차가 커졌다(6~7%) - 강한 perspective
   skew가 warpPerspective 렌더링의 서브픽셀 코너 검출 정확도를 떨어뜨리기
   때문(렌더링 방법론의 한계이지 계산 로직의 문제가 아님, "순수 좌표"만으로
   재현하면 완벽했다). 그래서 화면 전역을 억지로 채우기보다 무작위 위치
   샘플링 + 이미지 장수를 늘리는 쪽으로 방향을 바꿨다.

3. object_points가 보드의 (0,0,0) 코너를 원점으로 삼기 때문에, tvec을
   그대로 무작위로 뽑으면 보드 "중심"이 아니라 보드 "모서리"가 카메라
   중심축 근처에 오게 되어 cx/cy에 수십 픽셀의 체계적 편향이 생겼다.
   그래서 tvec은 항상 "보드 중심이 카메라 좌표계에서 원하는 위치에 오도록"
   역산한다 (아래 board_center 보정 참고).

4. fx/fy는 여러 시드에서 0.4~3% 수준으로 안정적으로 수렴했지만, k1/k2
   개별 계수는 절대오차가 꽤 크게(최대 0.4) 흔들렸다 - 이건 버그가 아니라
   초점거리와 저차 방사왜곡 계수 사이의 잘 알려진 상관/모호성(fx-k1
   trade-off)이다. RMS 재투영 오차는 그런 경우에도 계속 낮게 유지된다.
   그래서 이 모듈은 k1/k2 개별 값을 pass/fail 기준에 넣지 않고 참고
   정보로만 리포트한다 - fx/fy/cx/cy/RMS가 더 안정적이고 의미 있는 지표.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from calibration.types import (
    CameraConfig,
    CameraModelType,
    Dataset,
    PatternConfig,
    PatternType,
)
from calibration.detector import detect_dataset
from calibration.models.pinhole import calibrate_pinhole
from calibration.models.extended_pinhole import calibrate_extended_pinhole

# --- 합성 데이터의 "정답" 카메라 파라미터 (임의로 고정, 재현 가능하게) ---
TRUE_FX = TRUE_FY = 1100.0
IMG_W, IMG_H = 1920, 1080
TRUE_K = np.array([[TRUE_FX, 0, IMG_W / 2], [0, TRUE_FY, IMG_H / 2], [0, 0, 1]])
TRUE_D = np.array([-0.28, 0.10, 0.0, 0.0, 0.0])  # k1, k2, p1, p2, k3

_SQUARES_X, _SQUARES_Y = 7, 5
_SQUARE_SIZE, _MARKER_SIZE = 0.04, 0.03
_DICTIONARY = "DICT_5X5_100"

DEFAULT_N_IMAGES = 30
DEFAULT_SEED = 7

# --- 통과 기준 (여러 시드에서 실측한 값을 바탕으로 여유를 두고 설정, 위 4번 참고) ---
FX_FY_ERROR_PCT_THRESHOLD = 6.0     # fx, fy는 정답의 6% 이내 (실측 최대 3% 수준 + 여유)
CENTER_ERROR_PX_THRESHOLD = 60.0    # cx, cy는 60px 이내 (1920x1080 기준, 실측 최대 ~35px + 여유)
RMS_ERROR_PX_THRESHOLD = 1.5        # 재투영 RMS (합성 데이터라 노이즈 없음에도 렌더링 오차 존재)
PINHOLE_RMS_ERROR_PX_THRESHOLD = 3.0  # Pinhole은 왜곡을 0으로 고정하므로 RMS가 구조적으로 더 큼


@dataclass
class SelfCheckResult:
    model_name: CameraModelType
    label: str
    success: bool
    passed: bool
    message: str = ""
    rms_error: float | None = None
    fx: float | None = None
    fy: float | None = None
    fx_error_pct: float | None = None
    fy_error_pct: float | None = None
    cx_error_px: float | None = None
    cy_error_px: float | None = None
    k1_error: float | None = None  # 참고용 (pass/fail에는 반영 안 함, 위 4번 이유)
    k2_error: float | None = None
    distortion_array_length: int | None = None
    free_param_count: int | None = None       # 실제로 0이 아닌(추정된) 계수 개수
    expected_free_param_count: int | None = None


def _generate_synthetic_dataset(
    n_images: int = DEFAULT_N_IMAGES, seed: int = DEFAULT_SEED
) -> tuple[Dataset, CameraConfig, PatternConfig]:
    """진짜 3D 회전을 준 ChArUco 합성 이미지를 생성해 detect까지 실행."""
    pattern_config = PatternConfig(
        type=PatternType.CHARUCO, squares_x=_SQUARES_X, squares_y=_SQUARES_Y,
        square_size=_SQUARE_SIZE, marker_size=_MARKER_SIZE, dictionary=_DICTIONARY,
    )
    camera_config = CameraConfig(width=IMG_W, height=IMG_H, sensor_name="self-check-synthetic")

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    board = cv2.aruco.CharucoBoard((_SQUARES_X, _SQUARES_Y), _SQUARE_SIZE, _MARKER_SIZE, aruco_dict)
    board_w_m, board_h_m = _SQUARES_X * _SQUARE_SIZE, _SQUARES_Y * _SQUARE_SIZE
    board_center_3d = np.array([board_w_m / 2, board_h_m / 2, 0.0])
    board_corners_3d = np.array(
        [[0, 0, 0], [board_w_m, 0, 0], [board_w_m, board_h_m, 0], [0, board_h_m, 0]], dtype=np.float32
    )
    board_img = cv2.cvtColor(board.generateImage((1400, 1000), marginSize=0), cv2.COLOR_GRAY2BGR)

    map1, map2 = cv2.initUndistortRectifyMap(TRUE_K, TRUE_D, None, TRUE_K, (IMG_W, IMG_H), cv2.CV_32FC1)

    out_dir = Path(tempfile.mkdtemp(prefix="cc_self_check_"))
    rng = np.random.default_rng(seed)
    n = 0
    attempts = 0
    while n < n_images and attempts < n_images * 8:
        attempts += 1
        rvec = rng.uniform(-0.6, 0.6, 3)
        dist_m = rng.uniform(0.55, 1.3)
        # 보드 "중심"이 카메라 좌표계에서 (x,y,dist_m)에 오도록 tvec을 역산한다.
        # (object_points의 원점이 보드 모서리라, tvec을 그대로 쓰면 모서리가
        #  카메라 중심축 근처에 와서 cx/cy에 체계적 편향이 생긴다 - 위 3번 참고)
        center_cam = np.array([rng.uniform(-0.14, 0.14), rng.uniform(-0.11, 0.11), dist_m])
        R, _ = cv2.Rodrigues(rvec)
        tvec = center_cam - R @ board_center_3d

        proj, _ = cv2.projectPoints(board_corners_3d, rvec, tvec, TRUE_K, TRUE_D)
        proj = proj.reshape(4, 2)
        if (
            proj[:, 0].min() < 0 or proj[:, 0].max() > IMG_W
            or proj[:, 1].min() < 0 or proj[:, 1].max() > IMG_H
        ):
            continue
        src = np.float32([[0, 0], [1400, 0], [1400, 1000], [0, 1000]])
        M = cv2.getPerspectiveTransform(src, proj.astype(np.float32))
        canvas = np.full((IMG_H, IMG_W, 3), 255, dtype=np.uint8)
        warped = cv2.warpPerspective(board_img, M, (IMG_W, IMG_H), borderValue=(255, 255, 255))
        mask = cv2.warpPerspective(np.full((1000, 1400), 255, dtype=np.uint8), M, (IMG_W, IMG_H))
        canvas[mask > 0] = warped[mask > 0]
        distorted = cv2.remap(canvas, map1, map2, interpolation=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
        cv2.imwrite(str(out_dir / f"img_{n:02d}.jpg"), distorted)
        n += 1

    if n < max(4, n_images // 2):
        raise RuntimeError(
            f"합성 이미지 생성에 실패했습니다 ({n}/{n_images}장만 생성됨). "
            "카메라 파라미터/보드 범위 설정을 확인하세요."
        )

    paths = sorted(str(p) for p in out_dir.glob("*.jpg"))
    dataset = detect_dataset(paths, pattern_config)
    return dataset, camera_config, pattern_config


def _evaluate(
    result, label: str, expected_free_param_count: int | None, rms_threshold: float
) -> SelfCheckResult:
    if not result.success or result.camera_matrix is None:
        return SelfCheckResult(
            model_name=result.model_name, label=label, success=False, passed=False,
            message=result.error_message or "캘리브레이션 실패",
        )

    K = result.camera_matrix
    fx, fy = float(K[0, 0]), float(K[1, 1])
    fx_err = abs(fx - TRUE_FX) / TRUE_FX * 100
    fy_err = abs(fy - TRUE_FY) / TRUE_FY * 100
    cx_err = abs(K[0, 2] - TRUE_K[0, 2])
    cy_err = abs(K[1, 2] - TRUE_K[1, 2])

    D = result.distortion.ravel() if result.distortion is not None else np.array([])
    k1_err = abs(D[0] - TRUE_D[0]) if D.size > 0 else None
    k2_err = abs(D[1] - TRUE_D[1]) if D.size > 1 else None
    free_param_count = int(np.count_nonzero(D))

    rms_ok = result.rms_error is not None and result.rms_error <= rms_threshold
    # k1/k2 개별 절대오차는 fx와의 상관/모호성 때문에 pass/fail에 넣지 않는다
    # (모듈 docstring 4번 참고) - fx/fy/cx/cy/RMS가 더 안정적인 지표다.
    passed = (
        fx_err <= FX_FY_ERROR_PCT_THRESHOLD
        and fy_err <= FX_FY_ERROR_PCT_THRESHOLD
        and cx_err <= CENTER_ERROR_PX_THRESHOLD
        and cy_err <= CENTER_ERROR_PX_THRESHOLD
        and rms_ok
        and (expected_free_param_count is None or free_param_count == expected_free_param_count)
    )

    msg = (
        f"fx={fx:.2f}(오차 {fx_err:.2f}%) fy={fy:.2f}(오차 {fy_err:.2f}%) "
        f"cx오차={cx_err:.1f}px cy오차={cy_err:.1f}px RMS={result.rms_error:.4f}px"
    )

    return SelfCheckResult(
        model_name=result.model_name, label=label, success=True, passed=passed, message=msg,
        rms_error=result.rms_error, fx=fx, fy=fy, fx_error_pct=fx_err, fy_error_pct=fy_err,
        cx_error_px=cx_err, cy_error_px=cy_err, k1_error=k1_err, k2_error=k2_err,
        distortion_array_length=int(D.size), expected_free_param_count=expected_free_param_count,
        free_param_count=free_param_count,
    )


def run_pinhole_accuracy_check(
    n_images: int = DEFAULT_N_IMAGES, seed: int = DEFAULT_SEED
) -> SelfCheckResult:
    """Pinhole(왜곡 0 고정) 모델 검증.

    Pinhole은 구조적으로 왜곡을 추정하지 않으므로, 진짜 왜곡이 있는 합성
    카메라에 맞추면 fx/fy가 어느 정도 편향되는 게 정상이다(모델 미스매치).
    그래서 fx/fy 정확도는 Extended Pinhole만큼 타이트하게 요구하지 않고,
    "계산이 정상적으로 성공했는지 + 왜곡 계수가 정말 전부 0으로 고정됐는지 +
    재투영 RMS가 비정상적으로 크지 않은지"만 확인한다.
    """
    dataset, camera_config, _ = _generate_synthetic_dataset(n_images, seed)
    result = calibrate_pinhole(dataset, camera_config)
    if not result.success or result.camera_matrix is None:
        return SelfCheckResult(
            model_name=CameraModelType.PINHOLE, label="Pinhole", success=False, passed=False,
            message=result.error_message or "캘리브레이션 실패",
        )
    D = result.distortion.ravel() if result.distortion is not None else np.array([])
    all_zero = bool(np.all(D == 0))
    rms_ok = result.rms_error is not None and result.rms_error <= PINHOLE_RMS_ERROR_PX_THRESHOLD
    K = result.camera_matrix
    fx, fy = float(K[0, 0]), float(K[1, 1])
    msg = (
        f"fx={fx:.2f} fy={fy:.2f} (참고용 - Pinhole은 왜곡 미보정이라 정답과 다소 차이날 수 있음) "
        f"distortion 전부 0={all_zero} RMS={result.rms_error:.4f}px"
    )
    return SelfCheckResult(
        model_name=CameraModelType.PINHOLE, label="Pinhole", success=True,
        passed=all_zero and rms_ok, message=msg, rms_error=result.rms_error, fx=fx, fy=fy,
        distortion_array_length=int(D.size), expected_free_param_count=0, free_param_count=int(np.count_nonzero(D)),
    )


def run_extended_pinhole_accuracy_check(
    use_rational_model: bool = False, n_images: int = DEFAULT_N_IMAGES, seed: int = DEFAULT_SEED
) -> SelfCheckResult:
    """Extended Pinhole(Rational) 정확도 검증.

    use_rational_model=True면 k1~k6,p1,p2 (자유도 8)까지 추정하는 경로(GUI의
    "Rational model 사용" 체크박스, CLI의 --rational)를 그대로 검증한다.
    배열 길이 자체는 OpenCV 버전에 따라 8이 아니라 14로 나올 수 있다는 게
    이미 확인됐으므로(대화 중 실측), free_param_count(0이 아닌 값의 개수)로
    "실제 추정된" 자유도를 판정한다.
    """
    dataset, camera_config, _ = _generate_synthetic_dataset(n_images, seed)
    result = calibrate_extended_pinhole(dataset, camera_config, use_rational_model=use_rational_model)
    label = "Extended Pinhole (Rational, 8계수)" if use_rational_model else "Extended Pinhole (5계수)"
    expected = 8 if use_rational_model else 5
    return _evaluate(result, label, expected_free_param_count=expected, rms_threshold=RMS_ERROR_PX_THRESHOLD)


def run_all_self_checks(
    n_images: int = DEFAULT_N_IMAGES, seed: int = DEFAULT_SEED
) -> list[SelfCheckResult]:
    """GUI "자체 진단" 버튼이 호출하는 진입점 - 세 가지 검증을 한 번에 실행."""
    return [
        run_pinhole_accuracy_check(n_images, seed),
        run_extended_pinhole_accuracy_check(use_rational_model=False, n_images=n_images, seed=seed),
        run_extended_pinhole_accuracy_check(use_rational_model=True, n_images=n_images, seed=seed),
    ]
