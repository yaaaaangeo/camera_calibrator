"""
tests/conftest.py
====================

여러 테스트가 공통으로 필요로 하는 "왜곡이 있는 합성 ChArUco 데이터셋"을
세션 스코프 fixture로 한 번만 만들어 재사용한다. 매 테스트 함수마다 이미지를
새로 렌더링하면 테스트 스위트 전체가 느려지므로, 무거운 합성 데이터 생성은
세션당 한 번으로 제한한다.

이 fixture들은 대화 중 직접 손으로 돌렸던 검증 스크립트
(_manual_test_frame_quality.py, _manual_test_radial_profile.py,
_manual_test_straightness.py 등)의 합성 데이터 생성 로직을 그대로 재사용한
것이다 - 그때는 한 번 쓰고 지웠지만, 이제 정식으로 fixture화한다.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from calibration.types import (
    CameraConfig,
    PatternConfig,
    PatternType,
)

# ---------------------------------------------------------------------------
# 패턴/카메라 설정
# ---------------------------------------------------------------------------

CHARUCO_SQUARES_X = 7
CHARUCO_SQUARES_Y = 5
CHARUCO_SQUARE_SIZE = 0.04
CHARUCO_MARKER_SIZE = 0.03
CHARUCO_DICT = "DICT_5X5_100"

IMG_W, IMG_H = 1920, 1080

# 의도적으로 눈에 띄는 방사 왜곡을 준 "진짜" 카메라 파라미터 - 여러 테스트가
# "이 왜곡을 정확히 되돌렸는가"를 검증하는 기준(ground truth)으로 쓴다.
TRUE_FX = TRUE_FY = 1100.0
TRUE_K = np.array([[TRUE_FX, 0, IMG_W / 2], [0, TRUE_FY, IMG_H / 2], [0, 0, 1]])
TRUE_D = np.array([-0.28, 0.10, 0.0, 0.0, 0.0])


@pytest.fixture(scope="session")
def pattern_config() -> PatternConfig:
    return PatternConfig(
        type=PatternType.CHARUCO,
        squares_x=CHARUCO_SQUARES_X,
        squares_y=CHARUCO_SQUARES_Y,
        square_size=CHARUCO_SQUARE_SIZE,
        marker_size=CHARUCO_MARKER_SIZE,
        dictionary=CHARUCO_DICT,
    )


@pytest.fixture(scope="session")
def camera_config() -> CameraConfig:
    return CameraConfig(width=IMG_W, height=IMG_H, sensor_name="pytest-synthetic")


@pytest.fixture(scope="session")
def charuco_board():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    return cv2.aruco.CharucoBoard(
        (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y), CHARUCO_SQUARE_SIZE, CHARUCO_MARKER_SIZE, aruco_dict
    )


# ---------------------------------------------------------------------------
# 합성 이미지 데이터셋 (왜곡 있음) - 파이프라인 통합 테스트용
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synthetic_distorted_dataset_dir(tmp_path_factory, charuco_board) -> str:
    """실제 렌즈 왜곡(TRUE_K, TRUE_D)을 적용한 합성 ChArUco 사진 여러 장을
    만들어 디스크에 저장하고 폴더 경로를 반환한다.

    detect_dataset()에 그대로 넘길 수 있는, 실제 파일 기반 데이터셋이다.
    """
    out_dir = tmp_path_factory.mktemp("synthetic_charuco")

    base_img = cv2.cvtColor(charuco_board.generateImage((1400, 1000), marginSize=40), cv2.COLOR_GRAY2BGR)
    map1, map2 = cv2.initUndistortRectifyMap(TRUE_K, TRUE_D, None, TRUE_K, (IMG_W, IMG_H), cv2.CV_32FC1)

    rng = np.random.default_rng(42)
    n_images = 16
    for i in range(n_images):
        scale = 0.35 + rng.random() * 0.35
        bw, bh = int(1400 * scale), int(1000 * scale)
        board_small = cv2.resize(base_img, (bw, bh))
        canvas = np.full((IMG_H, IMG_W, 3), 255, dtype=np.uint8)
        x0 = int(rng.integers(0, max(IMG_W - bw, 1)))
        y0 = int(rng.integers(0, max(IMG_H - bh, 1)))
        src = np.float32([[0, 0], [bw, 0], [bw, bh], [0, bh]])
        jitter = 0.06 * bw
        dst = src + rng.uniform(-jitter, jitter, src.shape).astype(np.float32)
        dst[:, 0] += x0
        dst[:, 1] += y0
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(board_small, M, (IMG_W, IMG_H), borderValue=(255, 255, 255))
        mask = cv2.warpPerspective(np.full((bh, bw), 255, dtype=np.uint8), M, (IMG_W, IMG_H))
        canvas[mask > 0] = warped[mask > 0]

        distorted = cv2.remap(canvas, map1, map2, interpolation=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
        cv2.imwrite(str(out_dir / f"img_{i:02d}.jpg"), distorted)

    return str(out_dir)


@pytest.fixture(scope="session")
def synthetic_dataset(synthetic_distorted_dataset_dir, pattern_config):
    """위 이미지 폴더를 실제로 detect_dataset()에 넣어 검출까지 끝낸 Dataset.
    세션 내 여러 테스트가 검출을 반복하지 않도록 한 번만 계산해 공유한다.
    """
    import glob
    from calibration.detector import detect_dataset

    paths = sorted(glob.glob(f"{synthetic_distorted_dataset_dir}/*.jpg"))
    return detect_dataset(paths, pattern_config)
