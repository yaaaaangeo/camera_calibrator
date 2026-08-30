"""
tests/test_smoke_pipeline.py
=================================

가벼운 파이프라인 스모크 테스트 - "느린" 통합 테스트(test_pipeline_integration.py,
test_recommender_accuracy.py 등, @pytest.mark.slow)와 달리 여기는 일부러
`slow` 마커를 안 붙였다. 그래서 `pytest -m "not slow"`(기본 빠른 실행)에도
포함되어, 매번 몇 초 안에 "핵심 파이프라인 연결이 안 끊어졌는지"를 확인해준다.

빠른 이유:
  1. 이미지 렌더링/ChArUco 검출을 안 한다 - 3D->2D 직접 사영으로 코너를
     만든다(scripts/tune_model_score_weights.py와 같은 방식). 실제 검출
     알고리즘 자체의 정확성은 test_detector.py, test_pipeline_integration.py
     (느린 스위트)가 이미 커버한다 - 여기서는 "그 다음 단계들이 서로 잘
     맞물리는지"만 본다.
  2. 3모델 중 Fisheye를 뺐다 - 셋 중 계산이 가장 오래 걸리고 발산 위험도
     크다 (설계 문서 2번). Pinhole+Extended 둘만으로도 "캘리브레이션 엔진 ->
     비교 -> 검증 -> 추천" 배선이 끊어졌는지는 충분히 잡아낼 수 있다.
  3. 프레임 수를 8~10장으로 최소화하고, outlier pruning 없이 1회만 계산한다.

무거운 결과(전체 3모델 + 실제 이미지 검출 + export까지)의 정확성 자체를
검증하고 싶다면 test_pipeline_integration.py(@pytest.mark.slow)를 봐라 -
이 파일은 그걸 대체하지 않고 보완한다.
"""

from __future__ import annotations

import cv2
import numpy as np

from calibration.compare import format_comparison_table
from calibration.models.extended_pinhole import calibrate_extended_pinhole
from calibration.models.pinhole import calibrate_pinhole
from calibration.quality import analyze_dataset_quality
from calibration.recommender import compute_model_scores
from calibration.types import (
    CameraConfig,
    CameraModelType,
    Dataset,
    DetectionResult,
    Frame,
    FrameStatus,
    ImageInfo,
    PatternConfig,
    PatternType,
)
from calibration.validation import validate_all_models

W, H = 640, 480
TRUE_K = np.array([[500.0, 0, W / 2], [0, 500.0, H / 2], [0, 0, 1]])
TRUE_D = np.array([-0.2, 0.05, 0.0, 0.0, 0.0])


def _tiny_synthetic_dataset(pattern: PatternConfig, n_frames: int = 8) -> Dataset:
    """이미지 파일/검출 없이 3D->2D 직접 사영으로 프레임을 만든다 - 빠름."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    board = cv2.aruco.CharucoBoard(
        (pattern.squares_x, pattern.squares_y), pattern.square_size, pattern.marker_size, aruco_dict
    )
    pts3d = board.getChessboardCorners().astype(np.float32)
    n_corners = pts3d.shape[0]
    ids = np.arange(n_corners, dtype=np.int32).reshape(-1, 1)

    rng = np.random.default_rng(0)
    frames: list[Frame] = []
    attempts = 0
    while len(frames) < n_frames and attempts < n_frames * 10:
        attempts += 1
        rvec = (rng.random(3) - 0.5) * 0.5
        tvec = np.array([(rng.random() - 0.5) * 0.2, (rng.random() - 0.5) * 0.2, 0.35 + rng.random() * 0.2])
        proj, _ = cv2.projectPoints(pts3d.reshape(-1, 1, 3), rvec, tvec, TRUE_K, TRUE_D)
        proj = proj.reshape(-1, 2)
        if np.any(proj < 0) or np.any(proj[:, 0] > W) or np.any(proj[:, 1] > H):
            continue

        image_id = f"smoke_{len(frames):02d}"
        info = ImageInfo(image_id=image_id, path="-", width=W, height=H)
        det = DetectionResult(
            image_id=image_id, success=True,
            corners=proj.reshape(-1, 1, 2).astype(np.float32),
            object_points=pts3d.reshape(-1, 1, 3), ids=ids, num_corners=n_corners,
        )
        frames.append(Frame(image_info=info, detection=det, status=FrameStatus.DETECTED))

    return Dataset(frames=frames)


def _pattern_config() -> PatternConfig:
    return PatternConfig(
        type=PatternType.CHARUCO, squares_x=7, squares_y=5,
        square_size=0.04, marker_size=0.03, dictionary="DICT_5X5_100",
    )


def test_smoke_pinhole_and_extended_calibrate_successfully():
    """가장 기본적인 배선 확인: Detection -> 2모델 계산이 안 끊어졌는지."""
    pattern = _pattern_config()
    dataset = _tiny_synthetic_dataset(pattern)
    camera_config = CameraConfig(width=W, height=H)

    assert dataset.num_detected == len(dataset.frames)

    pinhole = calibrate_pinhole(dataset, camera_config)
    extended = calibrate_extended_pinhole(dataset, camera_config)

    assert pinhole.success, pinhole.error_message
    assert extended.success, extended.error_message
    assert pinhole.camera_matrix is not None
    assert extended.camera_matrix is not None
    assert pinhole.rms_error is not None and pinhole.rms_error >= 0
    assert extended.rms_error is not None and extended.rms_error >= 0


def test_smoke_quality_gate_does_not_crash():
    """Coverage/Diversity 분석이 작은 데이터셋에서도 안 죽는지."""
    pattern = _pattern_config()
    dataset = _tiny_synthetic_dataset(pattern)
    camera_config = CameraConfig(width=W, height=H)

    warnings = analyze_dataset_quality(dataset, camera_config)
    assert isinstance(warnings, list)
    assert dataset.coverage_grid  # 계산은 됐어야 함


def test_smoke_validation_and_recommendation_pipeline():
    """Hold-out Validation -> Model Score -> 추천까지 이어지는 배선 확인
    (Fisheye는 뺀 2모델 기준). run_all_models 대신 두 모델만 직접 계산해서
    3모델 전체를 도는 test_pipeline_integration.py보다 훨씬 빠르다.
    """
    pattern = _pattern_config()
    dataset = _tiny_synthetic_dataset(pattern, n_frames=10)
    camera_config = CameraConfig(width=W, height=H)

    calibration_results = {
        CameraModelType.PINHOLE: calibrate_pinhole(dataset, camera_config),
        CameraModelType.EXTENDED_PINHOLE: calibrate_extended_pinhole(dataset, camera_config),
    }
    assert all(r.success for r in calibration_results.values())

    validation_results = validate_all_models(dataset, camera_config, pattern, test_ratio=0.3)
    # Fisheye는 계산 안 했으니 validation_results엔 있어도 calibration_results엔 없음 -
    # compute_model_scores가 그 경우를 어떻게 다루는지도 같이 확인된다.
    scores = compute_model_scores(calibration_results, validation_results)

    eligible_scores = [s for s in scores if s.model_name in calibration_results]
    assert len(eligible_scores) == 2
    assert sum(1 for s in eligible_scores if s.is_recommended) <= 1

    # 표 렌더링도 안 죽는지 (문자열 포맷팅 배선 확인)
    table_text = format_comparison_table(list(calibration_results.values()))
    assert len(table_text) > 0
