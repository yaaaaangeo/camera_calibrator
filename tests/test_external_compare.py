"""
tests/test_external_compare.py
===================================

calibration/external_compare.py - "예전 파라미터 vs 지금 구한 파라미터"
비교 기능의 회귀 테스트.

핵심으로 검증해야 할 것:
1. 진짜로 더 나쁜 파라미터가 "패배"로 정확히 판정되는지 (조작 없이).
2. 비교가 항상 "학습에 안 쓰인 test 프레임"에서만 이뤄지는지 - 내 파라미터가
   자기 학습 데이터로 스스로를 채점하는 사기가 불가능해야 한다.
3. 한 지표(RMS)만으로 승패를 가르지 않는지 - 지표가 엇갈리면 verdict가
   그 사실을 정직하게 밝히는지.
4. OpenCV YAML을 통한 외부 파라미터 불러오기가 왕복 가능한지.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from calibration.external_compare import (
    ComparisonSide,
    ExternalCameraParams,
    _build_verdict,
    compare_with_external_params,
)
from calibration.compare import run_all_models
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
from export.opencv import (
    detect_model_hint_from_opencv_yaml,
    export_opencv_yaml,
    load_camera_matrix_and_distortion_from_opencv_yaml,
)

W, H = 640, 480
TRUE_K = np.array([[500.0, 0, W / 2], [0, 500.0, H / 2], [0, 0, 1]])
TRUE_D = np.array([-0.22, 0.06, 0.0, 0.0, 0.0])


def _pattern_config() -> PatternConfig:
    return PatternConfig(
        type=PatternType.CHARUCO, squares_x=7, squares_y=5,
        square_size=0.04, marker_size=0.03, dictionary="DICT_5X5_100",
    )


def _synthetic_dataset(pattern: PatternConfig, n_frames: int = 30, seed: int = 0) -> Dataset:
    """이미지 렌더링/검출 없이 3D->2D 직접 사영으로 프레임을 만든다 (빠름) -
    test_smoke_pipeline.py와 동일한 접근."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    board = cv2.aruco.CharucoBoard(
        (pattern.squares_x, pattern.squares_y), pattern.square_size, pattern.marker_size, aruco_dict
    )
    pts3d = board.getChessboardCorners().astype(np.float32)
    n_corners = pts3d.shape[0]
    ids = np.arange(n_corners, dtype=np.int32).reshape(-1, 1)

    rng = np.random.default_rng(seed)
    frames: list[Frame] = []
    attempts = 0
    while len(frames) < n_frames and attempts < n_frames * 20:
        attempts += 1
        rvec = (rng.random(3) - 0.5) * 0.6
        tvec = np.array([(rng.random() - 0.5) * 0.25, (rng.random() - 0.5) * 0.25, 0.3 + rng.random() * 0.25])
        proj, _ = cv2.projectPoints(pts3d.reshape(-1, 1, 3), rvec, tvec, TRUE_K, TRUE_D)
        proj = proj.reshape(-1, 2)
        if np.any(proj < 0) or np.any(proj[:, 0] > W) or np.any(proj[:, 1] > H):
            continue
        image_id = f"ext_{len(frames):02d}"
        info = ImageInfo(image_id=image_id, path="-", width=W, height=H)
        det = DetectionResult(
            image_id=image_id, success=True,
            corners=proj.reshape(-1, 1, 2).astype(np.float32),
            object_points=pts3d.reshape(-1, 1, 3), ids=ids, num_corners=n_corners,
        )
        frames.append(Frame(image_info=info, detection=det, status=FrameStatus.DETECTED))

    assert len(frames) >= n_frames * 0.8, "합성 뷰가 너무 적게 생성됨"
    return Dataset(frames=frames)


@pytest.fixture(scope="module")
def camera_config() -> CameraConfig:
    return CameraConfig(width=W, height=H)


@pytest.fixture(scope="module")
def pattern_config() -> PatternConfig:
    return _pattern_config()


@pytest.fixture(scope="module")
def dataset(pattern_config) -> Dataset:
    return _synthetic_dataset(pattern_config, n_frames=30, seed=1)


@pytest.fixture(scope="module")
def my_pinhole_result(dataset, camera_config):
    results = run_all_models(dataset, camera_config, estimate_fisheye_uncertainty=False)
    return {r.model_name: r for r in results}[CameraModelType.PINHOLE]


@pytest.fixture(scope="module")
def my_validation(dataset, camera_config, pattern_config):
    results = validate_all_models(dataset, camera_config, pattern_config, test_ratio=0.3)
    return results[CameraModelType.PINHOLE]


# ---------------------------------------------------------------------------
# 핵심 시나리오: 진짜로 더 나쁜 파라미터가 실제로 패배로 판정되는지
# ---------------------------------------------------------------------------

def test_deliberately_bad_external_params_lose_on_every_metric(
    dataset, camera_config, pattern_config, my_validation,
):
    """왜곡을 아예 무시한(distortion=0) 파라미터는 실제 왜곡이 있는 이
    카메라에서 명백히 더 나빠야 한다 - 조작 없이 계산이 그렇게 판정하는지 확인."""
    bad_external = ExternalCameraParams(
        label="예전 결과(가짜, 왜곡 무시)",
        model_name=CameraModelType.PINHOLE,
        camera_matrix=TRUE_K.copy(),
        distortion=np.zeros(5),
    )
    result = compare_with_external_params(
        dataset, camera_config, pattern_config,
        CameraModelType.PINHOLE, my_validation, bad_external,
    )
    assert result.mine.success and result.external.success
    assert result.mine.test_rms < result.external.test_rms
    assert result.mine_win_count > result.external_win_count
    assert result.mine.label in result.verdict


def test_ground_truth_external_params_beat_a_deliberately_underfit_model(
    dataset, camera_config, pattern_config,
):
    """반대 방향도 확인 - 내가 캘리브레이션을 대충 했다면(왜곡 계수를 0으로
    고정), 외부의 정답에 가까운 파라미터가 이겨야 한다. '내가 만든 툴이니까
    항상 내가 이긴다'는 오해를 반증하기 위한 테스트."""
    # 일부러 왜곡 계수를 전부 고정(=0)해서 실제보다 훨씬 나쁜 "내 결과"를 만든다.
    flags = cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST
    object_points_list = [f.detection.object_points for f in dataset.enabled_frames]
    image_points_list = [f.detection.corners for f in dataset.enabled_frames]
    rms, K_bad, D_bad, _, _ = cv2.calibrateCamera(
        object_points_list, image_points_list, (W, H), None, None, flags=flags
    )

    from calibration.types import CalibrationResult, ValidationResult
    from calibration.validation import split_train_test, _subset_dataset, _test_reprojection_errors

    train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.3, seed=42)
    test_frames = _subset_dataset(dataset, test_ids).enabled_frames
    per_frame_error, _ = _test_reprojection_errors(test_frames, K_bad, D_bad, CameraModelType.PINHOLE)
    bad_test_rms = float(np.sqrt(np.mean(np.array(list(per_frame_error.values())) ** 2)))

    fake_my_validation = ValidationResult(
        train_frame_ids=train_ids, test_frame_ids=test_ids,
        test_rms=bad_test_rms, success=True,
    )
    # refit_on_train_split이 train_ids로 다시 학습할 텐데, 같은 flags가
    # 아니면 다른 값이 나온다 - 그래도 여전히 "왜곡 무시"보다는 나은 값이
    # 나올 뿐이니 external(진짜 정답)이 이기는지가 핵심. 다만 refit이
    # CALIB_FIX_K* 없이 정상 학습되면 오히려 mine이 좋아질 수 있으므로,
    # 이 테스트는 compare_with_external_params가 아니라 evaluate_side
    # 수준에서 "정답 파라미터가 왜곡-무시 파라미터를 이긴다"는 핵심만 확인한다.
    ground_truth = ExternalCameraParams(
        label="정답에 가까운 예전 결과",
        model_name=CameraModelType.PINHOLE,
        camera_matrix=TRUE_K.copy(),
        distortion=TRUE_D.copy(),
    )
    from calibration.external_compare import _evaluate_side

    truth_side = _evaluate_side(
        dataset, camera_config, pattern_config, test_ids,
        ground_truth.camera_matrix, ground_truth.distortion, ground_truth.model_name, ground_truth.label,
    )
    assert truth_side.success
    assert truth_side.test_rms < bad_test_rms, (
        "실제 정답에 가까운 파라미터가 왜곡을 무시한 파라미터보다 test RMS가 낮아야 함"
    )


# ---------------------------------------------------------------------------
# 공정성 - 같은 test 분할 재사용 + 내부 정합성 체크
# ---------------------------------------------------------------------------

def test_mine_side_reuses_the_same_holdout_split_as_validation(
    dataset, camera_config, pattern_config, my_validation,
):
    """재계산된 mine.test_rms가 기존 Hold-out(my_validation.test_rms)과
    (거의) 같아야 한다 - 그래야 '같은 절차를 그대로 재현했다'는 신뢰가 생긴다.
    """
    external = ExternalCameraParams(
        label="비교용", model_name=CameraModelType.PINHOLE,
        camera_matrix=TRUE_K.copy(), distortion=np.zeros(5),
    )
    result = compare_with_external_params(
        dataset, camera_config, pattern_config,
        CameraModelType.PINHOLE, my_validation, external,
    )
    assert result.mine.test_rms == pytest.approx(my_validation.test_rms, abs=1e-2)
    # 정합성이 맞으면 caveat에 "다릅니다" 경고가 없어야 한다.
    assert not any("다릅니다" in c for c in result.caveats)


def test_no_test_frames_returns_explanatory_verdict(dataset, camera_config, pattern_config):
    from calibration.types import ValidationResult

    empty_validation = ValidationResult(train_frame_ids=[], test_frame_ids=[], success=True)
    external = ExternalCameraParams(
        label="비교용", model_name=CameraModelType.PINHOLE,
        camera_matrix=TRUE_K.copy(), distortion=TRUE_D.copy(),
    )
    result = compare_with_external_params(
        dataset, camera_config, pattern_config,
        CameraModelType.PINHOLE, empty_validation, external,
    )
    assert not result.mine.success
    assert "test 프레임" in result.verdict or "이미지를 더" in result.verdict


# ---------------------------------------------------------------------------
# verdict 문구 - 지표가 엇갈리는 경우 정직하게 말하는지
# ---------------------------------------------------------------------------

def test_verdict_reports_full_sweep_when_all_metrics_agree():
    mine = ComparisonSide(label="내 결과", test_rms=0.3, edge_rms=0.4, straightness_residual=0.2, success=True)
    external = ComparisonSide(label="예전 결과", test_rms=0.5, edge_rms=0.6, straightness_residual=0.4, success=True)
    verdict = _build_verdict(mine, external, mine_win=10, external_win=2, n_common=12)
    assert "내 결과" in verdict
    assert "전부" in verdict


def test_verdict_reports_mixed_signal_honestly():
    mine = ComparisonSide(label="내 결과", test_rms=0.3, edge_rms=0.9, straightness_residual=0.2, success=True)
    external = ComparisonSide(label="예전 결과", test_rms=0.5, edge_rms=0.4, straightness_residual=0.4, success=True)
    verdict = _build_verdict(mine, external, mine_win=6, external_win=6, n_common=12)
    assert "엇갈" in verdict


def test_verdict_reports_failure_reason_when_side_fails():
    mine = ComparisonSide(label="내 결과", success=True, test_rms=0.3)
    external = ComparisonSide(label="예전 결과", success=False, error_message="pose 추정 실패")
    verdict = _build_verdict(mine, external, mine_win=0, external_win=0, n_common=0)
    assert "예전 결과" in verdict
    assert "pose 추정 실패" in verdict


# ---------------------------------------------------------------------------
# OpenCV YAML을 통한 외부 파라미터 불러오기
# ---------------------------------------------------------------------------

def test_load_camera_matrix_and_distortion_roundtrip(tmp_path, my_pinhole_result, camera_config, pattern_config):
    path = str(tmp_path / "camera.yaml")
    export_opencv_yaml(my_pinhole_result, camera_config, pattern_config, path)

    K, D = load_camera_matrix_and_distortion_from_opencv_yaml(path)
    assert K.shape == (3, 3)
    np.testing.assert_allclose(K, my_pinhole_result.camera_matrix, rtol=1e-6)
    np.testing.assert_allclose(D.reshape(-1), my_pinhole_result.distortion.reshape(-1), rtol=1e-6)

    hint = detect_model_hint_from_opencv_yaml(path)
    assert hint == CameraModelType.PINHOLE


def test_detect_model_hint_returns_none_for_foreign_yaml(tmp_path):
    """이 툴이 만들지 않은(calibration_model 필드가 없는) YAML도 읽을 수는
    있어야 하지만, 모델 종류는 억지로 추측하지 않고 None을 돌려줘야 한다."""
    path = str(tmp_path / "foreign.yaml")
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", TRUE_K)
    fs.write("distortion_coefficients", TRUE_D)
    fs.release()

    K, D = load_camera_matrix_and_distortion_from_opencv_yaml(path)
    np.testing.assert_allclose(K, TRUE_K)
    assert detect_model_hint_from_opencv_yaml(path) is None


def test_load_rejects_yaml_without_camera_matrix(tmp_path):
    path = str(tmp_path / "invalid.yaml")
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    fs.write("something_else", 1.0)
    fs.release()

    with pytest.raises(ValueError, match="camera_matrix"):
        load_camera_matrix_and_distortion_from_opencv_yaml(path)
